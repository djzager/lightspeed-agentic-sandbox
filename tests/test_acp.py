"""Offline transport concurrency tests with no model network calls."""

import asyncio
import json
import os
import signal
import sys
from contextlib import asynccontextmanager, suppress
from types import SimpleNamespace

import pytest
from websockets.asyncio.client import connect
from websockets.exceptions import InvalidStatus

from lightspeed_agentic.acp.config import ACPConfig, AskUserConfig
from lightspeed_agentic.acp.server import serve
from lightspeed_agentic.session import RuntimeSession, ToolPermissionRequest, UserInputRequest
from lightspeed_agentic.types import (
    ProviderQueryOptions,
    UsageEvent,
    UsageSnapshot,
    UserMessageEvent,
)

pytestmark = pytest.mark.asyncio


class FakeSession(RuntimeSession):
    """Real session lifecycle with a deterministic native transport-test graph."""

    def __init__(self, provider, options, **callbacks):
        super().__init__(provider, options, **callbacks)
        self.callbacks = callbacks
        self.cancelled = asyncio.Event()
        self.steering = asyncio.Queue()
        self.usage = UsageSnapshot()
        self.prompts = []
        self.mode = "auto"
        self.closed = False
        self.graph = FakeGraph(self)
        self.prepare_calls = 0
        self.pending_input = None

    async def prepare_graph(self):
        self.prepare_calls += 1
        self._resources = {"agent": self.graph}
        return self.graph

    async def prompt(self, *_args, **_kwargs):
        raise AssertionError("ACP must drive the native graph through the official server")

    async def steer(self, text):
        await self.steering.put((text, "steer-message-1"))
        return "steer-message-1"

    async def cancel(self):
        self.cancelled.set()
        await super().cancel()

    async def aclose(self):
        self.closed = True
        await super().aclose()


class FakeGraph:
    """Emit native LangGraph frames; never reconstruct provider events."""

    def __init__(self, session):
        self.session = session
        self.checkpointer = object()
        self.invocations = []

    async def aget_state(self, _config):
        return SimpleNamespace(interrupts=(), metadata={}, values={"messages": []})

    async def astream(self, value, *, config, stream_mode, subgraphs):
        from langchain_core.messages import AIMessageChunk, ToolMessage

        from lightspeed_agentic.session import SessionTurnLimitError

        assert stream_mode == ["messages", "updates", "custom"]
        assert subgraphs is True
        assert config["configurable"]["thread_id"] == self.session.session_id
        assert len(config["callbacks"]) == 1
        self.invocations.append((value, config))
        prompt = "\n".join(block["text"] for block in value["messages"][0]["content"])
        session = self.session
        session.prompts.append((prompt, session._max_turns))
        session.cancelled.clear()
        if prompt == "secret failure":
            raise RuntimeError("credential-supersecret model-content-supersecret")
        if session.pending_input == "permission":
            await session.permission_callback(ToolPermissionRequest("execute", {}, "pending-1"))
        elif session.pending_input == "human":
            await session.user_input_callback(
                UserInputRequest("Choose database", ("postgres", "mysql"))
            )
            yield (), "messages", (AIMessageChunk(content="must only continue with an answer"), {})
        elif prompt in ("wait", "steer"):
            yield (
                (),
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": "execute",
                                "args": '{"command":"safe"}',
                                "id": "call-1",
                                "index": 0,
                            }
                        ],
                    ),
                    {},
                ),
            )
            if prompt == "wait":
                await asyncio.Future()
            guidance, message_id = await session.steering.get()
            session._event_sink(UserMessageEvent(text=guidance, message_id=message_id))
            yield (), "messages", (AIMessageChunk(content=guidance), {})
        elif prompt == "approve":
            allowed = await session.permission_callback(
                ToolPermissionRequest("execute", {"command": "safe"}, "approval-1")
            )
            yield (
                (),
                "messages",
                (
                    AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": "execute",
                                "args": '{"command":"safe"}',
                                "id": "approval-1",
                                "index": 0,
                            }
                        ],
                    ),
                    {},
                ),
            )
            result = ToolMessage(
                content="allowed" if allowed else "denied",
                name="execute",
                tool_call_id="approval-1",
            )
            yield (), "messages", (result, {})
            from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware

            signature = ToolResultInspectionMiddleware._signature(
                "execute", "result", "approval-1", result.content
            )
            yield (), "custom", {"lightspeed.accepted_tool_results": {signature}}
        elif prompt == "ask":
            answer = await session.user_input_callback(
                UserInputRequest("Choose database", ("postgres", "mysql"), "ask-1")
            )
            yield (), "messages", (AIMessageChunk(content=answer), {})
        else:
            yield (), "messages", (AIMessageChunk(content=f"reply {len(session.prompts)}"), {})
        session.turns_used += 1
        session.usage = UsageSnapshot(
            input_tokens=len(session.prompts) * 10, output_tokens=len(session.prompts) * 2
        )
        session._event_sink(UsageEvent(usage=session.usage))
        if prompt == "limit":
            raise SessionTurnLimitError()


class FakeProvider:
    def __init__(self):
        self.sessions = []
        self.options = []
        self._model_route = SimpleNamespace(model="offline", telemetry_provider="offline")

    def create_session(self, _options, **callbacks):
        self.options.append(_options)
        session = FakeSession(self, _options, **callbacks)
        self.sessions.append(session)
        return session


@asynccontextmanager
async def server(
    tmp_path,
    *,
    ask_user=None,
    timeout_seconds=5,
    provider_override=None,
    mode="auto",
    config_overrides=None,
    options_overrides=None,
):
    config = ACPConfig(
        cwd=str(tmp_path), max_turns=7, ask_user=ask_user, mode=mode, **(config_overrides or {})
    )
    options = ProviderQueryOptions(
        prompt="",
        system_prompt="rules",
        max_turns=7,
        cwd=str(tmp_path),
        **(options_overrides or {}),
    )
    provider = provider_override or FakeProvider()
    listening = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        serve(
            provider,
            config,
            options,
            host="127.0.0.1",
            port=0,
            secret="test-secret",  # noqa: S106 -- local test transport only
            timeout_seconds=timeout_seconds,
            on_listening=listening.set_result,
        )
    )
    port = await asyncio.wait_for(listening, 5)
    try:
        yield f"ws://127.0.0.1:{port}/acp?token=test-secret", provider, config
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


async def send(ws, request_id, method, params):
    await ws.send(
        json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
    )


async def receive(ws):
    return json.loads(await asyncio.wait_for(ws.recv(), 5))


async def response(ws, request_id):
    notifications = []
    while True:
        frame = await receive(ws)
        if frame.get("id") == request_id:
            return frame, notifications
        notifications.append(frame)


async def initialize(ws, config):
    await send(ws, 1, "initialize", {"protocolVersion": 1})
    init, _ = await response(ws, 1)
    assert init["result"]["protocolVersion"] == 1
    await send(ws, 2, "session/new", {"cwd": config.cwd, "mcpServers": []})
    new, _ = await response(ws, 2)
    return new["result"]["sessionId"]


def prompt(session_id, text):
    return {"sessionId": session_id, "prompt": [{"type": "text", "text": text}]}


async def test_two_prompts_cumulative_usage(tmp_path, monkeypatch):
    from deepagents_acp.server import AgentServerACP

    official_graphs = []
    official_prompt = AgentServerACP.prompt

    async def record_official_prompt(agent, *args, **kwargs):
        official_graphs.append(agent._agent)
        return await official_prompt(agent, *args, **kwargs)

    monkeypatch.setattr(AgentServerACP, "prompt", record_official_prompt)
    async with server(tmp_path) as (url, provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        for request_id, text in ((3, "first"), (4, "second")):
            await send(ws, request_id, "session/prompt", prompt(session_id, text))
            result, updates = await response(ws, request_id)
            assert result["result"] == {
                "stopReason": "end_turn",
                "_meta": {"lightspeed": {"turnsUsed": 1}},
            }
            usage = [
                u["params"]["update"]
                for u in updates
                if u["params"]["update"]["sessionUpdate"] == "usage_update"
            ]
            assert usage[-1]["inputTokens"] == (request_id - 2) * 10
            assert usage[-1]["outputTokens"] == (request_id - 2) * 2
            assert "cost" not in usage[-1]
            assert all(
                u["method"] == "_lightspeed/session/update"
                for u in updates
                if u["params"]["update"]["sessionUpdate"] == "usage_update"
            )
        assert len(provider.sessions) == 1
        assert provider.sessions[0].prompts == [("first", 7), ("second", 7)]
        assert provider.sessions[0].prepare_calls == 2
        assert len(provider.sessions[0].graph.invocations) == 2
        assert [graph.graph for graph in official_graphs] == [provider.sessions[0].graph] * 2


async def test_cancel_running_prompt_without_blocking_reader(tmp_path):
    async with server(tmp_path) as (url, _provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "wait"))
        await receive(ws)
        tool = await receive(ws)
        assert tool["params"]["update"]["toolCallId"] == "call-1"
        await ws.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}}
            )
        )
        result, _ = await response(ws, 3)
        assert result["result"]["stopReason"] == "cancelled"
        await send(ws, 4, "session/prompt", prompt(session_id, "again"))
        result, _ = await response(ws, 4)
        assert result["result"]["stopReason"] == "end_turn"


@pytest.mark.parametrize(
    ("outcome", "expected"),
    [
        ({"outcome": "selected", "optionId": "allow"}, "allowed"),
        ({"outcome": "selected", "optionId": "bogus"}, "denied"),
        ({"outcome": "cancelled"}, "denied"),
    ],
)
async def test_permission_reply_during_prompt(tmp_path, outcome, expected):
    async with server(tmp_path) as (url, _provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "approve"))
        await receive(ws)
        ask = await receive(ws)
        assert ask["method"] == "session/request_permission"
        assert ask["params"]["toolCall"]["toolCallId"] == "approval-1"
        await ws.send(
            json.dumps({"jsonrpc": "2.0", "id": ask["id"], "result": {"outcome": outcome}})
        )
        result, updates = await response(ws, 3)
        assert "error" not in result
        tool = next(
            frame["params"]["update"]
            for frame in updates
            if frame["params"]["update"]["sessionUpdate"] == "tool_call_update"
        )
        assert tool["toolCallId"] == "approval-1"
        assert tool["content"][0]["content"]["text"] == (
            f"**Command:**\n```bash\nsafe\n```\n\n**Output:**\n```\n{expected}\n```\n"
        )


async def test_steer_matches_active_run_and_preserves_message_id(tmp_path):
    async with server(tmp_path) as (url, _provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "steer"))
        metadata = await receive(ws)
        run_id = metadata["params"]["update"]["_meta"]["goose"]["runId"]
        await receive(ws)
        await send(
            ws,
            4,
            "_goose/unstable/session/steer",
            {**prompt(session_id, "use postgres"), "expectedRunId": run_id},
        )
        frames = [await receive(ws) for _ in range(6)]
        steer_reply = next(frame for frame in frames if frame.get("id") == 4)
        assert steer_reply["result"]["messageId"] == "steer-message-1"
        user = next(
            frame
            for frame in frames
            if frame.get("params", {}).get("update", {}).get("sessionUpdate")
            == "user_message_chunk"
        )
        assert user["params"]["update"]["_meta"]["goose"]["messageId"] == "steer-message-1"


async def test_turn_limit_standard_stop_reason(tmp_path):
    async with server(tmp_path) as (url, _provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "limit"))
        result, _ = await response(ws, 3)
        assert result["result"]["stopReason"] == "max_turn_requests"


async def test_launcher_approval_policy_cannot_be_reduced(tmp_path):
    async with server(tmp_path, mode="approve") as (url, provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/set_mode", {"sessionId": session_id, "modeId": "auto"})
        assert (await receive(ws))["error"]["code"] == -32602
        assert provider.sessions[0].mode == "approve"
        await send(ws, 4, "session/set_mode", {"sessionId": session_id, "modeId": "approve"})
        assert (await receive(ws))["result"] == {}


@pytest.mark.parametrize(("requested", "expected"), [(2, 2), (100, 7)])
async def test_prompt_turn_budget_is_clamped_to_launcher_ceiling(tmp_path, requested, expected):
    async with server(tmp_path) as (url, provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(
            ws,
            3,
            "session/prompt",
            {**prompt(session_id, "handoff"), "_meta": {"lightspeed": {"maxTurns": requested}}},
        )
        result, _ = await response(ws, 3)
        assert result["result"]["stopReason"] == "end_turn"
        assert provider.sessions[0].prompts == [("handoff", expected)]


@pytest.mark.parametrize("requested", [True, 0, -1, "2", None])
async def test_invalid_prompt_turn_budget_is_rejected_before_execution(tmp_path, requested):
    async with server(tmp_path) as (url, provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(
            ws,
            3,
            "session/prompt",
            {**prompt(session_id, "handoff"), "_meta": {"lightspeed": {"maxTurns": requested}}},
        )
        assert (await receive(ws))["error"]["code"] == -32602
        assert provider.sessions[0].prompts == []


async def test_viewer_load_replays_history_and_disconnect_preserves_owner(tmp_path):
    async with server(tmp_path) as (url, provider, config), connect(url) as owner:
        session_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(session_id, "wait"))
        await receive(owner)
        await receive(owner)
        # Tee private upstream pipes authenticate by header, without a URL token.
        async with connect(
            url.split("?")[0], additional_headers={"X-Secret-Key": "test-secret"}
        ) as viewer:
            await send(viewer, 1, "initialize", {"protocolVersion": 1})
            result, _ = await response(viewer, 1)
            assert result["result"]["agentCapabilities"]["loadSession"] is True
            await send(
                viewer,
                2,
                "session/load",
                {"sessionId": session_id, "cwd": config.cwd, "mcpServers": []},
            )
            result, updates = await response(viewer, 2)
            assert result["result"] == {}
            assert updates[0]["params"]["update"]["content"]["text"] == "wait"
            assert updates[-1]["params"]["update"]["toolCallId"] == "call-1"
            await send(viewer, 3, "session/prompt", prompt(session_id, "untrusted owner"))
            assert (await receive(viewer))["error"]["code"] == -32000
        assert not provider.sessions[0].cancelled.is_set()
        await owner.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}}
            )
        )
        result, _ = await response(owner, 3)
        assert result["result"]["stopReason"] == "cancelled"


async def test_replay_preserves_cumulative_usage_extension_method(tmp_path):
    async with server(tmp_path) as (url, _provider, config), connect(url) as owner:
        session_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(session_id, "first"))
        await response(owner, 3)
        async with connect(url) as viewer:
            await send(viewer, 1, "initialize", {"protocolVersion": 1})
            await response(viewer, 1)
            await send(
                viewer,
                2,
                "session/load",
                {"sessionId": session_id, "cwd": config.cwd, "mcpServers": []},
            )
            result, updates = await response(viewer, 2)
            assert result["result"] == {}
            usage_frames = [
                frame
                for frame in updates
                if frame["params"]["update"]["sessionUpdate"] == "usage_update"
            ]
            assert usage_frames
            assert all(frame["method"] == "_lightspeed/session/update" for frame in usage_frames)
            assert usage_frames[-1]["params"]["update"]["inputTokens"] == 10


class AwaitingPermissionProvider(FakeProvider):
    """Real RuntimeSession cancellation around an offline suspended provider call."""

    def create_session(self, options, **callbacks):
        session = super().create_session(options, **callbacks)
        session.pending_input = "permission"
        return session


class AwaitingHumanProvider(AwaitingPermissionProvider):
    def create_session(self, options, **callbacks):
        session = super().create_session(options, **callbacks)
        session.pending_input = "human"
        return session


async def test_cancel_while_permission_response_pending(tmp_path):
    async with (
        server(tmp_path, provider_override=AwaitingPermissionProvider()) as (
            url,
            _provider,
            config,
        ),
        connect(url) as ws,
    ):
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "approve"))
        await receive(ws)
        ask = await receive(ws)
        assert ask["method"] == "session/request_permission"
        await ws.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": session_id}}
            )
        )
        result, _ = await response(ws, 3)
        assert result["result"]["stopReason"] == "cancelled"
        assert result["result"]["_meta"]["lightspeed"]["turnsUsed"] == 0
        # A stale answer after cancellation is inert; the reader remains usable.
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": ask["id"],
                    "result": {"outcome": {"outcome": "selected", "optionId": "allow"}},
                }
            )
        )
        await send(ws, 4, "session/set_mode", {"sessionId": session_id, "modeId": "auto"})
        assert (await receive(ws))["result"] == {}


async def test_fail_closed_inputs_auth_and_payload_free_errors(tmp_path, capsys, caplog):
    async with server(tmp_path) as (url, provider, config):
        with pytest.raises(InvalidStatus):
            async with connect(url.replace("test-secret", "wrong")):
                pass
        async with connect(url) as ws:
            await send(ws, 1, "initialize", {"protocolVersion": True})
            assert (await receive(ws))["error"]["code"] == -32602
            session_id = await initialize(ws, config)
            await send(
                ws,
                3,
                "session/new",
                {
                    "cwd": config.cwd,
                    "mcpServers": [
                        {"name": "untrusted", "command": "/bin/sh", "args": [], "env": []}
                    ],
                },
            )
            assert (await receive(ws))["error"]["code"] == -32602
            await send(
                ws,
                4,
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "image", "data": "untrusted"}]},
            )
            assert (await receive(ws))["error"]["code"] == -32602
            assert provider.sessions[0].prompts == []
            await send(ws, 5, "session/prompt", prompt(session_id, "secret failure"))
            result, _ = await response(ws, 5)
            assert result["error"]["message"] == "runtime operation failed"
    assert capsys.readouterr().out == ""
    assert "supersecret" not in caplog.text


async def test_sigterm_cleans_active_session_and_provider(tmp_path):
    script = tmp_path / "signal_runtime.py"
    script.write_text("""import asyncio, sys
from pathlib import Path
from types import SimpleNamespace
import lightspeed_agentic.acp.__main__ as main
import lightspeed_agentic.config as config
import lightspeed_agentic.mcp as mcp
import lightspeed_agentic.providers.deepagents as deepagents
import lightspeed_agentic.providers.model_router as router
import lightspeed_agentic.readiness as readiness
import lightspeed_agentic.tls as tls
from lightspeed_agentic.acp.config import ACPConfig
from lightspeed_agentic.session import RuntimeSession
from langchain_core.messages import AIMessageChunk
root = Path(sys.argv[1])
config.resolve_model_family = lambda: SimpleNamespace()
config.parse_reasoning_config = lambda: None
config.parse_tool_output_inspection_enabled = lambda: True
config.parse_agent_timeout = lambda: 30
router.resolve_model_route = lambda **kwargs: SimpleNamespace()
readiness.run_readiness_checks = lambda *args: (True, {})
tls.configure_tls = lambda: None
mcp.parse_mcp_servers = lambda: []
async def admit(servers): return []
mcp.discover_and_admit_mcp_servers = admit
mcp.split_admitted_mcp_servers = lambda servers: ([], [])
mcp.render_mcp_policy_context = lambda policies: ""
class Provider:
    _model_route = SimpleNamespace(model="offline", telemetry_provider="offline")
    def create_session(self, options, **callbacks):
        return RuntimeSession(self, options, **callbacks)
    async def prepare_graph(self, options, *, session):
        if not session._resources:
            session._resources = {"agent": Graph()}
        return session._resources
    async def aclose(self):
        (root / "provider-closed").write_text("yes")
class Graph:
    checkpointer = True
    async def aget_state(self, config):
        return SimpleNamespace(interrupts=(), values={"messages": []})
    async def astream(self, value, **kwargs):
        try:
            yield (), "messages", (AIMessageChunk(content="", tool_call_chunks=[{
                "name": "execute", "args": "{}", "id": "active-tool", "index": 0
            }]), {})
            await asyncio.Future()
        finally:
            (root / "session-closed").write_text("yes")
deepagents.DeepAgentsProvider = lambda route: Provider()
original_serve = main.serve
async def serve(*args, **kwargs):
    kwargs["on_listening"] = lambda port: (root / "port").write_text(str(port))
    kwargs["secret"] = "test-secret"
    await original_serve(*args, **kwargs)
main.serve = serve
try:
    asyncio.run(main.run(ACPConfig(cwd=str(root), max_turns=7), host="127.0.0.1", port=0))
except asyncio.CancelledError:
    pass
""")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(script),
        str(tmp_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        async with asyncio.timeout(10):
            while not (tmp_path / "port").exists():  # noqa: ASYNC110 -- separate process readiness
                await asyncio.sleep(0.02)
        port = (tmp_path / "port").read_text()
        async with connect(f"ws://127.0.0.1:{port}/acp?token=test-secret") as ws:
            config = ACPConfig(cwd=str(tmp_path), max_turns=7)
            session_id = await initialize(ws, config)
            await send(ws, 3, "session/prompt", prompt(session_id, "wait"))
            await receive(ws)
            assert (await receive(ws))["params"]["update"]["toolCallId"] == "active-tool"
            process.send_signal(signal.SIGTERM)
            stdout, stderr = await asyncio.wait_for(process.communicate(), 5)
            assert process.returncode == 0, stderr.decode()
            assert stdout == b""
        assert (tmp_path / "session-closed").read_text() == "yes"
        assert (tmp_path / "provider-closed").read_text() == "yes"
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.parametrize(
    ("action", "answer", "success"),
    [
        ("accept", "postgres", True),
        ("cancel", None, False),
        ("accept", "invalid", False),
        ("accept", "", False),
    ],
)
async def test_trusted_stdio_ask_user_round_trip(tmp_path, action, answer, success):
    script = tmp_path / "ask_user.py"
    script.write_text("""from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel
mcp = FastMCP("trusted-ask-user")
class Answer(BaseModel):
    answer: str
@mcp.tool()
async def ask_user(question: str, options: list[str], ctx: Context) -> str:
    result = await ctx.elicit(question, Answer)
    return "The human answered: " + result.data.answer if result.action == "accept" else "No answer"
mcp.run(transport="stdio")
""")
    trusted = AskUserConfig(command=sys.executable, args=[str(script)])
    async with server(tmp_path, ask_user=trusted) as (url, _provider, config), connect(url) as ws:
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "ask"))
        await receive(ws)
        ask = await receive(ws)
        assert ask["method"] == "elicitation/create"
        assert ask["params"]["sessionId"] == session_id
        await ws.send(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": ask["id"],
                    "result": {
                        "action": action,
                        "content": {"answer": answer} if answer is not None else None,
                    },
                }
            )
        )
        result, updates = await response(ws, 3)
        if success:
            assert result["result"]["stopReason"] == "end_turn"
            assert "postgres" in updates[0]["params"]["update"]["content"]["text"]
        else:
            assert result["error"]["message"] == "runtime operation failed"


@pytest.mark.parametrize("cancel_prompt", [True, False])
async def test_real_session_pending_stdio_elicitation_is_fail_closed_and_reaped(
    tmp_path, cancel_prompt
):
    script = tmp_path / "pending_ask.py"
    script.write_text("""import os
from pathlib import Path
from mcp.server.fastmcp import FastMCP, Context
from pydantic import BaseModel
Path(__file__).with_suffix(".pid").write_text(str(os.getpid()))
mcp = FastMCP("trusted-ask-user")
class Answer(BaseModel):
    answer: str
@mcp.tool()
async def ask_user(question: str, options: list[str], ctx: Context) -> str:
    result = await ctx.elicit(question, Answer)
    return result.data.answer if result.action == "accept" else "No answer"
mcp.run(transport="stdio")
""")
    trusted = AskUserConfig(command=sys.executable, args=[str(script)])
    async with (
        server(tmp_path, ask_user=trusted, provider_override=AwaitingHumanProvider()) as (
            url,
            _provider,
            config,
        ),
        connect(url) as ws,
    ):
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "ask"))
        await receive(ws)
        ask = await receive(ws)
        assert ask["method"] == "elicitation/create"
        helper_pid = int(script.with_suffix(".pid").read_text())
        if cancel_prompt:
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": session_id},
                    }
                )
            )
        else:
            # The headless harness returns this when there is no viewer to answer.
            await ws.send(
                json.dumps({"jsonrpc": "2.0", "id": ask["id"], "result": {"action": "cancel"}})
            )
        result, updates = await response(ws, 3)
        if cancel_prompt:
            assert result["result"]["stopReason"] == "cancelled"
        else:
            assert result["error"]["message"] == "runtime operation failed"
        assert not any(
            frame["params"]["update"]["sessionUpdate"] == "agent_message_chunk" for frame in updates
        )
        with pytest.raises(ProcessLookupError):
            os.kill(helper_pid, 0)


async def test_native_model_cutoff_is_an_acp_error_not_end_turn(tmp_path):
    from unittest.mock import Mock

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider

    from .helpers import make_model_route

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

    model = Model(
        responses=[
            AIMessage(
                content=[{"type": "thinking", "thinking": "Unfinished", "signature": "test"}],
                response_metadata={"stop_reason": "max_tokens"},
                usage_metadata={"input_tokens": 12, "output_tokens": 4096, "total_tokens": 4108},
            )
        ]
    )
    provider = DeepAgentsProvider(make_model_route(Mock(return_value=model)))
    async with (
        server(
            tmp_path,
            provider_override=provider,
            options_overrides={"tool_output_inspection_enabled": False},
        ) as (url, _provider, config),
        connect(url) as ws,
    ):
        session_id = await initialize(ws, config)
        await send(ws, 3, "session/prompt", prompt(session_id, "write plan"))
        result, updates = await response(ws, 3)
        assert result["error"] == {"code": -32603, "message": "runtime operation failed"}
        assert "result" not in result
        usage = [
            u["params"]["update"]
            for u in updates
            if u["params"]["update"]["sessionUpdate"] == "usage_update"
        ]
        assert usage[-1]["outputTokens"] == 4096
