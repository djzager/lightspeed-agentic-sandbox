"""Real official ACP loop and Lightspeed graphs; offline model and classifier."""

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from deepagents_acp.server import AgentServerACP
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGenerationChunk

from lightspeed_agentic.acp.config import ACPConfig
from lightspeed_agentic.acp.server import ACPConnection
from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed
from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
from lightspeed_agentic.types import ProviderQueryOptions

from .helpers import make_model_route

pytestmark = pytest.mark.asyncio


class MemoryWebSocket:
    def __init__(self):
        self.frames = []

    async def send(self, value):
        self.frames.append(json.loads(value))


async def native_connection(
    tmp_path, monkeypatch, responses, *, inspect=False, max_turns=7, pause_first=None
):
    monkeypatch.setenv("HOME", str(tmp_path))
    inputs = []

    class Model(FakeMessagesListChatModel):
        def bind_tools(self, _tools, **_kwargs):
            return self

        def _generate(self, messages, *args, **kwargs):
            inputs.append(list(messages))
            return super()._generate(messages, *args, **kwargs)

        def _stream(self, messages, stop=None, run_manager=None, **kwargs):
            message = self._generate(messages, stop, run_manager, **kwargs).generations[0].message
            chunks = [
                {
                    "name": call["name"],
                    "id": call["id"],
                    "args": json.dumps(call["args"]),
                    "index": index,
                }
                for index, call in enumerate(message.tool_calls)
            ]
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=message.content,
                    tool_call_chunks=chunks,
                    chunk_position="last",
                    usage_metadata=message.usage_metadata,
                    response_metadata=message.response_metadata,
                )
            )

        async def _astream(self, messages, stop=None, run_manager=None, **kwargs):
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                yield chunk
            if pause_first is not None and len(inputs) == 1:
                started, release = pause_first
                started.set()
                await release.wait()

    model = Model(responses=responses)
    route = replace(
        make_model_route(
            Mock(side_effect=[model, SimpleNamespace(profile={}, disable_streaming=False)])
        ),
        create_classifier_client=lambda _model: object(),
    )
    provider = DeepAgentsProvider(route)
    options = ProviderQueryOptions(
        prompt="",
        system_prompt="rules",
        cwd=str(tmp_path),
        max_turns=max_turns,
        tool_output_inspection_enabled=inspect,
    )
    config = ACPConfig(cwd=str(tmp_path), max_turns=max_turns)
    connection = ACPConnection(MemoryWebSocket(), provider, config, options)
    await connection.handle("initialize", {"protocolVersion": 1})
    session_id = (await connection.handle("session/new", {"cwd": str(tmp_path), "mcpServers": []}))[
        "sessionId"
    ]
    return connection, session_id, inputs


def tool_call(path, call_id="read"):
    return {"name": "read_file", "id": call_id, "args": {"file_path": str(path)}}


def prompt(session_id, text="read fixture", **kwargs):
    return {"sessionId": session_id, "prompt": [{"type": "text", "text": text}], **kwargs}


def results(connection):
    return [
        frame["params"]["update"]
        for frame in connection.websocket.frames
        if frame.get("params", {}).get("update", {}).get("sessionUpdate") == "tool_call_update"
    ]


async def test_official_loop_uses_native_graph_retains_two_prompt_history(tmp_path, monkeypatch):
    connection, sid, inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="one"), AIMessage(content="two")]
    )
    original = AgentServerACP.prompt
    calls = []

    async def observed(server, *args, **kwargs):
        calls.append(server)
        return await original(server, *args, **kwargs)

    monkeypatch.setattr(AgentServerACP, "prompt", observed)
    for text in ("first", "second"):
        result = await connection.handle("session/prompt", prompt(sid, text))
        assert result["stopReason"] == "end_turn"
    assert len(calls) == 2
    assert all(isinstance(server, AgentServerACP) for server in calls)
    state = connection.sessions[sid]
    assert state.graph.checkpointer is state.runtime._resources["agent"].checkpointer
    snapshot = await state.graph.aget_state(
        {"configurable": {"thread_id": state.runtime.session_id}}
    )
    assert [
        message.content
        for message in snapshot.values["messages"]
        if isinstance(message, HumanMessage)
    ] == [[{"type": "text", "text": "first"}], [{"type": "text", "text": "second"}]]
    assert len(inputs) == 2
    await state.runtime.aclose()


async def test_delayed_acceptance_keeps_pulling_stream_before_release(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.write_text("APPROVED-PAYLOAD")
    started, release = asyncio.Event(), asyncio.Event()

    async def inspect(*_args, **_kwargs):
        started.set()
        await release.wait()
        return SimpleNamespace(passed=True)

    monkeypatch.setattr("lightspeed_agentic.inspection.inspector.inspect_tool_result", inspect)
    connection, sid, _inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="", tool_calls=[tool_call(source)]), AIMessage(content="done")],
        inspect=True,
    )
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid)))
    await asyncio.wait_for(started.wait(), 5)
    assert not results(connection)
    assert "APPROVED-PAYLOAD" not in json.dumps(connection.websocket.frames)
    release.set()
    assert (await asyncio.wait_for(task, 5))["stopReason"] == "end_turn"
    assert "APPROVED-PAYLOAD" in json.dumps(results(connection))
    await connection.sessions[sid].runtime.aclose()


@pytest.mark.parametrize("failure", ["reject", "technical", "mixed", "pending"])
async def test_rejected_failed_mixed_pending_never_enter_live_or_replay(
    tmp_path, monkeypatch, failure
):
    first, second = tmp_path / "first", tmp_path / "second"
    first.write_text("FIRST-PRIVATE-PAYLOAD")
    second.write_text("SECOND-PRIVATE-PAYLOAD")
    started = asyncio.Event()

    async def inspect(_client, **kwargs):
        started.set()
        if failure == "pending":
            await asyncio.Future()
        if failure == "technical":
            raise RuntimeError("classifier technical failure")
        return SimpleNamespace(passed=failure == "mixed" and kwargs["tool_call_id"] == "first")

    monkeypatch.setattr("lightspeed_agentic.inspection.inspector.inspect_tool_result", inspect)
    calls = [tool_call(first, "first")]
    if failure == "mixed":
        calls.append(tool_call(second, "second"))
    connection, sid, inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="", tool_calls=calls), AIMessage(content="must not run")],
        inspect=True,
    )
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid)))
    if failure == "pending":
        await asyncio.wait_for(started.wait(), 5)
        await connection.handle("session/cancel", {"sessionId": sid})
        assert (await task)["stopReason"] == "cancelled"
    else:
        with pytest.raises(ToolResultSafetyInspectionFailed):
            await task
    assert len(inputs) == 1
    assert not results(connection)
    viewer = ACPConnection(
        MemoryWebSocket(),
        connection.provider,
        connection.config,
        connection.options,
        sessions=connection.sessions,
    )
    await viewer.handle("initialize", {"protocolVersion": 1})
    await viewer.handle("session/load", {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []})
    assert not results(viewer)
    assert "PRIVATE-PAYLOAD" not in json.dumps(viewer.websocket.frames)
    assert len(inputs) == 1
    await connection.sessions[sid].runtime.aclose()


async def test_max_turns_inspects_tool_then_handoff_reuses_history(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.write_text("APPROVED-PAYLOAD")
    inspection = AsyncMock(return_value=SimpleNamespace(passed=True))
    monkeypatch.setattr("lightspeed_agentic.inspection.inspector.inspect_tool_result", inspection)
    connection, sid, inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="", tool_calls=[tool_call(source)]), AIMessage(content="handoff done")],
        inspect=True,
        max_turns=1,
    )
    result = await connection.handle("session/prompt", prompt(sid))
    assert result == {"stopReason": "max_turn_requests", "_meta": {"lightspeed": {"turnsUsed": 1}}}
    assert "APPROVED-PAYLOAD" in json.dumps(results(connection))
    assert len(inputs) == 1
    result = await connection.handle("session/prompt", prompt(sid, "continue"))
    assert result["stopReason"] == "end_turn"
    assert any(isinstance(message, ToolMessage) for message in inputs[-1])
    assert inspection.await_count == 1
    await connection.sessions[sid].runtime.aclose()


async def test_native_timeout_repairs_history_before_followup(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.write_text("PENDING-PAYLOAD")
    started = asyncio.Event()

    async def inspect(*_args, **_kwargs):
        if not started.is_set():
            started.set()
            await asyncio.Future()
        return SimpleNamespace(passed=True)

    monkeypatch.setattr("lightspeed_agentic.inspection.inspector.inspect_tool_result", inspect)
    connection, sid, _inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="", tool_calls=[tool_call(source)]), AIMessage(content="followup")],
        inspect=True,
    )
    connection.timeout_seconds = 0.1
    with pytest.raises(TimeoutError):
        await connection.handle("session/prompt", prompt(sid))
    assert started.is_set()
    assert not results(connection)
    runtime = connection.sessions[sid].runtime
    graph = runtime._resources["agent"]
    snapshot = await graph.aget_state({"configurable": {"thread_id": runtime.session_id}})
    assert any(isinstance(message, ToolMessage) for message in snapshot.values["messages"])
    # Continuation may inspect the retained result, but must not re-run its tool.
    connection.timeout_seconds = 5
    assert (await connection.handle("session/prompt", prompt(sid, "continue")))[
        "stopReason"
    ] == "end_turn"
    await runtime.aclose()


async def test_failed_tool_status_and_edit_completion_are_preserved(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.write_text("before")
    edit = {
        "name": "edit_file",
        "id": "edit",
        "args": {"file_path": str(source), "old_string": "missing", "new_string": "after"},
    }
    connection, sid, _inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="", tool_calls=[edit]), AIMessage(content="done")]
    )
    assert (await connection.handle("session/prompt", prompt(sid)))["stopReason"] == "end_turn"
    updates = results(connection)
    assert len(updates) == 1
    assert updates[0]["toolCallId"] == "edit"
    assert updates[0]["status"] == "failed"
    assert source.read_text() == "before"
    await connection.sessions[sid].runtime.aclose()


@pytest.mark.parametrize("mismatch", ["content", "status", "namespace", "future_occurrence"])
async def test_gate_matches_exact_current_occurrence_and_namespace(mismatch):
    from lightspeed_agentic.acp.runtime_graph import InspectedGraph
    from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware

    message = ToolMessage(content="PRIVATE-RESULT", name="lookup", tool_call_id="same-id")
    signature = ToolResultInspectionMiddleware._signature(
        "lookup", "result", "same-id", message.content
    )
    changed, namespace = message, ()
    if mismatch == "content":
        changed = message.model_copy(update={"content": "CHANGED-RESULT"})
    if mismatch == "status":
        changed = message.model_copy(update={"status": "error"})
    if mismatch == "namespace":
        namespace = ("child",)
    frames = [
        ((), "messages", (changed, {})),
        (namespace, "custom", {"lightspeed.accepted_tool_results": {signature}}),
    ]
    if mismatch == "future_occurrence":
        frames.append(((), "messages", (message, {})))

    class Graph:
        checkpointer = object()

        async def astream(self, _value, **kwargs):
            assert kwargs["stream_mode"] == ["messages", "updates", "custom"]
            for frame in frames:
                yield frame

    async def publish(_event):
        pass

    session = SimpleNamespace(
        _max_turns=3,
        session_id="native-thread",
        _options=SimpleNamespace(tool_output_inspection_enabled=True),
    )
    gate = InspectedGraph(Graph(), session, publish)
    seen = []

    async def collect():
        async for frame in gate.astream(
            {}, config={}, stream_mode=["messages", "updates"], subgraphs=True
        ):
            seen.append(frame)

    with pytest.raises(ToolResultSafetyInspectionFailed):
        await collect()
    assert len(seen) == (1 if mismatch == "future_occurrence" else 0)


async def test_private_replay_seam_never_reads_checkpoint_history(tmp_path, monkeypatch):
    import inspect

    assert list(inspect.signature(AgentServerACP._replay_session).parameters) == [
        "self",
        "session_id",
        "agent",
    ]
    connection, sid, inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="answer")]
    )
    await connection.handle("session/prompt", prompt(sid, "original user prompt"))
    state = connection.sessions[sid]
    before = list(state.history)
    monkeypatch.setattr(
        type(state.graph.graph),
        "aget_state_history",
        Mock(side_effect=AssertionError("raw checkpoint replay")),
    )
    viewer = ACPConnection(
        MemoryWebSocket(),
        connection.provider,
        connection.config,
        connection.options,
        sessions=connection.sessions,
    )
    await viewer.handle("initialize", {"protocolVersion": 1})
    await viewer.handle("session/load", {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []})
    assert viewer.websocket.frames == [{"jsonrpc": "2.0", **envelope} for envelope in before]
    assert state.history == before
    assert len(inputs) == 1
    await state.runtime.aclose()


async def test_cancel_during_preparation_is_active_and_cannot_start_model(tmp_path, monkeypatch):
    connection, sid, inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="followup")]
    )
    started, stopped = asyncio.Event(), asyncio.Event()
    original = connection.provider.prepare_graph

    async def prepare(*_args, **_kwargs):
        started.set()
        try:
            await asyncio.Future()
        finally:
            stopped.set()

    monkeypatch.setattr(connection.provider, "prepare_graph", prepare)
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid)))
    await asyncio.wait_for(started.wait(), 5)
    await connection.handle("session/cancel", {"sessionId": sid})
    assert (await task)["stopReason"] == "cancelled"
    assert stopped.is_set()
    assert not inputs
    monkeypatch.setattr(connection.provider, "prepare_graph", original)
    assert (await connection.handle("session/prompt", prompt(sid, "continue")))[
        "stopReason"
    ] == "end_turn"
    await connection.sessions[sid].runtime.aclose()


@pytest.mark.parametrize("budget", [1, 2])
async def test_steering_during_final_response_keeps_same_prompt_budget(
    tmp_path, monkeypatch, budget
):
    started, release = asyncio.Event(), asyncio.Event()
    connection, sid, inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="first"), AIMessage(content="guided")],
        max_turns=budget,
        pause_first=(started, release),
    )
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid)))
    await asyncio.wait_for(started.wait(), 5)
    state = connection.sessions[sid]
    guidance = await connection.handle(
        "_goose/unstable/session/steer",
        {**prompt(sid, "use postgres"), "expectedRunId": state.run_id},
    )
    release.set()
    result = await task
    assert result["stopReason"] == ("max_turn_requests" if budget == 1 else "end_turn")
    assert len(inputs) == budget
    assert result["_meta"]["lightspeed"]["turnsUsed"] == budget
    frames = json.dumps(connection.websocket.frames)
    assert guidance["messageId"] in frames
    assert "use postgres" in frames
    await state.runtime.aclose()


async def test_shaping_uses_prepared_schema_counts_usage_without_main_turn(tmp_path, monkeypatch):
    main = AIMessage(
        content="answer", usage_metadata={"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
    )
    shaped = AIMessage(
        content="",
        tool_calls=[{"name": "OutputModel", "id": "shape", "args": {"value": "okay"}}],
        usage_metadata={"input_tokens": 7, "output_tokens": 4, "total_tokens": 11},
    )
    connection, sid, _inputs = await native_connection(
        tmp_path, monkeypatch, [main, shaped], max_turns=1
    )
    route = connection.provider._model_route
    model = next(route.create_model.side_effect)
    connection.provider._model_route = replace(route, create_model=Mock(return_value=model))
    runtime = connection.sessions[sid].runtime
    runtime._options.output_schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    result = await connection.handle("session/prompt", prompt(sid))
    assert result["stopReason"] == "end_turn"
    assert runtime.turns_used == 1
    assert runtime.usage.input_tokens == 10
    assert runtime.usage.output_tokens == 6
    await runtime.aclose()


async def test_active_model_cancel_awaits_task_and_followup_remains_usable(tmp_path, monkeypatch):
    started, release = asyncio.Event(), asyncio.Event()
    connection, sid, inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="interrupted"), AIMessage(content="continued")],
        pause_first=(started, release),
    )
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid)))
    await asyncio.wait_for(started.wait(), 5)
    await connection.handle("session/cancel", {"sessionId": sid})
    assert task.done()
    assert (await task)["stopReason"] == "cancelled"
    assert (await connection.handle("session/prompt", prompt(sid, "continue")))[
        "stopReason"
    ] == "end_turn"
    assert len(inputs) == 2
    await connection.sessions[sid].runtime.aclose()


async def test_terminal_publication_rejects_late_steering_and_cancel_keeps_response(
    tmp_path, monkeypatch
):
    from lightspeed_agentic.acp.server import RPCError

    connection, sid, inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="answer")]
    )
    blocked, release = asyncio.Event(), asyncio.Event()
    original = connection.websocket.send

    async def send(value):
        frame = json.loads(value)
        update = frame.get("params", {}).get("update", {})
        if update.get("_meta", {}).get("lightspeed", {}).get("turnsUsed") == 1:
            blocked.set()
            await release.wait()
        await original(value)

    monkeypatch.setattr(connection.websocket, "send", send)
    task = asyncio.create_task(
        connection.dispatch(
            {"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": prompt(sid)}
        )
    )
    await asyncio.wait_for(blocked.wait(), 5)
    state = connection.sessions[sid]
    with pytest.raises(RPCError, match="no active run"):
        await connection.handle(
            "_goose/unstable/session/steer",
            {**prompt(sid, "late guidance"), "expectedRunId": state.run_id},
        )
    await connection.handle("session/cancel", {"sessionId": sid})
    release.set()
    await asyncio.wait_for(task, 5)
    response = next(frame for frame in connection.websocket.frames if frame.get("id") == 3)
    assert response["result"]["stopReason"] == "end_turn"
    assert len(inputs) == 1
    await state.runtime.aclose()


async def test_viewer_load_keeps_original_pending_permission_with_owner(tmp_path, monkeypatch):
    source = tmp_path / "fixture"
    source.write_text("APPROVED-PAYLOAD")
    connection, sid, inputs = await native_connection(
        tmp_path,
        monkeypatch,
        [AIMessage(content="", tool_calls=[tool_call(source)]), AIMessage(content="done")],
    )
    state = connection.sessions[sid]
    state.runtime.mode = "approve"
    asking = asyncio.Event()
    original = connection.call

    async def call(method, params):
        asking.set()
        return await original(method, params)

    monkeypatch.setattr(connection, "call", call)
    task = asyncio.create_task(connection.handle("session/prompt", prompt(sid, "original prompt")))
    await asyncio.wait_for(asking.wait(), 5)
    assert len(connection.pending) == 1
    pending_id, future = next(iter(connection.pending.items()))
    viewer = ACPConnection(
        MemoryWebSocket(),
        connection.provider,
        connection.config,
        connection.options,
        sessions=connection.sessions,
    )
    await viewer.handle("initialize", {"protocolVersion": 1})
    await viewer.handle("session/load", {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []})
    assert "original prompt" in json.dumps(viewer.websocket.frames)
    assert not any(
        frame.get("method") == "session/request_permission" for frame in viewer.websocket.frames
    )
    assert connection.pending[pending_id] is future
    assert not future.done()
    assert len(inputs) == 1
    future.set_result({"outcome": {"outcome": "selected", "optionId": "allow"}})
    assert (await task)["stopReason"] == "end_turn"
    assert "APPROVED-PAYLOAD" in json.dumps(results(connection))
    await state.runtime.aclose()


async def test_bounded_authorized_journal_announces_truncation(tmp_path, monkeypatch):
    connection, sid, _inputs = await native_connection(
        tmp_path, monkeypatch, [AIMessage(content="unused")]
    )
    connection.MAX_HISTORY_BYTES = 256
    for index in range(10):
        connection.record(
            sid,
            {
                "method": "session/update",
                "params": {
                    "sessionId": sid,
                    "update": {
                        "sessionUpdate": "agent_message_chunk",
                        "content": {"type": "text", "text": str(index) * 32},
                    },
                },
            },
        )
    state = connection.sessions[sid]
    assert state.history_bytes <= 256
    assert state.history_truncated
    viewer = ACPConnection(
        MemoryWebSocket(),
        connection.provider,
        connection.config,
        connection.options,
        sessions=connection.sessions,
    )
    await viewer.handle("initialize", {"protocolVersion": 1})
    await viewer.handle("session/load", {"sessionId": sid, "cwd": str(tmp_path), "mcpServers": []})
    assert (
        viewer.websocket.frames[0]["params"]["update"]["_meta"]["lightspeed"]["historyTruncated"]
        is True
    )
    assert viewer.websocket.frames[-1]["params"]["update"]["content"]["text"] == "9" * 32
    await state.runtime.aclose()


async def test_native_subagent_inspection_stays_internal_and_parent_is_independent(
    tmp_path, monkeypatch
):
    source = tmp_path / "fixture"
    source.write_text("CHILD-PRIVATE-TOOL-PAYLOAD")
    inspection = AsyncMock(return_value=SimpleNamespace(passed=True))
    monkeypatch.setattr("lightspeed_agentic.inspection.inspector.inspect_tool_result", inspection)
    delegate = {
        "name": "task",
        "id": "delegate",
        "args": {
            "description": "Read the local fixture and report done",
            "subagent_type": "general-purpose",
        },
    }
    replies = [
        AIMessage(content="", tool_calls=[delegate]),
        AIMessage(content="", tool_calls=[tool_call(source)]),
        AIMessage(content="child done"),
        AIMessage(content="parent done"),
    ]
    connection, sid, inputs = await native_connection(tmp_path, monkeypatch, replies, inspect=True)
    result = await connection.handle("session/prompt", prompt(sid))
    assert result["stopReason"] == "end_turn"
    assert result["_meta"]["lightspeed"]["turnsUsed"] == 2
    assert len(inputs) == 4
    assert "CHILD-PRIVATE-TOOL-PAYLOAD" not in json.dumps(connection.websocket.frames)
    assert {update["toolCallId"] for update in results(connection)} == {"delegate"}
    assert inspection.await_count == 2
    await connection.sessions[sid].runtime.aclose()
