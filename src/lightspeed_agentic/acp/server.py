"""ACP v1 JSON-RPC over the harness's authenticated local WebSocket transport.

Only normalized runtime events cross this boundary. In particular, raw graph
messages and uninspected tool results must never be streamed here.
"""

from __future__ import annotations

import asyncio
import hmac
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from http import HTTPStatus
from typing import Any
from urllib.parse import parse_qs, urlsplit
from uuid import uuid4

from lightspeed_agentic.acp.ask_user import ask_user
from lightspeed_agentic.acp.config import ACPConfig
from lightspeed_agentic.types import ProviderQueryOptions


class RPCError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code


def _prompt(params: dict[str, Any]) -> str:
    blocks = params.get("prompt")
    if not isinstance(blocks, list) or not blocks:
        raise RPCError(-32602, "text prompt required")
    if any(
        not isinstance(b, dict) or b.get("type") != "text" or not isinstance(b.get("text"), str)
        for b in blocks
    ):
        raise RPCError(-32602, "only text content is supported")
    text = "\n".join(b["text"] for b in blocks)
    if not text.strip():
        raise RPCError(-32602, "text prompt required")
    return text


@dataclass
class SessionState:
    runtime: Any
    owner: Any = None
    run_id: str | None = None
    message_id: str = ""
    history: list[dict[str, Any]] = field(default_factory=list)
    cancel_requested: bool = False


class ACPConnection:
    """One reader demultiplexes replies while prompt/cancel/asks run concurrently."""

    def __init__(
        self,
        websocket: Any,
        provider: Any,
        config: ACPConfig,
        options: ProviderQueryOptions,
        *,
        timeout_seconds: float = 600,
        sessions: dict[str, SessionState] | None = None,
    ) -> None:
        self.websocket = websocket
        self.provider = provider
        self.config = config
        self.options = options
        self.timeout_seconds = timeout_seconds
        self.initialized = False
        self.sessions = sessions if sessions is not None else {}
        self.pending: dict[str, asyncio.Future[dict[str, Any]]] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.request_ids: set[str | int] = set()
        self.send_lock = asyncio.Lock()

    async def send(self, message: dict[str, Any]) -> None:
        async with self.send_lock:
            await self.websocket.send(json.dumps({"jsonrpc": "2.0", **message}, allow_nan=False))

    async def update(self, session_id: str, update: dict[str, Any]) -> None:
        self.sessions[session_id].history.append(update)
        await self.send(
            {"method": "session/update", "params": {"sessionId": session_id, "update": update}}
        )

    async def call(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        request_id = "lightspeed-" + uuid4().hex
        future: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self.pending[request_id] = future
        try:
            await self.send({"id": request_id, "method": method, "params": params})
            return await future
        finally:
            self.pending.pop(request_id, None)

    async def serve(self) -> None:
        def completed(task: asyncio.Task[None]) -> None:
            self.tasks.discard(task)
            if not task.cancelled():
                # A peer can disappear during a response write. Retrieve failures
                # without logging peer-provided close reasons or protocol content.
                task.exception()

        try:
            async for raw in self.websocket:
                try:
                    frame = json.loads(raw)
                except (ValueError, UnicodeError):
                    await self.send(
                        {"id": None, "error": {"code": -32700, "message": "invalid JSON"}}
                    )
                    continue
                if not isinstance(frame, dict) or frame.get("jsonrpc") != "2.0":
                    await self.send(
                        {"id": None, "error": {"code": -32600, "message": "invalid request"}}
                    )
                    continue
                if "method" not in frame:
                    frame_id = frame.get("id")
                    future = self.pending.get(frame_id) if isinstance(frame_id, str) else None
                    if future is not None and not future.done():
                        result = frame.get("result")
                        if "error" in frame or not isinstance(result, dict):
                            future.set_exception(RPCError(-32603, "client request failed"))
                        else:
                            future.set_result(result)
                    continue
                task = asyncio.create_task(self.dispatch(frame))
                self.tasks.add(task)
                task.add_done_callback(completed)
        finally:
            owned = [state for state in self.sessions.values() if state.owner is self]
            await asyncio.gather(
                *(state.runtime.cancel() for state in owned), return_exceptions=True
            )
            for task in list(self.tasks):
                task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            for future in self.pending.values():
                if not future.done():
                    future.cancel()
            await asyncio.gather(
                *(state.runtime.aclose() for state in owned), return_exceptions=True
            )

    async def dispatch(self, frame: dict[str, Any]) -> None:
        request_id = frame.get("id")
        is_request = "id" in frame
        registered = False
        try:
            if is_request:
                if (
                    not isinstance(request_id, str | int)
                    or isinstance(request_id, bool)
                    or request_id in self.request_ids
                ):
                    raise RPCError(-32600, "invalid or duplicate request id")
                self.request_ids.add(request_id)
                registered = True
            method, params = frame.get("method"), frame.get("params", {})
            if not isinstance(method, str) or not isinstance(params, dict):
                raise RPCError(-32602, "invalid method parameters")
            if not is_request and method != "session/cancel":
                return
            result = await self.handle(method, params)
            if is_request:
                await self.send({"id": request_id, "result": result})
        except RPCError as exc:
            if is_request:
                await self.send(
                    {
                        "id": request_id if type(request_id) in (str, int) else None,
                        "error": {"code": exc.code, "message": str(exc)},
                    }
                )
        except Exception:
            # Model, subprocess, and validation exceptions can contain secrets/content.
            if is_request:
                await self.send(
                    {
                        "id": request_id,
                        "error": {"code": -32603, "message": "runtime operation failed"},
                    }
                )
        finally:
            if registered:
                self.request_ids.discard(request_id)

    def session(self, params: dict[str, Any]) -> tuple[str, SessionState]:
        session_id = params.get("sessionId")
        if not isinstance(session_id, str) or session_id not in self.sessions:
            raise RPCError(-32602, "unknown session")
        return session_id, self.sessions[session_id]

    async def handle(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method == "initialize":
            if type(params.get("protocolVersion")) is not int or params["protocolVersion"] != 1:
                raise RPCError(-32602, "ACP protocol version 1 required")
            self.initialized = True
            return {
                "protocolVersion": 1,
                "agentInfo": {"name": "lightspeed-agentic", "version": "0.1.0"},
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {},
                    "mcpCapabilities": {},
                },
            }
        if not self.initialized:
            raise RPCError(-32000, "initialize required")
        if method == "session/new":
            try:
                self.config.validate_session(params)
            except ValueError as exc:
                raise RPCError(-32602, "session configuration rejected") from exc
            if len(self.sessions) >= 16:
                raise RPCError(-32000, "session limit reached")
            session_id = uuid4().hex

            async def permission(request: Any) -> bool:
                reply = await self.call(
                    "session/request_permission",
                    {
                        "sessionId": session_id,
                        "toolCall": {
                            "toolCallId": request.tool_call_id,
                            "title": request.tool_name,
                            "kind": "other",
                            "status": "pending",
                            "rawInput": request.arguments,
                        },
                        "options": [
                            {"optionId": "allow", "name": "Allow once", "kind": "allow_once"},
                            {"optionId": "reject", "name": "Reject once", "kind": "reject_once"},
                        ],
                    },
                )
                outcome = reply.get("outcome")
                return (
                    isinstance(outcome, dict)
                    and outcome.get("outcome") == "selected"
                    and outcome.get("optionId") == "allow"
                )

            async def user_input(request: Any) -> str:
                if self.config.ask_user is None:
                    raise RuntimeError("human input is unavailable")

                async def elicit(payload: dict[str, Any]) -> dict[str, Any]:
                    return await self.call(
                        "elicitation/create", {**payload, "sessionId": session_id}
                    )

                return await ask_user(
                    self.config.ask_user, request.question, getattr(request, "options", ()), elicit
                )

            runtime = self.provider.create_session(
                self.options,
                permission_callback=permission,
                user_input_callback=user_input if self.config.ask_user else None,
            )
            runtime.mode = self.config.mode
            self.sessions[session_id] = SessionState(runtime=runtime, owner=self)
            return {"sessionId": session_id}
        session_id, state = self.session(params)
        if method == "session/load":
            try:
                self.config.validate_session(params)
            except ValueError as exc:
                raise RPCError(-32602, "session configuration rejected") from exc
            for update in list(state.history):
                await self.send(
                    {
                        "method": "session/update",
                        "params": {"sessionId": session_id, "update": update},
                    }
                )
            return {}
        if state.owner is not self:
            raise RPCError(-32000, "session belongs to another connection")
        if method == "session/cancel":
            state.cancel_requested = True
            await state.runtime.cancel()
            return {}
        if method == "session/set_mode":
            mode = params.get("modeId")
            if mode not in ("auto", "approve"):
                raise RPCError(-32602, "unsupported session mode")
            if self.config.mode == "approve" and mode != "approve":
                raise RPCError(-32602, "launcher approval policy cannot be reduced")
            if state.run_id is not None:
                raise RPCError(-32000, "cannot change mode during a prompt")
            state.runtime.mode = mode
            return {}
        if method == "_goose/unstable/session/steer":
            prompt = _prompt(params)
            if state.run_id is None or params.get("expectedRunId") != state.run_id:
                raise RPCError(-32000, "active run does not match")
            message_id = await state.runtime.steer(prompt)
            return {"runId": state.run_id, "messageId": message_id}
        if method != "session/prompt":
            raise RPCError(-32601, "method not supported")
        prompt = _prompt(params)
        if state.run_id is not None:
            raise RPCError(-32000, "session already has an active prompt")
        state.run_id, state.message_id = uuid4().hex, uuid4().hex
        state.cancel_requested = False
        state.history.append(
            {
                "sessionUpdate": "user_message_chunk",
                "content": {"type": "text", "text": prompt},
                "_meta": {"goose": {"messageId": uuid4().hex}},
            }
        )
        await self.update(
            session_id,
            {"sessionUpdate": "session_info_update", "_meta": {"goose": {"runId": state.run_id}}},
        )
        try:
            async with asyncio.timeout(self.timeout_seconds):
                stop_reason = await self.consume(session_id, state, prompt)
                result: dict[str, Any] = {"stopReason": stop_reason}
                turns = getattr(state.runtime, "turns_used", None)
                if type(turns) is int and turns >= 0:
                    result["_meta"] = {"lightspeed": {"turnsUsed": turns}}
                return result
        finally:
            state.run_id = None

    async def consume(self, session_id: str, state: SessionState, prompt: str) -> str:
        if state.cancel_requested:
            return "cancelled"
        stop_reason = "end_turn"
        async for event in state.runtime.prompt(prompt, max_turns=self.config.max_turns):
            kind = event.type
            if kind == "user_message":
                state.message_id = uuid4().hex
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "user_message_chunk",
                        "content": {"type": "text", "text": event.text},
                        "_meta": {
                            "goose": {
                                "runId": state.run_id,
                                "messageId": event.message_id,
                                "steer": True,
                            }
                        },
                    },
                )
            elif kind in ("text_delta", "thinking_delta"):
                text = event.text if kind == "text_delta" else event.thinking
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "agent_message_chunk"
                        if kind == "text_delta"
                        else "agent_thought_chunk",
                        "content": {"type": "text", "text": text},
                        "_meta": {"goose": {"messageId": state.message_id}},
                    },
                )
            elif kind == "tool_call":
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call",
                        "toolCallId": event.call_id,
                        "title": event.name,
                        "kind": "other",
                        "status": "in_progress",
                        "rawInput": event.input,
                    },
                )
            elif kind == "tool_result":
                await self.update(
                    session_id,
                    {
                        "sessionUpdate": "tool_call_update",
                        "toolCallId": event.call_id,
                        "status": "failed" if event.error_type else "completed",
                        "content": [
                            {"type": "content", "content": {"type": "text", "text": event.output}}
                        ],
                    },
                )
            elif kind in ("usage", "usage_update", "result"):
                usage = event.usage if kind == "usage" else state.runtime.usage
                update: dict[str, Any] = {"sessionUpdate": "usage_update"}
                for source, target in (
                    ("input_tokens", "inputTokens"),
                    ("output_tokens", "outputTokens"),
                ):
                    value = getattr(usage, source, None)
                    if type(value) is int and value >= 0:
                        update[target] = value
                turns = getattr(event, "turns_used", None)
                if type(turns) is int and turns >= 0:
                    update["_meta"] = {"lightspeed": {"turnsUsed": turns}}
                await self.update(session_id, update)
                if kind == "result":
                    reason = getattr(event, "stop_reason", "end_turn")
                    stop_reason = "max_turn_requests" if reason == "max_turns" else reason
        return stop_reason


async def serve(
    provider: Any,
    config: ACPConfig,
    options: ProviderQueryOptions,
    *,
    host: str,
    port: int,
    secret: str,
    timeout_seconds: float = 600,
    on_listening: Callable[[int], None] | None = None,
) -> None:
    """Expose only a token-authenticated loopback endpoint, with bounded frames."""
    from websockets.asyncio.server import serve as websocket_serve

    if host != "127.0.0.1" or not secret:
        raise ValueError("ACP requires loopback binding and a nonempty secret")

    def authenticate(connection: Any, request: Any) -> Any:
        url = urlsplit(request.path)
        tokens = parse_qs(url.query).get("token", [])
        if not tokens:
            tokens = request.headers.get_all("X-Secret-Key")
        if (
            url.path != "/acp"
            or len(tokens) != 1
            or not hmac.compare_digest(tokens[0].encode(), secret.encode())
        ):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    sessions: dict[str, SessionState] = {}

    async def handler(websocket: Any) -> None:
        try:
            await ACPConnection(
                websocket,
                provider,
                config,
                options,
                timeout_seconds=timeout_seconds,
                sessions=sessions,
            ).serve()
        except Exception:
            # Provider cleanup exceptions must not enter WebSocket handler logs.
            await websocket.close(code=1011, reason="runtime connection closed")

    async with websocket_serve(
        handler, host, port, process_request=authenticate, max_size=1_048_576, origins=[None]
    ) as server:
        if on_listening:
            on_listening(next(iter(server.sockets)).getsockname()[1])
        await asyncio.Future()
