"""Pinned official ACP server with Lightspeed host policy and viewer replay.

Imports stay in this optional ACP package. The host wraps AgentServerACP.prompt
for execution policy; its protocol rendering and graph prompt loop are reused.
"""

from __future__ import annotations

from typing import Any, cast

from deepagents_acp.server import AgentServerACP

from lightspeed_agentic.acp.runtime_graph import InspectedGraph


class LightspeedACPServer(AgentServerACP):
    def __init__(self, connection: Any) -> None:
        self.connection = connection
        super().__init__(cast(Any, self._graph), load_sessions=True)
        self.on_connect(cast(Any, self))

    def _graph(self, _context: Any) -> InspectedGraph:
        session_id = self._building_session
        return cast(InspectedGraph, self.connection.sessions[session_id].graph)

    def _reset_agent(self, session_id: str) -> None:
        self._building_session = session_id
        super()._reset_agent(session_id)

    def _session_config(self, session_id: str) -> Any:
        config = super()._session_config(session_id)
        state = self.connection.sessions.get(session_id)
        if state is not None:
            config["configurable"]["thread_id"] = state.runtime.session_id
        return config

    async def _persist_session(self, session_id: str) -> None:
        # RuntimeSession owns its checkpoint metadata. Viewer journals are process-local.
        del session_id

    async def _replay_session(self, session_id: str, agent: Any) -> None:
        """Private pinned seam: never union raw checkpoint histories for display."""
        del agent
        state = self.connection.sessions[session_id]
        if (
            self.connection.config.replay_primary_session
            and not state.history
            and not state.is_primary
        ):
            state = next(
                (
                    candidate
                    for candidate in self.connection.sessions.values()
                    if candidate.is_primary
                ),
                state,
            )
        if state.history_truncated:
            replay_id = next(
                sid for sid, candidate in self.connection.sessions.items() if candidate is state
            )
            await self.connection.send(
                {
                    "method": "session/update",
                    "params": {
                        "sessionId": replay_id,
                        "update": {
                            "sessionUpdate": "session_info_update",
                            "_meta": {"lightspeed": {"historyTruncated": True}},
                        },
                    },
                }
            )
        for envelope in list(state.history):
            await self.connection.send(envelope)

    async def session_update(self, session_id: str, update: Any, **_kwargs: Any) -> None:
        payload = update.model_dump(by_alias=True, exclude_none=True)
        state = self.connection.sessions[session_id]
        if payload.get("sessionUpdate") in ("agent_message_chunk", "agent_thought_chunk"):
            payload["_meta"] = {"goose": {"messageId": state.message_id}}
        if state.graph is not None and payload.get("toolCallId") in state.graph.tool_errors:
            payload["status"] = "failed"
        await self.connection.update(session_id, payload)

    async def _process_tool_call_chunks(
        self,
        session_id: str,
        message_chunk: Any,
        active_tool_calls: dict[str, Any],
        tool_call_accumulator: dict[str, Any],
    ) -> None:
        await super()._process_tool_call_chunks(
            session_id, message_chunk, active_tool_calls, tool_call_accumulator
        )
        # Upstream suppresses edit_file completion output after publishing its
        # input diff. Lightspeed still owes the viewer its inspected result.
        # This changes presentation metadata only; the actual tool stays native.
        for call in active_tool_calls.values():
            if call.get("name") == "edit_file":
                call["name"] = "lightspeed_edit_file_result"

    # The SDK router owns standard ACP method parameter decoding and routing.
    # These wrappers enforce the launcher/session policy before delegation.
    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: Any = None,
        client_info: Any = None,
        **kwargs: Any,
    ) -> Any:
        del client_capabilities, client_info
        return await self.connection.handle(
            "initialize", {"protocolVersion": protocol_version, **kwargs}
        )

    async def new_session(
        self, cwd: str, additional_directories: Any = None, mcp_servers: Any = None, **kwargs: Any
    ) -> Any:
        del additional_directories
        return await self.connection.handle(
            "session/new",
            {
                "cwd": cwd,
                "mcpServers": [
                    s.model_dump(by_alias=True, exclude_none=True) for s in mcp_servers or []
                ],
                **kwargs,
            },
        )

    async def load_session(
        self,
        cwd: str,
        session_id: str,
        additional_directories: Any = None,
        mcp_servers: Any = None,
        **kwargs: Any,
    ) -> Any:
        del additional_directories
        return await self.connection.handle(
            "session/load",
            {
                "cwd": cwd,
                "sessionId": session_id,
                "mcpServers": [
                    s.model_dump(by_alias=True, exclude_none=True) for s in mcp_servers or []
                ],
                **kwargs,
            },
        )

    async def set_session_mode(self, mode_id: str, session_id: str, **_kwargs: Any) -> Any:
        return await self.connection.handle(
            "session/set_mode", {"sessionId": session_id, "modeId": mode_id}
        )

    async def prompt(
        self, prompt: Any, session_id: str, message_id: str | None = None, **kwargs: Any
    ) -> Any:
        del message_id
        return await self.connection.handle(
            "session/prompt",
            {
                "sessionId": session_id,
                "prompt": [block.model_dump(by_alias=True, exclude_none=True) for block in prompt],
                "_meta": kwargs,
            },
        )

    async def cancel(self, session_id: str, **_kwargs: Any) -> None:
        await self.connection.handle("session/cancel", {"sessionId": session_id})

    async def ext_method(self, method: str, params: dict[str, Any]) -> Any:
        return await self.connection.handle("_" + method, params)

    async def _unsupported(self, *_args: Any, **_kwargs: Any) -> Any:
        from acp.exceptions import RequestError

        raise RequestError(-32601, "method not supported")

    authenticate = _unsupported
    close_session = _unsupported
    list_sessions = _unsupported
    fork_session = _unsupported
    resume_session = _unsupported
    set_session_model = _unsupported
    set_config_option = _unsupported
    ext_notification = _unsupported
