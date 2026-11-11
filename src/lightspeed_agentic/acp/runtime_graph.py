"""Inspection-approved view of the native Lightspeed graph for official ACP."""

from __future__ import annotations

from collections import deque
from collections.abc import AsyncIterator
from typing import Any

from lightspeed_agentic.inspection.errors import ToolResultSafetyInspectionFailed


class InspectedGraph:
    """Keep consuming native frames while inspection approves buffered results.

    Acceptance applies only to pending occurrences with the exact content/status
    signature and source namespace. A child boundary cannot approve its parent.
    Replay never uses this graph's checkpoint history; the server journals only
    authorized ACP publication attempts instead.
    """

    MAX_PENDING_RESULTS = 256
    MAX_PENDING_BYTES = 8 * 1024 * 1024

    def __init__(self, graph: Any, session: Any, publish_event: Any) -> None:
        self.graph = graph
        self.session = session
        self.publish_event = publish_event
        self.events: deque[Any] = deque()
        self.callback: Any = None
        self.tool_errors: set[str] = set()
        self.result_text = ""

    def __getattr__(self, name: str) -> Any:
        return getattr(self.graph, name)

    @property
    def checkpointer(self) -> Any:
        return self.graph.checkpointer

    @checkpointer.setter
    def checkpointer(self, value: Any) -> None:
        self.graph.checkpointer = value

    async def flush_events(self) -> None:
        while self.events:
            await self.publish_event(self.events.popleft())

    async def astream(
        self, value: Any, *, config: Any, stream_mode: Any, subgraphs: bool, **kwargs: Any
    ) -> AsyncIterator[Any]:
        from langchain_core.messages import ToolMessage

        from lightspeed_agentic.inspection.middleware import ToolResultInspectionMiddleware

        pending: list[tuple[tuple[str, ...], Any, Any, int]] = []
        pending_bytes = 0
        config = {
            **config,
            "configurable": {
                **config.get("configurable", {}),
                "thread_id": self.session.session_id,
            },
            "recursion_limit": max(1000, self.session._max_turns * 20),
            "callbacks": [*config.get("callbacks", []), self.callback],
        }
        modes = list(dict.fromkeys([*stream_mode, "custom"]))
        async for frame in self.graph.astream(
            value, config=config, stream_mode=modes, subgraphs=subgraphs, **kwargs
        ):
            namespace, kind, data = frame
            namespace = tuple(namespace)
            await self.flush_events()
            if kind == "messages" and isinstance(data[0], ToolMessage):
                if not self.session._options.tool_output_inspection_enabled:
                    if not namespace:
                        if data[0].status == "error":
                            self.tool_errors.add(data[0].tool_call_id)
                        else:
                            self.tool_errors.discard(data[0].tool_call_id)
                        yield frame
                    continue
                message = data[0]
                signature = ToolResultInspectionMiddleware._signature(
                    message.name or "",
                    "error" if message.status == "error" else "result",
                    message.tool_call_id or "",
                    message.content,
                )
                size = len(str(message.content).encode("utf-8"))
                pending_bytes += size
                pending.append((namespace, signature, frame, size))
                if (
                    len(pending) > self.MAX_PENDING_RESULTS
                    or pending_bytes > self.MAX_PENDING_BYTES
                ):
                    raise ToolResultSafetyInspectionFailed()
                continue
            if (
                kind == "custom"
                and isinstance(data, dict)
                and "lightspeed.accepted_tool_results" in data
            ):
                accepted = data["lightspeed.accepted_tool_results"]
                remaining = []
                for source, signature, buffered, size in pending:
                    if source == namespace and signature in accepted:
                        pending_bytes -= size
                        if not source:
                            message = buffered[2][0]
                            if message.status == "error":
                                self.tool_errors.add(message.tool_call_id)
                            else:
                                self.tool_errors.discard(message.tool_call_id)
                            yield buffered
                    else:
                        remaining.append((source, signature, buffered, size))
                pending = remaining
                continue
            if (
                kind == "messages"
                and not namespace
                and getattr(data[0], "type", None) in ("ai", "AIMessageChunk")
                and any(not source for source, *_rest in pending)
            ):
                raise ToolResultSafetyInspectionFailed()
            # Child graph messages remain internal, matching the existing runtime.
            if kind in stream_mode and not namespace:
                if kind == "messages" and getattr(data[0], "type", None) in (
                    "ai",
                    "AIMessageChunk",
                ):
                    self.result_text += "".join(
                        block.get("text", "")
                        for block in data[0].content_blocks
                        if block.get("type") == "text"
                    )
                yield frame
        await self.flush_events()
        if pending:
            raise ToolResultSafetyInspectionFailed()
