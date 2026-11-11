"""Transport-independent, in-memory conversations over the DeepAgents harness."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from lightspeed_agentic.types import (
    ProviderEvent,
    ProviderQueryOptions,
    ResultEvent,
    UsageEvent,
    UsageSnapshot,
    UserMessageEvent,
)

if TYPE_CHECKING:
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider

SessionMode = Literal["auto", "approve"]


@dataclass(frozen=True)
class ToolPermissionRequest:
    tool_name: str
    arguments: dict[str, Any]
    tool_call_id: str


@dataclass(frozen=True)
class UserInputRequest:
    question: str
    options: tuple[str, ...] = ()
    tool_call_id: str = ""


PermissionCallback = Callable[[ToolPermissionRequest], Awaitable[bool]]
UserInputCallback = Callable[[UserInputRequest], Awaitable[str]]


class SessionTurnLimitError(RuntimeError):
    """The session exhausted its main-model call budget after executing selected tools."""


class RuntimeSession:
    """Own one graph/checkpointer and serialize prompts against its stable thread.

    Usage contains observed main, subagent, summarization, and shaping completions.
    Isolated safety-classifier requests are deliberately outside that scope.
    """

    def __init__(
        self,
        provider: DeepAgentsProvider,
        options: ProviderQueryOptions,
        *,
        permission_callback: PermissionCallback | None = None,
        user_input_callback: UserInputCallback | None = None,
    ) -> None:
        self.session_id = f"ls-{uuid.uuid4().hex}"
        self.mode: SessionMode = "auto"
        self.permission_callback = permission_callback
        self.user_input_callback = user_input_callback
        self.usage = UsageSnapshot()
        self.turns_used = 0
        self._max_turns = options.max_turns
        self._provider = provider
        self._options = options
        self._resources: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._active: asyncio.Task[None] | None = None
        self._queue: asyncio.Queue[Any] | None = None
        self._cancel_requested = False
        self._closed = False
        self._steering: list[tuple[str, str]] = []

    def _observe_usage(self, response: Any, *, is_main: bool = True) -> None:
        """Capture present counts at model completion, even if a later tool is cancelled."""
        if is_main:
            self.turns_used += 1
        input_tokens, output_tokens = self.usage.input_tokens, self.usage.output_tokens
        for choices in getattr(response, "generations", None) or []:
            for generation in choices:
                metadata = getattr(getattr(generation, "message", None), "usage_metadata", None)
                if not isinstance(metadata, dict):
                    continue
                if metadata.get("input_tokens") is not None:
                    input_tokens = (input_tokens or 0) + metadata["input_tokens"]
                if metadata.get("output_tokens") is not None:
                    output_tokens = (output_tokens or 0) + metadata["output_tokens"]
        observed = UsageSnapshot(input_tokens=input_tokens, output_tokens=output_tokens)
        if observed != self.usage:
            self.usage = observed
            if self._queue is not None:
                self._queue.put_nowait(UsageEvent(usage=observed))

    async def prompt(
        self,
        prompt: str,
        *,
        max_turns: int | None = None,
        mode: SessionMode | None = None,
    ) -> AsyncIterator[ProviderEvent]:
        """Run a prompt; simultaneous calls queue until the active prompt completes."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Session is closed")
            if mode is not None:
                if mode not in ("auto", "approve"):
                    raise ValueError("Unsupported session mode")
                self.mode = mode
            options = replace(
                self._options,
                prompt=prompt,
                max_turns=self._options.max_turns if max_turns is None else max_turns,
            )
            if options.max_turns < 1:
                raise ValueError("max_turns must be positive")
            queue: asyncio.Queue[Any] = asyncio.Queue()
            self._queue = queue
            self._cancel_requested = False
            self.turns_used = 0
            self._max_turns = options.max_turns
            self._active = asyncio.create_task(self._run_prompt(options, queue))
            self._active.add_done_callback(lambda task: self._task_done(task, queue))
            try:
                while (item := await queue.get()) is not None:
                    if isinstance(item, BaseException):
                        raise item
                    yield item
            finally:
                if not self._active.done():
                    self._active.cancel()
                with suppress(asyncio.CancelledError):
                    await self._active
                self._active = None
                self._queue = None

    async def _run_prompt(
        self,
        options: ProviderQueryOptions,
        queue: asyncio.Queue[Any],
    ) -> None:
        from langgraph.errors import GraphRecursionError

        result_text = ""
        try:
            while True:
                followup = False
                async for event in self._provider._query(options, session=self):
                    if isinstance(event, ResultEvent):
                        if self._steering:
                            # A final model response has no next boundary. Consume accepted
                            # steering in this run before publishing its terminal result.
                            messages = self._take_steering()
                            options = replace(
                                options, prompt="\n\n".join(message.content for message in messages)
                            )
                            followup = True
                            break
                        event = replace(
                            event,
                            input_tokens=self.usage.input_tokens or 0,
                            output_tokens=self.usage.output_tokens or 0,
                            turns_used=self.turns_used,
                        )
                    elif event.type == "text_delta":
                        result_text += event.text
                    queue.put_nowait(event)
                if not followup:
                    break
        except BaseException as exc:
            reason: Literal["cancelled", "max_turns"]
            if isinstance(exc, asyncio.CancelledError) or self._cancel_requested:
                reason = "cancelled"
            elif isinstance(exc, (SessionTurnLimitError, GraphRecursionError)):
                reason = "max_turns"
            else:
                queue.put_nowait(exc)
                return
            try:
                await self._finish_unanswered_tools(reason)
            except Exception as repair_error:
                # A broken checkpoint cannot safely be continued.
                self._closed = True
                queue.put_nowait(repair_error)
                return
            queue.put_nowait(
                ResultEvent(
                    text=result_text,
                    input_tokens=self.usage.input_tokens or 0,
                    output_tokens=self.usage.output_tokens or 0,
                    stop_reason=reason,
                    turns_used=self.turns_used,
                )
            )
        finally:
            self._steering.clear()
            queue.put_nowait(None)

    @staticmethod
    def _task_done(task: asyncio.Task[None], queue: asyncio.Queue[Any]) -> None:
        # Cancellation before the worker's first instruction still terminates its stream.
        if task.cancelled():
            queue.put_nowait(ResultEvent(stop_reason="cancelled"))
            queue.put_nowait(None)
        elif error := task.exception():
            queue.put_nowait(error)
            queue.put_nowait(None)

    async def steer(self, text: str) -> str:
        """Queue user guidance for the next safe main-model boundary of this prompt."""
        if self._closed or self._active is None or self._active.done():
            raise RuntimeError("No active prompt to steer")
        message_id = uuid.uuid4().hex
        self._steering.append((message_id, text))
        return message_id

    def _take_steering(self) -> list[Any]:
        from langchain_core.messages import HumanMessage

        pending, self._steering = self._steering, []
        if self._queue is not None:
            for message_id, text in pending:
                self._queue.put_nowait(UserMessageEvent(text=text, message_id=message_id))
        return [HumanMessage(content=text, id=message_id) for message_id, text in pending]

    async def _finish_unanswered_tools(self, reason: str) -> None:
        """Close pending calls in history so the next prompt cannot replay their effects."""
        if not self._resources:
            return
        from langchain_core.messages import AIMessage, ToolMessage

        agent = self._resources["agent"]
        config = {"configurable": {"thread_id": self.session_id}}
        state = await agent.aget_state(config)
        pending: dict[str, Any] = {}
        for message in state.values.get("messages", []):
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    if call_id := call.get("id"):
                        pending[call_id] = call
            elif isinstance(message, ToolMessage):
                pending.pop(message.tool_call_id, None)
        if pending:
            await agent.aupdate_state(
                config,
                {
                    "messages": [
                        ToolMessage(
                            content=f"Tool execution stopped: {reason}.",
                            name=call["name"],
                            tool_call_id=call_id,
                            status="error",
                        )
                        for call_id, call in pending.items()
                    ]
                },
                as_node="tools",
            )

    async def cancel(self) -> None:
        """Cancel and await the active graph/tool coroutine; queued prompts remain queued."""
        active = self._active
        if active is not None and not active.done():
            self._cancel_requested = True
            active.cancel()
            with suppress(asyncio.CancelledError):
                await active

    async def aclose(self) -> None:
        """Cancel active work and close all owned clients exactly once."""
        from lightspeed_agentic.providers.deepagents import _close_model_clients

        self._closed = True
        await self.cancel()
        resources, self._resources = self._resources, {}
        try:
            if resources.get("chat_model") is not None:
                await _close_model_clients(resources["chat_model"])
        finally:
            if resources.get("classifier_model") is not None:
                await _close_model_clients(resources["classifier_model"])


def create_session_tool_middleware(
    session: RuntimeSession,
    *,
    consume_steering: bool = True,
) -> Any:
    """Ask permission immediately before SDK tool execution, independently of ACP."""
    from langchain.agents.middleware import AgentMiddleware
    from langchain_core.messages import ToolMessage

    class SessionToolMiddleware(AgentMiddleware[Any, Any, Any]):
        async def abefore_model(self, state: Any, runtime: Any) -> Any:
            del state, runtime
            if consume_steering and session._steering:
                return {"messages": session._take_steering()}
            return None

        async def awrap_model_call(self, request: Any, handler: Any) -> Any:
            # Installed inside inspection: final-cycle tool results still cross both
            # safety gates, while the next model call never starts.
            if consume_steering and session.turns_used >= session._max_turns:
                raise SessionTurnLimitError()
            return await handler(request)

        async def awrap_tool_call(self, request: Any, handler: Any) -> Any:
            call = request.tool_call
            if session.mode == "approve" and not (
                call["name"] == "ask_user" and session.user_input_callback is not None
            ):
                allowed = (
                    session.permission_callback is not None
                    and await session.permission_callback(
                        ToolPermissionRequest(call["name"], call["args"], call["id"])
                    )
                )
                if not allowed:
                    return ToolMessage(
                        content="Tool execution denied by user.",
                        name=call["name"],
                        tool_call_id=call["id"],
                        status="error",
                    )
            return await handler(request)

    return SessionToolMiddleware()


def create_ask_user_tool(session: RuntimeSession) -> Any:
    """Register the trusted user callback as a local tool rather than an MCP proxy."""
    from typing import Annotated

    from langchain_core.tools import InjectedToolCallId, tool

    async def ask_user(
        question: str,
        tool_call_id: Annotated[str, InjectedToolCallId],
        options: list[str] | None = None,
    ) -> str:
        """Ask the user a question, optionally offering a list of answer options."""
        callback = session.user_input_callback
        if callback is None:
            raise RuntimeError("No user input callback configured")
        return await callback(
            UserInputRequest(
                question=question, options=tuple(options or ()), tool_call_id=tool_call_id
            )
        )

    # Provider SDK imports remain optional; materialize the injected annotation
    # before LangChain resolves the function's postponed annotations.
    ask_user.__annotations__["tool_call_id"] = Annotated[str, InjectedToolCallId]
    return tool(ask_user)
