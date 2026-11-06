"""Bridge to the launcher's trusted ask_user MCP stdio server.

This is separate from remote MCP admission. The launcher pins the executable;
clients cannot choose executables or environment. Only ask_user is exposed.
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any

from lightspeed_agentic.acp.config import AskUserConfig


async def ask_user(
    config: AskUserConfig,
    question: str,
    options: tuple[str, ...],
    elicit: Callable[[dict[str, Any]], Awaitable[dict[str, Any]]],
) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client
    from mcp.types import ElicitResult

    answered = False

    async def elicitation(context: Any, params: Any) -> ElicitResult:
        del context
        nonlocal answered
        payload = params.model_dump(by_alias=True, exclude_none=True)
        if not isinstance(payload.get("requestedSchema"), dict):
            raise RuntimeError("invalid ask_user form")
        reply = await elicit(payload)
        content = reply.get("content")
        if (
            reply.get("action") != "accept"
            or not isinstance(content, dict)
            or not isinstance(content.get("answer"), str)
            or not content["answer"].strip()
            or (options and content["answer"] not in options)
        ):
            raise RuntimeError("human input was not supplied")
        answered = True
        return ElicitResult(action="accept", content=content)

    server = StdioServerParameters(command=config.command, args=config.args, env={})
    with open(os.devnull, "w") as errlog:  # noqa: ASYNC230 -- nonblocking special device
        async with (
            stdio_client(server, errlog=errlog) as (reader, writer),
            ClientSession(reader, writer, elicitation_callback=elicitation) as client,
        ):
            await client.initialize()
            tools = await client.list_tools()
            if [tool.name for tool in tools.tools] != ["ask_user"]:
                raise RuntimeError("invalid trusted ask_user server")
            result = await client.call_tool(
                "ask_user", {"question": question, "options": list(options)}
            )
            if not answered or result.isError:
                raise RuntimeError("human input was not supplied")
            return "\n".join(block.text for block in result.content if block.type == "text")
