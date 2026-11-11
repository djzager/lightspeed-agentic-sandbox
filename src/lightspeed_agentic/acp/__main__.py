"""Launch ACP separately from the batch entrypoint: python -m lightspeed_agentic.acp."""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import signal
from pathlib import Path

from lightspeed_agentic.acp.config import ACPConfig
from lightspeed_agentic.acp.server import serve


async def run(config: ACPConfig, *, host: str, port: int) -> None:
    from lightspeed_agentic.config import (
        parse_agent_timeout,
        parse_model_max_output_tokens,
        parse_reasoning_config,
        parse_tool_output_inspection_enabled,
        resolve_model_family,
    )
    from lightspeed_agentic.mcp import (
        discover_and_admit_mcp_servers,
        parse_mcp_servers,
        render_mcp_policy_context,
        split_admitted_mcp_servers,
    )
    from lightspeed_agentic.providers.deepagents import DeepAgentsProvider
    from lightspeed_agentic.providers.model_router import resolve_model_route
    from lightspeed_agentic.readiness import run_readiness_checks
    from lightspeed_agentic.tls import configure_tls
    from lightspeed_agentic.types import ProviderQueryOptions

    configure_tls()
    family = resolve_model_family()
    route = resolve_model_route(model_family=family)
    ready, _checks = run_readiness_checks(family, route)
    if not ready:
        raise RuntimeError("provider readiness failed")
    admitted = await discover_and_admit_mcp_servers(parse_mcp_servers())
    servers, policies = split_admitted_mcp_servers(admitted)
    policy_context = render_mcp_policy_context(policies)
    options = ProviderQueryOptions(
        prompt="",
        system_prompt="\n\n".join(filter(None, [config.system_prompt, policy_context])),
        max_turns=config.max_turns,
        cwd=config.cwd,
        mcp_servers=servers,
        reasoning_config=parse_reasoning_config(),
        max_output_tokens=parse_model_max_output_tokens(),
        tool_output_inspection_enabled=parse_tool_output_inspection_enabled(),
    )
    provider = DeepAgentsProvider(route)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None  # noqa: S101 -- run() always executes inside an asyncio task
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        await serve(
            provider,
            config,
            options,
            host=host,
            port=port,
            secret=os.environ.get("LIGHTSPEED_ACP_SECRET_KEY", ""),
            timeout_seconds=parse_agent_timeout(),
        )
    finally:
        try:
            await provider.aclose()
        finally:
            loop.remove_signal_handler(signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description="Lightspeed optional local ACP v1 server")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1"])
    parser.add_argument("--port", type=int, default=4001)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)
    try:
        config = ACPConfig.model_validate_json(args.config.read_text())
        asyncio.run(run(config, host=args.host, port=args.port))
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    except Exception:
        # Never expose configuration values, model responses, credentials, or tokens.
        parser.exit(1, "Lightspeed ACP startup or runtime failed.\n")


if __name__ == "__main__":
    main()
