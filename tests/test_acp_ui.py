"""Wire regressions matching tackle2-ui's ACP client and ChatPanel protocol."""

import json

import pytest
from websockets.asyncio.client import connect

from .test_acp import initialize, prompt, receive, response, send, server

pytestmark = pytest.mark.asyncio


async def test_ui_workspace_alias_new_and_reconnect_load_keep_execution_cwd_pinned(tmp_path):
    async with server(tmp_path, config_overrides={"cwd_aliases": ["/workspace"]}) as (
        url,
        provider,
        config,
    ):
        async with connect(url) as viewer:
            await send(viewer, 1, "initialize", {"protocolVersion": 1, "clientCapabilities": {}})
            await response(viewer, 1)
            await send(viewer, 2, "session/new", {"cwd": "/workspace", "mcpServers": []})
            created, _ = await response(viewer, 2)
            viewer_id = created["result"]["sessionId"]
            assert provider.options[0].cwd == config.cwd
            assert not provider.sessions[0].prompts
        async with connect(url) as reconnected:
            await send(reconnected, 1, "initialize", {"protocolVersion": 1})
            await response(reconnected, 1)
            await send(
                reconnected,
                2,
                "session/load",
                {"sessionId": viewer_id, "cwd": "/workspace", "mcpServers": []},
            )
            loaded, _ = await response(reconnected, 2)
            assert loaded["result"] == {}
            assert len(provider.sessions) == 1
            await send(
                reconnected, 3, "session/new", {"cwd": "/workspace/../etc", "mcpServers": []}
            )
            assert (await receive(reconnected))["error"]["code"] == -32602
            assert len(provider.sessions) == 1


async def test_unconfigured_ui_workspace_alias_is_rejected(tmp_path):
    async with server(tmp_path) as (url, provider, _config), connect(url) as viewer:
        await send(viewer, 1, "initialize", {"protocolVersion": 1})
        await response(viewer, 1)
        await send(viewer, 2, "session/new", {"cwd": "/workspace", "mcpServers": []})
        assert (await receive(viewer))["error"]["code"] == -32602
        assert not provider.sessions


@pytest.mark.parametrize("cwd", ["/etc", "/workspace/child", "workspace", "/workspace/../etc"])
async def test_workspace_alias_never_accepts_arbitrary_client_paths(tmp_path, cwd):
    async with (
        server(tmp_path, config_overrides={"cwd_aliases": ["/workspace"]}) as (
            url,
            provider,
            _config,
        ),
        connect(url) as viewer,
    ):
        await send(viewer, 1, "initialize", {"protocolVersion": 1})
        await response(viewer, 1)
        await send(viewer, 2, "session/new", {"cwd": cwd, "mcpServers": []})
        assert (await receive(viewer))["error"]["code"] == -32602
        assert not provider.sessions


@pytest.mark.parametrize("text", ["first", "limit", "secret failure", "wait"])
async def test_ui_active_run_metadata_cleared_on_every_terminal_path(tmp_path, text):
    async with server(tmp_path) as (url, _provider, config), connect(url) as owner:
        session_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(session_id, text))
        started = await receive(owner)
        active = started["params"]["update"]["_meta"]["goose"]["activeRunId"]
        assert isinstance(active, str)
        assert active
        if text == "wait":
            await owner.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/cancel",
                        "params": {"sessionId": session_id},
                    }
                )
            )
        result, updates = await response(owner, 3)
        terminal = [
            frame
            for frame in updates
            if frame.get("params", {}).get("update", {}).get("sessionUpdate")
            == "session_info_update"
        ]
        assert terminal[-1]["params"]["update"]["_meta"]["goose"]["activeRunId"] is None
        assert ("error" in result) == (text == "secret failure")


async def test_ui_active_run_is_cleared_when_prompt_times_out(tmp_path):
    async with (
        server(tmp_path, timeout_seconds=0.02) as (url, _provider, config),
        connect(url) as owner,
    ):
        session_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(session_id, "wait"))
        result, updates = await response(owner, 3)
        assert result["error"]["code"] == -32603
        lifecycle = [
            frame["params"]["update"]["_meta"]["goose"]["activeRunId"]
            for frame in updates
            if frame["params"]["update"]["sessionUpdate"] == "session_info_update"
        ]
        assert isinstance(lifecycle[0], str)
        assert lifecycle[-1] is None


async def test_ui_steer_discovery_retry_and_idle_error_data(tmp_path):
    async with server(tmp_path) as (url, provider, config), connect(url) as owner:
        session_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(session_id, "steer"))
        started = await receive(owner)
        run_id = started["params"]["update"]["_meta"]["goose"]["runId"]
        await receive(owner)
        for request_id, expected in ((4, "discover"), (5, "stale")):
            await send(
                owner,
                request_id,
                "_goose/unstable/session/steer",
                {**prompt(session_id, "use postgres"), "expectedRunId": expected},
            )
            error, _ = await response(owner, request_id)
            assert error["error"]["code"] == -32602
            assert error["error"]["data"] == {"actualRunId": run_id}
            assert provider.sessions[0].steering.empty()
        await send(
            owner,
            6,
            "_goose/unstable/session/steer",
            {**prompt(session_id, "use postgres"), "expectedRunId": run_id},
        )
        accepted, updates = await response(owner, 6)
        assert accepted["result"]["messageId"] == "steer-message-1"
        terminal, remainder = await response(owner, 3)
        assert terminal["result"]["stopReason"] == "end_turn"
        delivered = [
            frame
            for frame in updates + remainder
            if frame.get("params", {}).get("update", {}).get("sessionUpdate")
            == "user_message_chunk"
        ]
        assert delivered[-1]["params"]["update"]["_meta"]["goose"]["messageId"] == "steer-message-1"
        await send(
            owner,
            7,
            "_goose/unstable/session/steer",
            {**prompt(session_id, "too late"), "expectedRunId": run_id},
        )
        error, _ = await response(owner, 7)
        assert error["error"]["code"] == -32602
        assert "no active run" in error["error"]["data"]


async def test_silent_ui_reconnect_replays_only_launcher_primary_with_original_ids(tmp_path):
    overrides = {"cwd_aliases": ["/workspace"], "replay_primary_session": True}
    async with (
        server(tmp_path, config_overrides=overrides) as (url, _provider, config),
        connect(url) as owner,
    ):
        primary_id = await initialize(owner, config)
        await send(owner, 3, "session/prompt", prompt(primary_id, "wait"))
        await receive(owner)
        await receive(owner)
        async with connect(url) as secondary:
            secondary_id = await initialize(secondary, config)
            await send(secondary, 3, "session/prompt", prompt(secondary_id, "private secondary"))
            await response(secondary, 3)
            async with connect(url) as viewer:
                await send(viewer, 1, "initialize", {"protocolVersion": 1})
                await response(viewer, 1)
                await send(viewer, 2, "session/new", {"cwd": "/workspace", "mcpServers": []})
                created, _ = await response(viewer, 2)
                viewer_id = created["result"]["sessionId"]
            async with connect(url) as reconnect:
                await send(reconnect, 1, "initialize", {"protocolVersion": 1})
                await response(reconnect, 1)
                await send(
                    reconnect,
                    2,
                    "session/load",
                    {"sessionId": viewer_id, "cwd": "/workspace", "mcpServers": []},
                )
                loaded, updates = await response(reconnect, 2)
                assert loaded["result"] == {}
                assert updates
                assert {frame["params"]["sessionId"] for frame in updates} == {primary_id}
                assert "private secondary" not in json.dumps(updates)
                assert updates[0]["params"]["update"]["content"]["text"] == "wait"
                await send(reconnect, 3, "session/prompt", prompt(primary_id, "take ownership"))
                assert (await receive(reconnect))["error"]["code"] == -32000
                await send(
                    reconnect,
                    4,
                    "session/load",
                    {"sessionId": secondary_id, "cwd": "/workspace", "mcpServers": []},
                )
                _, secondary_updates = await response(reconnect, 4)
                assert {frame["params"]["sessionId"] for frame in secondary_updates} == {
                    secondary_id
                }
        await owner.send(
            json.dumps(
                {"jsonrpc": "2.0", "method": "session/cancel", "params": {"sessionId": primary_id}}
            )
        )
        result, _ = await response(owner, 3)
        assert result["result"]["stopReason"] == "cancelled"


async def test_closed_silent_viewer_sessions_do_not_exhaust_session_limit(tmp_path):
    async with server(tmp_path) as (url, provider, config), connect(url) as owner:
        primary_id = await initialize(owner, config)
        for _ in range(18):
            async with connect(url) as viewer:
                await initialize(viewer, config)
        await send(owner, 3, "session/prompt", prompt(primary_id, "first"))
        result, _ = await response(owner, 3)
        assert result["result"]["stopReason"] == "end_turn"
        assert provider.sessions[0].prompts == [("first", 7)]
