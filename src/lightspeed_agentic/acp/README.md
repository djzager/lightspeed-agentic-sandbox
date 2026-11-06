# Optional local ACP adapter

Install the desired model extras plus `acp`, then launch independently of the
batch entrypoint:

```sh
python -m lightspeed_agentic.acp --host 127.0.0.1 --port 4001 --config /path/runtime.json
```

The launcher supplies a private JSON configuration file:

```json
{
  "cwd": "/workspace/repo",
  "system_prompt": "You are an AI agent.",
  "max_turns": 200,
  "mode": "approve",
  "ask_user": {"command": "/usr/local/bin/harness", "args": ["ask-user"]}
}
```

`cwd` must identify an existing absolute directory. `max_turns` is required and
positive. `system_prompt` defaults to a generic instruction; `mode` defaults to
`auto`; `ask_user` is optional. An `approve` launcher policy cannot be reduced
through `session/set_mode`. Normal Lightspeed provider, TLS, reasoning, inspection,
credentials, and remote MCP admission configuration still apply.
`LIGHTSPEED_AGENT_TIMEOUT_SECONDS` supplies the positive per-prompt wall-clock
budget. No model calls occur until `session/prompt`.

The server binds only to `127.0.0.1`, accepts `/acp`, and requires a nonempty
`LIGHTSPEED_ACP_SECRET_KEY`. Clients send that secret with `X-Secret-Key` or the
`token` query parameter. Browser origins are rejected. The adapter uses ACP v1
JSON-RPC messages over the existing harness WebSocket transport; it does not
provide the standard ACP stdio transport.

Supported methods are `initialize`, `session/new`, `session/prompt`,
`session/cancel`, `session/set_mode`, and `session/load`. Text prompts only.
Multiple prompts share the same in-memory runtime/checkpointer. Load replays
normalized, inspected history across viewer connections while the process lives;
there is no restart persistence. Only the creating connection may execute or
cancel a session. Viewer steering uses the harness-compatible
`_goose/unstable/session/steer` request with `expectedRunId`; accepted guidance
enters the conversation at the next main-model boundary. Run/message metadata
uses `_meta.goose` for existing UI compatibility. Terminal model-turn count is
reported as `_meta.lightspeed.turnsUsed` in the prompt response.

## Cumulative usage extension

Token snapshots use a distinct notification method:

```json
{
  "jsonrpc": "2.0",
  "method": "_lightspeed/session/update",
  "params": {
    "sessionId": "session-id",
    "update": {
      "sessionUpdate": "usage_update",
      "inputTokens": 1200,
      "outputTokens": 240
    }
  }
}
```

Counts are cumulative observed session usage and are omitted when unavailable.
They cover the runtime's main, subagent, summarization, and output-shaping calls;
isolated safety-classifier calls are excluded. Terminal snapshots may also carry
`_meta.lightspeed.turnsUsed`, which is the completed main-model count for the most
recent prompt. Consumers replace snapshots rather than adding them. Replay keeps
the original notification method. The adapter does not fabricate monetary cost
or current context occupancy. These cumulative counts are separate from the
[ACP v1 context-usage update](https://github.com/agentclientprotocol/agent-client-protocol/blob/main/docs/protocol/v1/prompt-turn.mdx),
whose `used` and `size` describe current context occupancy.

## Human input and trust boundary

`approve` mode requests `session/request_permission` before tool execution.
Only the advertised one-shot allow selection grants permission; malformed,
cancelled, or unrecognized outcomes fail closed. Questions use the launcher's
dedicated trusted `ask_user` stdio executable, with MCP form elicitation forwarded
as `elicitation/create`. Invalid, empty, cancelled, or unoffered answers fail the
prompt. Cancellation remains responsive while either request awaits a reply.

`session/new.mcpServers` may be empty or exactly match the configured dedicated
`ask_user` executable/arguments with no environment overrides. Other client
executables are rejected. Remote MCP servers still pass normal canonical
admission; this bridge does not create a general stdio-server bypass. Only the
dedicated trusted human answer bypasses tool-output inspection.

SIGTERM cancels the server, closes active sessions and their tool processes, then
closes the provider. Errors are payload-free, helper stderr is suppressed, and
the adapter writes no protocol or model content to stdout. Offline verification
is in `tests/test_acp.py`; it includes concurrent wire operations, a real MCP stdio
bridge, replay, and process-level SIGTERM cleanup.
