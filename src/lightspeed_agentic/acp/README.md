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
  "cwd_aliases": ["/workspace"],
  "replay_primary_session": true,
  "system_prompt": "You are an AI agent.",
  "max_turns": 200,
  "mode": "approve",
  "ask_user": {"command": "/usr/local/bin/harness", "args": ["ask-user"]}
}
```

`cwd` must identify an existing absolute directory. Optional `cwd_aliases` are
explicit launcher-owned logical client paths, for example the UI's `/workspace`.
They default to empty and never change the execution directory or expose another
filesystem root. Unknown paths remain rejected. `max_turns` is required and
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
authorized viewer history across viewer connections while the process lives;
there is no restart persistence. The optional `replay_primary_session` setting
defaults to false. When enabled, the first successfully created session is the
launcher's primary session; the launcher must create it before exposing any
viewer transport. The harness enforces this by creating its session before
starting its tee. Loading a silent viewer session then replays only that primary
session's inspected history with its original session IDs. Other prompted
sessions retain their own history and are never combined. Closed silent viewer
sessions can be evicted when the session limit is reached; the UI falls back to
creating a new viewer session if an old one was evicted.
Only the creating connection may execute or
cancel a session. Viewer steering uses the harness-compatible
`_goose/unstable/session/steer` request with `expectedRunId`; accepted guidance
enters the conversation at the next main-model boundary. Run/message metadata
uses `_meta.goose` for existing UI compatibility. A `session_info_update` announces
`activeRunId` at prompt start and clears it to null on every completion, error,
timeout, or cancellation path. The existing `runId` alias remains available.
A stale or discovery steering request is rejected with `-32602` and
`error.data.actualRunId`; no active run returns `-32602` with a data string
containing `no active run`. Neither case queues guidance. Terminal model-turn count is
reported as `_meta.lightspeed.turnsUsed` in the prompt response.
The owner can reduce an individual prompt's budget with
`_meta.lightspeed.maxTurns`, a strictly positive integer clamped to the configured
`max_turns` ceiling. The harness uses this for the remaining handoff allowance;
omitting it keeps the configured ceiling. Budgets count completed main-model
calls, and tools selected by the final allowed call still execute and pass
inspection before the next model call is stopped.

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

## Model output limits

Set `LIGHTSPEED_MODEL_MAX_OUTPUT_TOKENS=16384` explicitly when a model needs a
larger per-response allowance than its SDK default. The setting applies to the
main graph model; separate safety-classifier and shaping calls keep their existing
settings. It is optional, must be a positive integer, and does not change monetary
or turn budgets. Reasoning can share the output allowance with answer text.

A provider-reported output cutoff fails the active prompt through ACP's existing
runtime-operation error. Its observed usage remains available, and a subsequent
explicit prompt can continue the same in-memory session. The adapter does not
automatically repeat a costly or truncated request.

For Anthropic-compatible models that support effort controls, an explicit setting
such as `LIGHTSPEED_REASONING_CONFIG='{"thinking":{"type":"enabled"},"effort":"low"}'`
keeps thinking enabled and maps effort to the native `output_config.effort` field.
The native `output_config` form is also accepted; normalized `effort` takes
precedence when both are supplied. No effort or thinking default is imposed, and
model/API support remains provider-specific. This configuration applies to the
agent pass; classifier and shaping calls keep reasoning unconfigured.

## Official server and native graph composition

The optional extra pins `deepagents-acp` to the immutable upstream source commit
`f2a6e02dc8e2dcb05885c5133b94102ccc79b794` (0.0.13) and ACP SDK 0.10.1.
`uv sync --frozen --extra dev` installs the tested dependency set. Production
batch extras, model defaults and backend selection remain unchanged. The Git
source pin is intentional: matching a PyPI version alone does not establish
that it contains the tested implementation.

`DeepAgentsProvider.prepare_graph()` supplies the existing native compiled graph,
backend, admitted tools, middleware and main/classifier clients. `RuntimeSession`
retains the graph/checkpointer and owns cancellation, steering, model-turn limits,
cumulative usage and cleanup. `LightspeedACPServer` subclasses the official
`AgentServerACP`; the official SDK router decodes standard methods, and the
actual official `prompt()` loop renders the graph's ACP messages and tool calls.
Lightspeed wraps that loop with launcher/owner policy and its existing callback
permissions/questions. It does not use the upstream HITL permission mechanism
or copy the upstream prompt loop.

`InspectedGraph` passes native frames through while buffering tool results. It
keeps pulling the graph so inspection can finish. Only a completed boundary's
acceptance releases pending occurrences with the exact tool name, result/error
status, call ID and content signature from that same subgraph namespace.
Classification-cache entries do not authorize output. Child graph frames remain
internal; they cannot authorize parent output. Buffers fail closed above 256
pending results or 8 MiB. Rejection, classifier failure, missing acceptance,
content/status changes and cancellation never release those pending payloads.
The normal model-input inspection middleware stays in its existing order,
including inspection of tools selected by the final allowed model call.

Viewer history records authorized publication attempts and incoming user prompts,
including consumed steering, original session/message IDs and cumulative usage
extensions. It never reconstructs display permission from checkpoint history.
The journal keeps a 16 MiB tail per session; truncated replay announces
`_meta.lightspeed.historyTruncated`. A viewer load does not transfer ownership,
execute tools, repeat a pending permission/question request, consume its answer,
or cancel the owning connection's active work. Pending replies remain attached
to the owner's original request and cancellation cleans them up. This journal
is process-local viewer replay, not durable ACP conversation recovery or proof
that a peer received a publication. There is no recovery after process exit.

The integration depends on five private upstream methods, all pinned and
covered by native compatibility/regression tests:

- `_reset_agent`: select the session's prepared native graph view.
- `_session_config`: use the same native thread for streaming, state and repair.
- `_persist_session`: keep metadata in RuntimeSession rather than pretending
  the viewer journal is a durable checkpoint store.
- `_replay_session`: replay only the authorized viewer journal.
- `_process_tool_call_chunks`: retain inspected `edit_file` completion output
  that upstream otherwise suppresses after its input diff. Failed tool results
  retain `failed` status through the publication wrapper.

Execution is active during asynchronous graph preparation and native streaming.
Cancel awaits the actual model/tool/classifier/question coroutine and repairs
unanswered tool calls before a later prompt can continue. Timeout and owner
cleanup use the same repair path. After the final graph boundary, steering
admission closes before shaping and terminal publication; late steering is
rejected. Terminal notification writes cannot let a late cancel remove the
pending prompt response. Accepted steering spends the same prompt's remaining
main-model allowance. Optional programmatic structured output still uses the
existing tool-free shaping pass and counts its observed usage without spending
a main-model turn.

`tests/test_acp_native.py` exercises the actual official class and native
Lightspeed/DeepAgents graphs with scripted offline models and classifiers.
It covers delayed acceptance, whole-batch failure, exact occurrence/namespace
matching, rejection/pending replay, two prompts, budget handoff, preparation and
model cancellation, timeout repair, shaping, failed edit results and terminal
steering/cancel races. `tests/test_acp.py` and `tests/test_acp_ui.py` cover real
loopback WebSocket framing, host policy, permission/question concurrency, trusted
stdio cleanup and existing UI wire behavior. No paid model or cluster is needed.
