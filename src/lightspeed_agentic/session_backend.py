"""Cancellation support for the SDK's otherwise thread-backed local shell tool."""

from __future__ import annotations

import asyncio
import os
import signal
from contextlib import suppress
from typing import Any


def cancellable_shell_backend() -> Any:
    """Keep SDK filesystem behavior while making shell process lifetime awaitable."""
    from deepagents.backends import LocalShellBackend
    from deepagents.backends.protocol import ExecuteResponse

    class CancellableLocalShellBackend(LocalShellBackend):
        async def aexecute(
            self,
            command: str,
            *,
            timeout: int | None = None,  # noqa: ASYNC109
        ) -> ExecuteResponse:
            if not command or not isinstance(command, str):
                return ExecuteResponse(
                    output="Error: Command must be a non-empty string.",
                    exit_code=1,
                    truncated=False,
                )
            effective_timeout = self._default_timeout if timeout is None else timeout
            if effective_timeout <= 0:
                raise ValueError(f"timeout must be positive, got {effective_timeout}")
            # Shield startup so cancellation cannot orphan a process before its handle
            # is returned. POSIX process groups include shell-spawned child commands.
            spawn = asyncio.create_task(
                asyncio.create_subprocess_shell(
                    command,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=self._env,
                    cwd=str(self.cwd),
                    start_new_session=True,
                )
            )
            try:
                process = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                process = await spawn
                await _kill_process_group(process)
                raise
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=effective_timeout,
                )
            except asyncio.CancelledError:
                await _kill_process_group(process)
                raise
            except TimeoutError:
                await _kill_process_group(process)
                suffix = (
                    " (custom timeout). The command may be stuck or require more time."
                    if timeout is not None
                    else ". For long-running commands, re-run using the timeout parameter."
                )
                return ExecuteResponse(
                    output=f"Error: Command timed out after {effective_timeout} seconds{suffix}",
                    exit_code=124,
                    truncated=False,
                )
            parts = []
            if stdout:
                parts.append(stdout.decode(errors="replace"))
            if stderr:
                parts.extend(
                    f"[stderr] {line}"
                    for line in stderr.decode(errors="replace").strip().split("\n")
                )
            output = "\n".join(parts) if parts else "<no output>"
            truncated = len(output) > self._max_output_bytes
            if truncated:
                output = output[: self._max_output_bytes]
                output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            if process.returncode != 0:
                output = f"{output.rstrip()}\n\nExit code: {process.returncode}"
            return ExecuteResponse(output=output, exit_code=process.returncode, truncated=truncated)

    return CancellableLocalShellBackend


async def _kill_process_group(process: asyncio.subprocess.Process) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    await process.communicate()
