"""Launcher-owned configuration; clients cannot add executable MCP servers."""

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, field_validator


class AskUserConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    command: StrictStr
    args: list[StrictStr] = Field(default_factory=list)

    @field_validator("command")
    @classmethod
    def absolute_command(cls, value: str) -> str:
        if not Path(value).is_absolute():
            raise ValueError("ask_user command must be absolute")
        return value


class ACPConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    cwd: StrictStr
    system_prompt: StrictStr = "You are an AI agent."
    max_turns: StrictInt = Field(gt=0)
    mode: Literal["auto", "approve"] = "auto"
    ask_user: AskUserConfig | None = None

    @field_validator("cwd")
    @classmethod
    def absolute_cwd(cls, value: str) -> str:
        if not Path(value).is_absolute() or not Path(value).is_dir():
            raise ValueError("cwd must be an existing absolute directory")
        return str(Path(value).resolve())

    def validate_session(self, params: dict[str, Any]) -> None:
        """Accept only the workspace and dedicated executable pinned by the launcher."""
        cwd = params.get("cwd")
        if (
            not isinstance(cwd, str)
            or not Path(cwd).is_absolute()
            or str(Path(cwd).resolve()) != self.cwd
        ):
            raise ValueError("session workspace does not match launcher configuration")
        servers = params.get("mcpServers")
        expected = []
        if self.ask_user is not None:
            expected = [
                {
                    "name": "ask_user",
                    "command": self.ask_user.command,
                    "args": self.ask_user.args,
                    "env": [],
                }
            ]
        if servers != [] and servers != expected:
            raise ValueError("client MCP servers are not permitted")
