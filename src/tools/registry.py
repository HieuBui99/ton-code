from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

from pydantic_ai import Agent, DeferredToolRequests

from src.agent.deps import AgentDeps
from src.tools import ToolKind, files
from src.tools import bash as bash_module
from src.tools import web as web_module

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    name: str
    func: Callable[..., object]
    kind: ToolKind
    retries: int | None = None


# The flat catalogue. Source of truth for registration and the tool-kind map.
TOOL_SPECS: list[ToolSpec] = [
    # Read-only file tools: no disk/exec side effect → auto-allowed by the gate.
    ToolSpec(name=files.READ_TOOL_NAME, func=files.read, kind=ToolKind.READ_ONLY),
    ToolSpec(name=files.GLOB_TOOL_NAME, func=files.glob, kind=ToolKind.READ_ONLY),
    ToolSpec(name=files.GREP_TOOL_NAME, func=files.grep, kind=ToolKind.READ_ONLY),
    # Mutating file tools: FILE_EDIT — edit mode auto-allows them, default asks.
    ToolSpec(name=files.WRITE_TOOL_NAME, func=files.write, kind=ToolKind.FILE_EDIT),
    ToolSpec(name=files.EDIT_TOOL_NAME, func=files.edit, kind=ToolKind.FILE_EDIT),
    # Bash: shell execution behind the executor seam → OTHER (edit mode still asks).
    ToolSpec(name=bash_module.BASH_TOOL_NAME, func=bash_module.bash, kind=ToolKind.OTHER),

    # Web: httpx GET → Markdown; network egress only, no local side effect → READ_ONLY.
    ToolSpec(
        name=web_module.WEB_FETCH_TOOL_NAME,
        func=web_module.web_fetch,
        kind=ToolKind.READ_ONLY,
    )
]

# Each tool's kind, derived from TOOL_SPECS; unknown tools default to OTHER (mutating → gated).
TOOL_KIND: dict[str, ToolKind] = {spec.name: spec.kind for spec in TOOL_SPECS}


def register_tools(agent: Agent[AgentDeps, str | DeferredToolRequests]) -> None:
    """
    Register every :data:`TOOL_SPECS` function on ``agent`` with per-agent restriction
    """
    for spec in TOOL_SPECS:
        agent.tool(spec.func, retries=spec.retries)
    logger.debug("registered %d tools: %s", len(TOOL_SPECS), [s.name for s in TOOL_SPECS])
