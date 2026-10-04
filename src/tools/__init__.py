from __future__ import annotations

from src.tools.kinds import ToolKind
from src.tools.registry import TOOL_KIND

KNOWN_TOOL_NAMES: frozenset[str] = frozenset(TOOL_KIND)

__all__ = ["KNOWN_TOOL_NAMES", "TOOL_KIND", "ToolKind", "tool_kind"]


def tool_kind(tool_name: str) -> ToolKind:
    return TOOL_KIND.get(tool_name, ToolKind.OTHER)
