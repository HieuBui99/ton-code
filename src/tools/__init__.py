from __future__ import annotations

import enum

from src.tools.registry import TOOL_KIND

KNOWN_TOOL_NAMES: frozenset[str] = frozenset(TOOL_KIND) 

__all__ = ["KNOWN_TOOL_NAMES", "TOOL_KIND", "tool_kind"]


class ToolKind(enum.Enum):
    READ_ONLY = "read_only"
    FILE_EDIT = "file_edit"
    OTHER = "other"

def tool_kind(tool_name: str) -> ToolKind:
    return TOOL_KIND.get(tool_name, ToolKind.OTHER)
