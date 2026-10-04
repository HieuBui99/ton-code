from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class TurnStarted:
    turn_id: int
    prompt: str
    kind: Literal["turn_started"] = "turn_started"


@dataclass(frozen=True, slots=True)
class TurnFinished:
    turn_id: int
    aborted: bool = False
    kind: Literal["turn_finished"] = "turn_finished"


@dataclass(frozen=True, slots=True)
class AssistantTextDelta:
    text: str
    kind: Literal["assistant_text_delta"] = "assistant_text_delta"


@dataclass(frozen=True, slots=True)
class ThinkingDelta:
    text: str
    kind: Literal["thinking_delta"] = "thinking_delta"


@dataclass(frozen=True, slots=True)
class ToolCallStarted:
    tool_call_id: str
    name: str
    args: str
    child_index: int | None = None
    kind: Literal["tool_call_started"] = "tool_call_started"


@dataclass(frozen=True, slots=True)
class ToolResult:
    tool_call_id: str
    name: str
    output: str
    ok: bool = True
    kind: Literal["tool_result"] = "tool_result"


@dataclass(frozen=True, slots=True)
class ContextUsage:
    input_tokens: int
    estimated: bool = False
    kind: Literal["context_usage"] = "context_usage"


@dataclass(frozen=True, slots=True)
class ContextCompacted:
    before_tokens: int
    kept_messages: int
    kind: Literal["context_compacted"] = "context_compacted"


@dataclass(frozen=True, slots=True)
class ContextMicrocompacted:
    elided_count: int
    before_tokens: int
    kind: Literal["context_microcompacted"] = "context_microcompacted"


@dataclass(frozen=True, slots=True)
class AgentError:
    message: str
    kind: Literal["agent_error"] = "agent_error"


Event = (
    TurnStarted
    | TurnFinished
    | AssistantTextDelta
    | ThinkingDelta
    | ToolCallStarted
    | ToolResult
    | ContextUsage
    | ContextCompacted
    | ContextMicrocompacted
    | AgentError
)
