from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rich.console import Console
from rich.text import Text

from src.entities import events

# Strip terminal instructions before Rich adds its own trusted colour sequences.
_ESCAPES = re.compile(
    r"(?:\x1b\]|\x9d)[^\x07\x1b\x9c]*(?:\x07|\x1b\\|\x9c|$)"
    r"|(?:\x1b[P^_X]|\x90)[\s\S]*?(?:\x1b\\|\x9c|$)"
    r"|(?:\x1b\[|\x9b)[0-?]*[ -/]*[@-~]"
    r"|\x1b[ -/]*[@-~]"
)
_CONTROLS = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def clean_text(value: str) -> str:
    return _CONTROLS.sub("", _ESCAPES.sub("", value.replace("\r\n", "\n")))


@dataclass(frozen=True, slots=True)
class UserSubmitted:
    text: str
    steering: bool = False


@dataclass(frozen=True, slots=True)
class Notice:
    text: str
    style: str = "dim"


TranscriptEvent = events.Event | UserSubmitted | Notice


def preview(value: str, *, lines: int = 10, characters: int = 2000) -> str:
    value = clean_text(value)
    shortened = "\n".join(value.split("\n")[:lines])[:characters]
    if len(shortened) < len(value):
        shortened += "\n… output omitted; use --verbose for full emitted output"
    return shortened


class TranscriptRenderer:
    """Turn events into permanent lines, never repainting the transcript."""

    def __init__(self, console: Console, *, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._pending = ""
        self._stream: str | None = None
        self._labelled = False
        self._tools: dict[str, str] = {}

    def _print(self, text: str, style: str = "") -> None:
        self.console.print(Text(clean_text(text), style=style), soft_wrap=True)

    def _line(self, text: str) -> None:
        if not self._labelled:
            self._print(
                "Thinking" if self._stream == "thinking" else "Assistant",
                "dim" if self._stream == "thinking" else "bold cyan",
            )
            self._labelled = True
        self._print(text, "dim" if self._stream == "thinking" else "")

    def flush(self) -> None:
        if self._pending:
            self._line(self._pending)
        self._pending = ""
        self._stream = None
        self._labelled = False

    def render(self, event: TranscriptEvent) -> None:
        if isinstance(event, events.ContextUsage):
            return  # Status updates must not split a pending assistant text line.
        if isinstance(event, events.ThinkingDelta) and not self.verbose:
            return
        if isinstance(event, events.AssistantTextDelta | events.ThinkingDelta):
            stream = (
                "thinking" if isinstance(event, events.ThinkingDelta) else "assistant"
            )
            if stream != self._stream:
                self.flush()
                self._stream = stream
            # Sanitize after assembling a line: escape sequences may span deltas.
            self._pending += event.text
            while "\n" in self._pending:
                line, self._pending = self._pending.split("\n", 1)
                self._line(line)
            return

        self.flush()
        if isinstance(event, UserSubmitted):
            self._print("You (steering)" if event.steering else "You", "bold green")
            self._print(event.text)
        elif isinstance(event, Notice):
            self._print(event.text, event.style)
        elif isinstance(event, events.TurnStarted):
            pass  # UserSubmitted already records the prompt exactly once.
        elif isinstance(event, events.ToolCallStarted):
            self._tools[event.tool_call_id] = event.name
            args = event.args
            if not self.verbose:
                try:
                    args = json.dumps(json.loads(args), ensure_ascii=False)
                except ValueError, TypeError:
                    pass
                args = clean_text(args).replace("\n", " ")
                args = args[:200] + ("…" if len(args) > 200 else "")
            self._print(f"Tool {event.name} [{event.tool_call_id}]: {args}", "dim")
        elif isinstance(event, events.ToolResult):
            name = self._tools.pop(event.tool_call_id, event.name)
            self._print(
                f"{'OK' if event.ok else 'FAILED'} {name} [{event.tool_call_id}]",
                "green" if event.ok else "red",
            )
            self._print(
                clean_text(event.output) if self.verbose else preview(event.output),
                "dim" if event.ok else "red",
            )
        elif isinstance(event, events.ContextCompacted):
            self._print(
                f"Context compacted: {event.before_tokens:,} input tokens; {event.kept_messages} recent messages kept.",
                "dim",
            )
        elif isinstance(event, events.ContextMicrocompacted):
            self._print(
                f"Context trimmed: {event.elided_count} old tool outputs removed.",
                "dim",
            )
        elif isinstance(event, events.AgentError):
            self._print(f"Error: {event.message}", "red")
        elif isinstance(event, events.TurnFinished):
            self._tools.clear()
            self._print(
                "Turn interrupted." if event.aborted else "Turn complete.", "dim"
            )
