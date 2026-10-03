from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.entities import events


@dataclass(slots=True)
class VerboseFlag:
    enabled: bool = False

    def toggle(self) -> bool:
        """Flip the flag in place and return the NEW state (what the keybind renders)."""
        self.enabled = not self.enabled
        return self.enabled


EventSink = Callable[[events.Event], None]


@dataclass(slots=True)
class AgentDeps:
    """
    PydanticAI dependency injection
    """

    cwd: Path
    emit: EventSink
    verbose: VerboseFlag = field(default_factory=VerboseFlag)
    harness_home: Path | None = None
    context_window_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.harness_home is None:
            self.harness_home = self.cwd
