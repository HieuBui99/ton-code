from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

# The three states a task can be in. ``get_args`` derives the validation set from this single
# annotation so the allowed values are declared in exactly one place.
TaskStatus = Literal["pending", "in_progress", "completed"]
_VALID_STATUSES: frozenset[str] = frozenset(get_args(TaskStatus))


@dataclass(frozen=True, slots=True)
class Task:
    id: str
    content: str
    status: TaskStatus = "pending"

    def __post_init__(self) -> None:
        if self.status not in _VALID_STATUSES:
            allowed = ", ".join(sorted(_VALID_STATUSES))
            raise ValueError(f"invalid task status {self.status!r}; expected one of: {allowed}")
        if not self.content.strip():
            raise ValueError("task content must be a non-empty description")
