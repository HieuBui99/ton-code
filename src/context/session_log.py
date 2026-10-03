from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from pydantic_ai.messages import ModelMessagesTypeAdapter

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage

logger = logging.getLogger(__name__)

# Header schema version — bump if the on-disk line format changes.
_HEADER_VERSION = 1
# Typed-line discriminators: header / turn messages / compaction checkpoint / clear marker.
_HEADER_TYPE = "session"
_MESSAGES_TYPE = "messages"
_COMPACTION_TYPE = "compaction"
_CLEAR_TYPE = "clear"
_SUFFIX = ".jsonl"
# Compact UTC filename timestamp, lexically sortable — load_latest takes the lexicographic max.
_FILENAME_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(slots=True)
class SessionLog:
    path: Path

    @property
    def session_id(self) -> str:
        return self.path.stem.split("_", 1)[-1]

    @classmethod
    def create(
        cls,
        sessions_dir: Path,
        *,
        cwd: Path,
        now: datetime | None = None,
        session_id: UUID | None = None,
    ) -> SessionLog:
        resolved_now = now or _utc_now()
        if resolved_now.tzinfo is None:
            raise ValueError("now must be a timezone-aware (UTC) datetime, not naive")
        resolved_now = resolved_now.astimezone(UTC)
        resolved_id = session_id or uuid4()

        sessions_dir.mkdir(parents=True, exist_ok=True)
        path = sessions_dir / f"{resolved_now:{_FILENAME_TS_FORMAT}}_{resolved_id}{_SUFFIX}"

        header = {
            "type": _HEADER_TYPE,
            "version": _HEADER_VERSION,
            "session_id": str(resolved_id),
            "cwd": str(cwd),
            "created_at": resolved_now.isoformat(),
        }
        with path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps(header) + "\n")
        logger.debug("opened session log %s", path)
        return cls(path=path)

    def append_turn(self, new_messages: list[ModelMessage]) -> None:
        if not new_messages:
            return
        payload = json.loads(ModelMessagesTypeAdapter.dump_json(new_messages))
        entry = {"type": _MESSAGES_TYPE, "messages": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        logger.debug("appended %d message(s) to %s", len(new_messages), self.path)

    def append_compaction(self, summary_message: ModelMessage, tail: list[ModelMessage]) -> None:
        summary_payload = json.loads(ModelMessagesTypeAdapter.dump_json([summary_message]))
        tail_payload = json.loads(ModelMessagesTypeAdapter.dump_json(tail))
        entry = {
            "type": _COMPACTION_TYPE,
            "summary": summary_payload,
            "tail": tail_payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        logger.debug("appended compaction checkpoint (tail=%d) to %s", len(tail), self.path)

    def append_clear(self) -> None:
        entry = {"type": _CLEAR_TYPE}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        logger.debug("appended clear marker to %s", self.path)


def load(path: Path) -> list[ModelMessage]:
    if not path.is_file():
        return []

    history: list[ModelMessage] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if _is_clear_line(line):
            history = []  # /clear marker: compaction-to-zero — discard prior, restart empty
            continue
        compacted = _parse_compaction_line(line)
        if compacted is not None:
            history = compacted  # checkpoint: discard prior, restart from [summary, *tail]
            continue
        batch = _parse_messages_line(line)
        if batch is not None:
            history.extend(batch)
    return history


def load_latest(sessions_dir: Path) -> list[ModelMessage] | None:
    latest = _latest_session_file(sessions_dir)
    if latest is None:
        return None
    return load(latest)


def resolve_session(sessions_dir: Path, identifier: str) -> Path | None:
    if not sessions_dir.is_dir():
        return None

    direct = sessions_dir / identifier
    if direct.is_file():
        return direct

    for candidate in _session_files(sessions_dir):
        if candidate.name == identifier or identifier in candidate.stem:
            return candidate
    return None


def _session_files(sessions_dir: Path) -> list[Path]:
    if not sessions_dir.is_dir():
        return []
    return [p for p in sessions_dir.iterdir() if p.is_file() and p.suffix == _SUFFIX]


def _latest_session_file(sessions_dir: Path) -> Path | None:
    files = _session_files(sessions_dir)
    if not files:
        return None
    return max(files, key=lambda p: p.name)


def _parse_messages_line(line: str) -> list[ModelMessage] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj: Any = json.loads(stripped)
    except json.JSONDecodeError:
        logger.debug("skipping unparseable session-log line: %r", line[:80])
        return None
    if not isinstance(obj, dict) or obj.get("type") != _MESSAGES_TYPE:
        return None
    try:
        return ModelMessagesTypeAdapter.validate_python(obj["messages"])
    except Exception:
        logger.debug("skipping malformed messages entry in session log", exc_info=True)
        return None


def _is_clear_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    try:
        obj: Any = json.loads(stripped)
    except json.JSONDecodeError:
        return False
    return isinstance(obj, dict) and obj.get("type") == _CLEAR_TYPE


def _parse_compaction_line(line: str) -> list[ModelMessage] | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        obj: Any = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict) or obj.get("type") != _COMPACTION_TYPE:
        return None
    try:
        summary = ModelMessagesTypeAdapter.validate_python(obj["summary"])
        tail = ModelMessagesTypeAdapter.validate_python(obj["tail"])
    except Exception:
        logger.debug("skipping malformed compaction entry in session log", exc_info=True)
        return None
    return [*summary, *tail]
