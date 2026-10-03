from __future__ import annotations

import logging
import os
import re
import tempfile
from collections.abc import Awaitable, Callable
from pathlib import Path, PurePosixPath

import anyio
from pydantic_ai import ModelRetry, RunContext

from src.agent.deps import AgentDeps
from src.config.settings import settings
from src.tools.truncate import truncate

logger = logging.getLogger(__name__)

READ_TOOL_NAME = "read"
GLOB_TOOL_NAME = "glob"
GREP_TOOL_NAME = "grep"
WRITE_TOOL_NAME = "write"
EDIT_TOOL_NAME = "edit"

FILE_TOOLS_MUTATING: dict[str, bool] = {
    WRITE_TOOL_NAME: False,
    EDIT_TOOL_NAME: False,
}

_UTF8_BOM = "﻿"


def _is_within(base: Path, candidate: Path) -> bool:
    """True iff ``candidate`` is ``base`` itself or lives under it (both already resolved)."""
    return candidate == base or base in candidate.parents


def _resolve_in_cwd(cwd: Path, raw: str) -> Path:
    """Resolve ``raw`` under ``cwd``; raise :class:`pydantic_ai.ModelRetry` when it escapes."""
    base = cwd.resolve()
    target = (base / raw).resolve()
    if not _is_within(base, target):
        raise ModelRetry(
            f"Path {raw!r} resolves outside the working directory; stay within the project tree."
        )
    return target


def _reject_escaping_pattern(pattern: str) -> None:
    """Reject a glob ``pattern`` that escapes ``cwd`` (absolute or ``..``) before globbing.

    Also avoids ``Path.glob``'s ``NotImplementedError`` on absolute patterns; :func:`_contain`
    is the post-glob second line of defence (in-tree symlinks resolving outside ``cwd``).
    """
    if pattern.startswith("/") or ".." in Path(pattern).parts:
        raise ModelRetry(
            f"Glob pattern {pattern!r} points outside the working directory; "
            "use a pattern relative to the project tree (no '..' or absolute paths)."
        )


def _contain(base: Path, matches: list[Path]) -> list[Path]:
    """Keep only ``matches`` whose *resolved* real path stays under ``base`` (files only).

    Resolving first drops an in-tree symlink pointing outside ``cwd`` — its contents must
    never reach the model.
    """
    return sorted(m for m in matches if m.is_file() and _is_within(base, m.resolve()))


def _bridge[T](op: Callable[..., Awaitable[T]], *args: object) -> T:
    """Run an async backend file op from a sync tool, rendering an infra failure as a retry (§4).

    Bridges via :func:`anyio.from_thread.run`; an infra failure below the seam becomes a
    model-readable :class:`pydantic_ai.ModelRetry` while the op's own ``ModelRetry``\\ s pass
    straight through. A :class:`WorkspaceEscape` from a real-fs backend's physical containment is
    an :class:`OSError`, so it is caught by base class — files.py never imports it, keeping the
    ``none`` path free of any sandbox import (ADR-0012 §9).
    """
    try:
        return anyio.from_thread.run(op, *args)
    except ModelRetry:
        raise
    except (RuntimeError, OSError) as exc:
        logger.debug("sandbox file op failed: %s", exc)
        raise ModelRetry(f"Sandbox file operation failed: {exc}") from exc


def _resolve_logical(raw: str) -> str:
    """Resolve ``raw`` to a Workspace-relative POSIX path, rejecting escapes (ADR-0012 §4).

    Backend-agnostic path math — **never** host ``Path.resolve`` (a modal Workspace path is not a
    host path): fold ``.`` / ``..`` against the logical Workspace root and reject anything that
    escapes it (a ``..`` climbing above the root, or an absolute path) with the same
    :class:`pydantic_ai.ModelRetry` refusal :func:`_resolve_in_cwd` gives in ``none`` mode.
    Shared by both backends; returns the root-relative path (``""`` for the root).
    """
    pure = PurePosixPath(raw)
    escape = ModelRetry(
        f"Path {raw!r} resolves outside the working directory; stay within the project tree."
    )
    if pure.is_absolute():
        raise escape
    parts: list[str] = []
    for part in pure.parts:
        if part == ".":
            continue
        if part == "..":
            if not parts:
                raise escape
            parts.pop()
        else:
            parts.append(part)
    return "/".join(parts)


def read(
    ctx: RunContext[AgentDeps],
    path: str,
    offset: int | None = None,
    limit: int | None = None,
) -> str:
    """
    Read a text file with 1-indexed, numbered lines
    """
    target = _resolve_in_cwd(ctx.deps.cwd, path)
    if not target.exists():
        raise ModelRetry(f"No such file: {path!r}.")
    if target.is_dir():
        raise ModelRetry(f"{path!r} is a directory; use glob to list its contents.")
    try:
        content = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelRetry(f"Could not read {path!r}: {exc}.") from exc

    return _render_numbered(content, path, offset=offset, limit=limit)


def _render_numbered(content: str, path: str, *, offset: int | None, limit: int | None) -> str:
    """Number ``content``'s lines into the model-facing read result — shared by both modes (§4).

    The rendering tail both the ``none`` and sandbox paths call, so a sandbox read reads
    byte-identically to a host read: 1-indexed ``cat -n`` numbering for the ``offset`` / ``limit``
    window, then the safety cap through :mod:`decode.tools.truncate` with the overflow spill note.
    An ``offset`` past end-of-file is a model-readable :class:`pydantic_ai.ModelRetry`.
    """
    start = 1 if offset is None else max(offset, 1)
    # `limit` is the caller's window; the truncate safety cap below bounds runaway output.
    line_limit = None if limit is None else max(limit, 0)
    numbered = _number_lines(content, start=start, limit=line_limit)
    if numbered is None:
        raise ModelRetry(f"{path!r} has no line {start}: offset is past the end of the file.")

    result = truncate(
        numbered, max_lines=settings.max_output_lines, max_bytes=settings.max_output_bytes
    )
    text = result.text.rstrip("\n")
    if result.truncated:
        text += (
            f"\n\n[output truncated to {settings.max_output_lines} lines / "
            f"{settings.max_output_bytes} bytes; full content at {result.full_path}]"
        )
    return text


def _number_lines(content: str, *, start: int, limit: int | None) -> str | None:
    """Render ``content`` as ``<lineno>\\t<line>`` for ``[start, start+limit)`` (``None`` = to end).

    Line numbers are absolute (1-indexed) so a windowed read re-pages correctly. Returns
    ``None`` when ``start`` is past the last line (the caller turns that into a ``ModelRetry``).
    """
    lines = content.splitlines()
    if start > len(lines):
        return None
    window = lines[start - 1 :] if limit is None else lines[start - 1 : start - 1 + limit]
    return "\n".join(f"{start + i}\t{line}" for i, line in enumerate(window))


def glob(ctx: RunContext[AgentDeps], pattern: str) -> str:
    """
    List paths matching shell glob ``pattern`` under ``cwd``, paths only
    """
    _reject_escaping_pattern(pattern)
    base = ctx.deps.cwd.resolve()
    matches = _contain(base, list(base.glob(pattern)))
    if not matches:
        raise ModelRetry(f"No files match {pattern!r} under the working directory.")
    return "\n".join(str(p.relative_to(base)) for p in matches)


def grep(
    ctx: RunContext[AgentDeps],
    pattern: str,
    path: str | None = None,
    glob: str | None = None,
) -> str:
    """
    Regex-search file contents under ``cwd``, returning ``path:lineno:line`` hits
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise ModelRetry(f"Invalid regular expression {pattern!r}: {exc}.") from exc

    base = ctx.deps.cwd.resolve()
    candidates = _grep_candidates(base, path=path, glob=glob)

    hits: list[str] = []
    for file in candidates:
        try:
            text = file.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            # Skip binary / unreadable files silently — a search should not crash on one.
            continue
        rel = file.relative_to(base)
        for lineno, line in enumerate(text.splitlines(), start=1):
            if regex.search(line):
                hits.append(f"{rel}:{lineno}:{line}")

    if not hits:
        raise ModelRetry(f"No matches for {pattern!r} in the searched files.")
    return _render_grep_hits(hits)


def _render_grep_hits(hits: list[str]) -> str:
    """Render ``path:lineno:line`` hits into the model-facing grep result — shared by both modes (§4).

    Hits joined and passed through the safety cap (:mod:`decode.tools.truncate`) with the
    overflow spill note, so sandbox and host greps read identically.
    """
    result = truncate(
        "\n".join(hits) + "\n",
        max_lines=settings.max_output_lines,
        max_bytes=settings.max_output_bytes,
    )
    text = result.text.rstrip("\n")
    if result.truncated:
        text += f"\n\n[matches truncated; full results at {result.full_path}]"
    return text


def _grep_candidates(base: Path, *, path: str | None, glob: str | None) -> list[Path]:
    """The list of files grep will search, resolved under ``base`` (sorted, files only).

    ``path`` wins over ``glob``; with neither, the whole tree (``**/*``). Both routes stay
    contained under ``base`` (escaping patterns rejected up front, out-of-tree symlinks dropped),
    so grep never reads an out-of-tree file. Raises :class:`pydantic_ai.ModelRetry` for a
    missing explicit ``path`` or an escaping ``glob`` pattern.
    """
    if path is not None:
        target = _resolve_in_cwd(base, path)
        if not target.is_file():
            raise ModelRetry(f"No such file to search: {path!r}.")
        return [target]
    pattern = glob or "**/*"
    _reject_escaping_pattern(pattern)
    return _contain(base, list(base.glob(pattern)))


def write(ctx: RunContext[AgentDeps], path: str, content: str) -> str:
    """
    Create or overwrite a file with ``content``
    """
    target = _resolve_in_cwd(ctx.deps.cwd, path)
    if target.is_dir():
        raise ModelRetry(f"{path!r} is a directory; choose a file path to write.")
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_bytes(target, content.encode("utf-8"))
    logger.debug("wrote %d bytes to %r", len(content), path)
    return f"Wrote {path!r} ({len(content)} characters)."



def edit(ctx: RunContext[AgentDeps], path: str, old_string: str, new_string: str) -> str:
    """
    Replace a **unique** occurrence of ``old_string`` with ``new_string``
    """
    if old_string == "":
        raise ModelRetry("old_string is empty; provide the exact text to replace.")

    target = _resolve_in_cwd(ctx.deps.cwd, path)
    if not target.is_file():
        raise ModelRetry(f"No such file to edit: {path!r}.")
    try:
        # Raw bytes (not read_text): universal newlines would destroy the CR/CRLF style to restore.
        raw = target.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ModelRetry(f"Could not read {path!r} to edit: {exc}.") from exc

    final = _apply_edit(raw, old_string, new_string)
    _atomic_write_bytes(target, final.encode("utf-8"))
    logger.debug("edited %r", path)
    return f"Edited {path!r} (replaced 1 occurrence)."
    


def _apply_edit(raw_text: str, old_string: str, new_string: str) -> str:
    """Apply edit's unique-match replacement to ``raw_text`` — shared by both modes"""
    had_bom = raw_text.startswith(_UTF8_BOM)
    body = raw_text[len(_UTF8_BOM) :] if had_bom else raw_text
    eol = _detect_eol(body)
    normalized = _to_lf(body)
    needle = _to_lf(old_string)
    new_normalized = _replace_unique(normalized, needle, _to_lf(new_string))
    restored = new_normalized.replace("\n", eol) if eol != "\n" else new_normalized
    return (_UTF8_BOM + restored) if had_bom else restored


def _detect_eol(text: str) -> str:
    """Return the dominant line-ending style of ``text``: CRLF checked first, then lone CR, else LF."""
    if "\r\n" in text:
        return "\r\n"
    if "\r" in text:
        return "\r"
    return "\n"


def _to_lf(text: str) -> str:
    """Normalize CRLF and lone-CR line endings to LF (so matching is newline-agnostic)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _replace_unique(haystack: str, needle: str, replacement: str) -> str:
    """Replace the single occurrence of ``needle`` in ``haystack`` (exact, then fuzzy).

    Exact match first (``0`` → not found, ``>1`` → ambiguous); only on no exact match fall back
    to the whitespace-normalized fuzzy search (:func:`_fuzzy_unique_span`), which must still
    resolve to a single span. Raises a model-readable :class:`pydantic_ai.ModelRetry` on 0 / >1.
    """
    exact = haystack.count(needle)
    if exact == 1:
        return haystack.replace(needle, replacement, 1)
    if exact > 1:
        raise ModelRetry(
            f"old_string is ambiguous, {exact} matches found; add surrounding context to "
            "old_string so it identifies exactly one location."
        )

    span = _fuzzy_unique_span(haystack, needle)
    if span is None:
        raise ModelRetry(
            "old_string not found in the file; it must match the file's current text "
            "(check for typos, indentation, or stale content)."
        )
    start, end = span
    return haystack[:start] + replacement + haystack[end:]


def _normalize_ws(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip the ends — the fuzzy-match key."""
    return " ".join(text.split())


def _fuzzy_unique_span(haystack: str, needle: str) -> tuple[int, int] | None:
    """Find the unique ``[start, end)`` span in ``haystack`` matching ``needle`` after whitespace collapse.

    Scans spans that begin/end on non-whitespace and keeps those whose :func:`_normalize_ws`
    form equals the normalized ``needle``. Returns the single match, ``None`` for zero, and
    raises :class:`pydantic_ai.ModelRetry` (``ambiguous``) for more than one *distinct* span.
    Cold path (exact match missed, bounded files, human-gated), so the all-pairs scan is fine.
    """
    target = _normalize_ws(needle)
    if target == "":
        return None

    starts = [i for i, ch in enumerate(haystack) if not ch.isspace()]
    ends = [i + 1 for i, ch in enumerate(haystack) if not ch.isspace()]
    matches: list[tuple[int, int]] = []
    for start in starts:
        for end in ends:
            if end <= start:
                continue
            if _normalize_ws(haystack[start:end]) == target:
                matches.append((start, end))

    # Keep the shortest end per start: longer ends differ only by whitespace already absorbed.
    minimal = _minimal_spans(matches)
    if not minimal:
        return None
    if len(minimal) > 1:
        raise ModelRetry(
            f"old_string is ambiguous, {len(minimal)} matches found (after normalizing "
            "whitespace); add surrounding context so it identifies exactly one location."
        )
    return minimal[0]


def _minimal_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reduce ``spans`` to the distinct minimal-length match per start (distinct starts → ambiguous upstream)."""
    shortest_by_start: dict[int, tuple[int, int]] = {}
    for start, end in spans:
        current = shortest_by_start.get(start)
        if current is None or end < current[1]:
            shortest_by_start[start] = (start, end)
    return sorted(shortest_by_start.values())


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    """Write ``data`` to ``target`` atomically-ish (sibling temp file + ``os.replace``).

    A reader never sees a half-written file and a crash mid-write leaves the original intact;
    the temp file shares ``target``'s directory so the rename stays on one filesystem
    (``os.replace`` is atomic only within a filesystem).
    """
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".decode-write-")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
