from __future__ import annotations

import logging
from pathlib import Path

from pydantic_ai import ModelRetry, RunContext

from src.agent.deps import AgentDeps
from src.config.settings import settings
from src.tools.exec import CommandExecutor, ExecResult, LocalExecutor
from src.tools.truncate import Truncated, truncate

logger = logging.getLogger(__name__)

BASH_TOOL_NAME = "bash"

_EXECUTOR: CommandExecutor = LocalExecutor()
_executor_selected = False

def _get_executor() -> CommandExecutor:
    return _EXECUTOR


def active_executor() -> CommandExecutor:
    return _get_executor()


async def warm_executor(workspace: Path) -> None:
    if settings.sandbox_mode == "none":
        return
    executor = _get_executor()
    start = getattr(executor, "start", None)
    if start is not None:
        await start(workspace)


def reset_executor() -> None:
    global _EXECUTOR, _executor_selected
    _EXECUTOR = LocalExecutor()
    _executor_selected = False


async def close_executor() -> None:
    global _EXECUTOR, _executor_selected
    executor = _EXECUTOR
    _EXECUTOR = LocalExecutor()
    _executor_selected = False
    aclose = getattr(executor, "aclose", None)
    if aclose is not None:
        await aclose()
        return
    close = getattr(executor, "close", None)
    if close is not None:
        close()


async def export_executor() -> None:
    executor = _EXECUTOR
    export = getattr(executor, "export", None)
    if export is not None:
        await export()


async def bash(
    ctx: RunContext[AgentDeps],
    command: str,
    timeout: float | None = None,
) -> str:
    """
    Run a shell ``command`` in the working directory and report its result
    """
    if not command.strip():
        raise ModelRetry("command is empty; provide a shell command to run.")
    timeout_s = _resolve_timeout(timeout)

    result = await _get_executor().run(command, cwd=ctx.deps.cwd, timeout_s=timeout_s)
    logger.debug(
        "bash ran (exit=%d, timed_out=%s, command=%r)",
        result.exit_code,
        result.timed_out,
        command,
    )
    return _render(result, timeout_s=timeout_s)


def _resolve_timeout(timeout: float | None) -> float:
    """Resolve the effective timeout: default from settings, clamped to the configured max.

    A model-supplied value is clamped to ``settings.bash_timeout_s`` (the model cannot extend
    its own leash) and must be positive (else a model-correctable :class:`ModelRetry`).
    """
    if timeout is None:
        return settings.bash_timeout_s
    if timeout <= 0:
        raise ModelRetry("timeout must be a positive number of seconds.")
    return min(timeout, settings.bash_timeout_s)


def _render(result: ExecResult, *, timeout_s: float) -> str:
    """
    Render an :class:`ExecResult` into the model-facing reply (status + truncated streams)
    """
    if result.timed_out:
        header = (
            f"Command timed out after {timeout_s:g}s and was terminated "
            f"(exit code {result.exit_code}). Partial output below."
        )
    else:
        header = f"Exit code: {result.exit_code}."

    sections = [header]
    stdout_section = _stream_section("stdout", result.stdout)
    if stdout_section:
        sections.append(stdout_section)
    stderr_section = _stream_section("stderr", result.stderr)
    if stderr_section:
        sections.append(stderr_section)
    if result.note:
        sections.append(result.note)
    return "\n\n".join(sections)


def _stream_section(label: str, content: str) -> str | None:
    """Format one captured stream as a labelled, truncated section with a spill notice (``None`` if empty)."""
    if content == "":
        return None
    capped: Truncated = truncate(
        content, max_lines=settings.max_output_lines, max_bytes=settings.max_output_bytes
    )
    body = capped.text.rstrip("\n")
    if capped.truncated:
        body += (
            f"\n\n[{label} truncated to {settings.max_output_lines} lines / "
            f"{settings.max_output_bytes} bytes; full content at {capped.full_path}]"
        )
    return f"{label}:\n{body}"
