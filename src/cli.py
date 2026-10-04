from __future__ import annotations

import asyncio
import os
import sys
import traceback
from pathlib import Path

import click
from rich.console import Console
from rich.text import Text

from src.tui.renderer import clean_text


async def _run(
    model: str | None, verbose: bool, console: Console, *, env_file: Path | None = None
) -> None:
    # TonCode owns terminal output; suppress Pydantic AI's direct stderr banner.
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    # Lazy imports keep --help usable even with invalid runtime configuration.
    from src.config.settings import context_window_for, load_settings

    settings = load_settings(env_file)

    if settings.llm_provider == "openai":
        from urllib.parse import urlsplit

        endpoint = urlsplit(settings.model_endpoint_url)
        if endpoint.scheme not in ("http", "https") or not endpoint.hostname:
            raise ValueError(
                "Set MODEL_ENDPOINT_URL to an http(s) OpenAI-compatible server URL in ~/.config/ton-code/.env, a local .env, or --env-file."
            )
    elif not settings.gemini_api_key.get_secret_value():
        raise ValueError(
            "Set GEMINI_API_KEY in your .env configuration when LLM_PROVIDER=gemini."
        )

    model_name = model or settings.active_model()
    if not model_name.strip():
        raise ValueError("The model name must not be empty.")
    cwd = Path.cwd().resolve()
    window = settings.compaction_context_window_tokens
    window_assumed = settings.context_window_is_assumed
    if model and "compaction_context_window_tokens" not in settings.model_fields_set:
        from src.config.settings import UNKNOWN_MODEL_CONTEXT_WINDOW

        window = context_window_for(model) or UNKNOWN_MODEL_CONTEXT_WINDOW
        window_assumed = context_window_for(model) is None

    from src.agent.deps import AgentDeps, VerboseFlag
    from src.agent.factory import build_agent
    from src.agent.loop import AgentTurnHandler
    from src.context.session_log import SessionLog
    from src.tui.app import TerminalApp

    app = TerminalApp(
        console,
        model=model_name,
        cwd=cwd,
        verbose=verbose,
        context_window=window,
        window_assumed=window_assumed,
    )
    agent = build_agent(model=model)
    sessions_dir = (
        settings.sessions_dir
        if settings.sessions_dir.is_absolute()
        else cwd / settings.sessions_dir
    )
    log = SessionLog.create(sessions_dir, cwd=cwd)
    deps = AgentDeps(
        cwd=cwd,
        emit=app.emit,
        verbose=VerboseFlag(verbose),
        context_window_tokens=window,
    )
    handler = AgentTurnHandler(
        agent,
        deps=deps,
        session_log=log,
        session_id=log.session_id,
        compaction_model=agent.model,
    )
    await app.run(handler)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--model", help="Override the configured provider's model.")
@click.option(
    "--env-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="Load this .env file instead of global and local configuration.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show thinking, full emitted tool output, and tracebacks.",
)
@click.version_option("0.1.0", prog_name="TonCode")
def main(model: str | None, env_file: Path | None, verbose: bool) -> None:
    """Run TonCode in the current directory with an append-style terminal transcript."""
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise click.ClickException(
            "Interactive mode requires terminal input and output."
        )
    try:
        asyncio.run(_run(model, verbose, Console(), env_file=env_file))
    except KeyboardInterrupt:
        raise click.exceptions.Exit(130) from None
    except Exception as exc:  # noqa: BLE001 - CLI error boundary
        message = traceback.format_exc() if verbose else str(exc)
        Console(stderr=True).print(Text(f"Error: {clean_text(message)}", style="red"))
        raise click.exceptions.Exit(1) from None


if __name__ == "__main__":
    main()
