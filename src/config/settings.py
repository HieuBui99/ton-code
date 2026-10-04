from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


MODEL_CONTEXT_WINDOWS: tuple[tuple[str, int], ...] = (
    ("gemini-3.5", 1_048_576),
    ("gemini-2.5", 1_048_576),
    ("GLM-5.3-Flash", 1_048_576),
)

UNKNOWN_MODEL_CONTEXT_WINDOW = 200_000


def env_files(env_file: Path | None = None) -> tuple[Path, ...]:
    """An explicit file replaces automatic discovery; later files override earlier ones."""
    if env_file is not None:
        return (env_file.expanduser().resolve(),)
    return (Path.home() / ".config/ton-code/.env", Path.cwd() / ".env")


def context_window_for(model_id: str) -> int | None:
    """The known input window for ``model_id``, or ``None`` when no table row matches.

    ``None`` (rather than the fallback) is the signal the caller needs to warn — the fallback is a
    guess, and a guess the operator never sees is how a 4x-wrong window survives for months.
    """
    lowered = model_id.lower()
    for pattern, window in MODEL_CONTEXT_WINDOWS:
        if pattern.lower() in lowered:
            return window
    return None


class Settings(BaseSettings):
    """Runtime configuration. Defaults are safe for tests, not production."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    llm_provider: Literal["openai", "gemini"] = "openai"

    model_endpoint_url: str = ""
    model_endpoint_model: str = "GLM-5.3-Flash"
    model_endpoint_api_key: SecretStr = SecretStr("")

    # gemini (default): google-genai API-key path.
    gemini_api_key: SecretStr = SecretStr("")
    gemini_model: str = "gemini-3.5-flash"

    # --- Logging ---
    log_level: str = "INFO"

    # --- Tool execution / output truncation ---
    bash_timeout_s: float = 120.0
    max_output_lines: int = 2000
    max_output_bytes: int = 50_000
    web_fetch_timeout_s: float = 30.0

    # ``sleep(seconds)`` is capped to this value (never rejected) so a model cannot stall a turn.
    sleep_max_s: float = 60.0

    # --- Memory caps ---
    memory_max_lines: int = 200
    memory_max_bytes: int = 25_000

    # ``compaction_enabled`` gates ONLY the automatic cascade; manual ``/compact`` ignores it.
    compaction_enabled: bool = True

    compaction_context_window_tokens: int = Field(1_048_576, gt=0)
    # A tier fires when input_tokens >= window * (1 - reserve). INVARIANT: micro reserves more than
    # full so it fires first — ``microcompaction_reserve_fraction > compaction_reserve_fraction``.
    compaction_reserve_fraction: float = Field(0.20, ge=0.0, le=1.0)
    microcompaction_reserve_fraction: float = Field(0.40, ge=0.0, le=1.0)
    # Token budget of the recent tail full compaction keeps verbatim (microcompaction's "recent" cutoff).
    compaction_keep_recent_tokens: int = 20_000
    # When set, the on-exit MEMORY.md LLM compressor runs at the ``memory_max_lines`` cap instead of
    # pure drop-oldest.
    memory_compression_enabled: bool = True

    # --- Harness artifacts: everything decode writes lives under <cwd>/.decode. ---
    decode_dir: Path = Path(".agents")

    # --- Persistence: JSONL session log ---
    sessions_dir: Path = Path(".agents/sessions")

    # --- Permissions: optional {"permissions": {"allow": [...], "deny": [...]}} rules file. ---
    # Missing/malformed is non-fatal (the gate falls back to mode-only).
    permissions_file: Path = Path(".agents/settings.json")

    skills_dir: Path = Path(".agents/skills")

    def active_model(self) -> str:
        """The model id for the active provider, or the override if set."""
        provider = self.llm_provider
        if provider == "gemini":
            return self.gemini_model
        if provider == "openai":
            return self.model_endpoint_model
        raise ValueError(f"unknown llm_provider {provider!r}")

    @property
    def context_window_is_assumed(self) -> bool:
        return (
            "compaction_context_window_tokens" not in self.model_fields_set
            and context_window_for(self.active_model()) is None
        )

    @model_validator(mode="after")
    def _derive_compaction_context_window(self) -> Settings:
        if "compaction_context_window_tokens" in self.model_fields_set:
            return self
        known = context_window_for(self.active_model())
        window = known if known is not None else UNKNOWN_MODEL_CONTEXT_WINDOW
        if known is None:
            logger.debug(
                "no known context window for model %r; assuming %d tokens",
                self.active_model(),
                window,
            )
        object.__setattr__(self, "compaction_context_window_tokens", window)
        return self


# CLI loads dotenv configuration before constructing any agents. Keep this shared
# object so modules importing settings see the same validated runtime configuration.
settings = Settings(_env_file=None)


def load_settings(env_file: Path | None = None) -> Settings:
    configured = Settings(_env_file=env_files(env_file))
    for name in Settings.model_fields:
        setattr(settings, name, getattr(configured, name))
    settings.__pydantic_fields_set__ = configured.model_fields_set.copy()
    return settings
