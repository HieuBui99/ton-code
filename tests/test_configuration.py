import asyncio
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rich.console import Console

from src.cli import _run
from src.config.settings import Settings, env_files, load_settings, settings


class ConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.previous = settings.model_copy(deep=True)
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.home = self.base / "home"
        self.workspace = self.base / "workspace"
        self.workspace.mkdir()
        self.global_file = self.home / ".config/ton-code/.env"
        self.global_file.parent.mkdir(parents=True)
        self.global_file.write_text(
            "LLM_PROVIDER=openai\nMODEL_ENDPOINT_URL=http://localhost:8000\nMODEL_ENDPOINT_MODEL=global-model\n"
        )
        self.local_file = self.workspace / ".env"
        self.local_file.write_text("MODEL_ENDPOINT_MODEL=local-model\n")
        self.addCleanup(self.restore_settings)

    def restore_settings(self):
        for name in Settings.model_fields:
            setattr(settings, name, getattr(self.previous, name))
        settings.__pydantic_fields_set__ = self.previous.model_fields_set.copy()

    def test_global_local_and_environment_precedence(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("pathlib.Path.home", return_value=self.home),
            patch("pathlib.Path.cwd", return_value=self.workspace),
        ):
            configured = load_settings()
            self.assertIs(configured, settings)
            self.assertEqual(configured.active_model(), "local-model")
            self.assertEqual(configured.model_endpoint_url, "http://localhost:8000")
            self.local_file.unlink()
            self.assertEqual(load_settings().active_model(), "global-model")
            os.environ["MODEL_ENDPOINT_MODEL"] = "environment-model"
            self.assertEqual(load_settings().active_model(), "environment-model")

    def test_explicit_file_excludes_automatic_files(self):
        explicit = self.base / "custom.env"
        explicit.write_text("LLM_PROVIDER=gemini\nGEMINI_MODEL=explicit-model\n")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("pathlib.Path.home", return_value=self.home),
            patch("pathlib.Path.cwd", return_value=self.workspace),
        ):
            self.assertEqual(env_files(explicit), (explicit,))
            configured = load_settings(explicit)
            self.assertEqual(configured.active_model(), "explicit-model")
            self.assertEqual(configured.model_endpoint_url, "")

    def test_launch_uses_external_configuration_and_current_workspace(self):
        from src.tui.app import TerminalApp

        explicit = self.base / "custom.env"
        explicit.write_text(
            "LLM_PROVIDER=openai\nMODEL_ENDPOINT_URL=http://localhost:8000\nMODEL_ENDPOINT_MODEL=external-model\nCOMPACTION_CONTEXT_WINDOW_TOKENS=64000\n"
        )
        recorded = {}

        async def fake_run(app, handler):
            recorded["app"] = app
            recorded["handler"] = handler

        with (
            patch.dict(os.environ, {}, clear=True),
            patch("pathlib.Path.cwd", return_value=self.workspace),
            patch.object(TerminalApp, "run", fake_run),
        ):
            asyncio.run(
                _run(None, False, Console(file=io.StringIO()), env_file=explicit)
            )
        app = recorded["app"]
        handler = recorded["handler"]
        self.assertEqual(app.model, "external-model")
        self.assertEqual(app.context.window, 64_000)
        self.assertFalse(app.context.window_assumed)
        self.assertEqual(app.cwd, self.workspace)
        self.assertEqual(handler._deps.cwd, self.workspace)
        self.assertTrue(handler._session_log.path.is_relative_to(self.workspace))
