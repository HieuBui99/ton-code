import asyncio
import io
import subprocess
import sys
import unittest
from unittest.mock import patch

from click.testing import CliRunner
from rich.console import Console

from src.cli import _run, main
from src.config.settings import Settings


class CliTests(unittest.TestCase):
    def test_help_works_without_valid_settings(self):
        result = subprocess.run(
            [sys.executable, "-m", "src", "--help"],
            capture_output=True,
            text=True,
            env={"LLM_PROVIDER": "invalid", "PATH": "/usr/bin:/bin"},
            check=False,
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--verbose", result.stdout)

    def test_redirected_invocation_fails_cleanly(self):
        result = CliRunner().invoke(main)
        self.assertEqual(result.exit_code, 1)
        self.assertIn("requires terminal input and output", result.output)
        self.assertNotIn("Traceback", result.output)

    def test_provider_configuration_validation(self):
        console = Console(file=io.StringIO())
        for configuration, message in [
            (
                {"llm_provider": "openai", "model_endpoint_url": ""},
                "MODEL_ENDPOINT_URL",
            ),
            ({"llm_provider": "gemini", "gemini_api_key": ""}, "GEMINI_API_KEY"),
        ]:
            settings = Settings(_env_file=None, **configuration)
            with (
                patch("src.config.settings.load_settings", return_value=settings),
                self.assertRaisesRegex(ValueError, message),
            ):
                asyncio.run(_run(None, False, console))

    def test_known_and_unknown_context_windows(self):
        known = Settings(_env_file=None, model_endpoint_model="GLM-5.3-Flash")
        self.assertEqual(known.compaction_context_window_tokens, 1_048_576)
        self.assertFalse(known.context_window_is_assumed)
        unknown = Settings(_env_file=None, model_endpoint_model="unknown")
        self.assertEqual(unknown.compaction_context_window_tokens, 200_000)
        self.assertTrue(unknown.context_window_is_assumed)
