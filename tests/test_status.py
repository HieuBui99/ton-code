import io
import unittest
from pathlib import Path

from prompt_toolkit.utils import get_cwidth
from rich.console import Console

from src.tui.branding import print_banner
from src.tui.status import ContextStatus


class StatusTests(unittest.TestCase):
    def test_context_bar_unknown_exact_estimated_and_over_capacity(self):
        context = ContextStatus(100_000)
        self.assertIn("[----------] -- / 100,000 · --", context.label())
        context.tokens = 50_000
        self.assertIn("[#####-----] 50,000 / 100,000 · 50%", context.label())
        context.tokens = 80_000
        self.assertEqual(context.style, "class:context.warning")
        context.tokens = 120_000
        self.assertIn("[##########] 120,000 / 100,000 · 120%", context.label())
        self.assertEqual(context.style, "class:context.danger")
        context.estimated = True
        context.window_assumed = True
        self.assertIn("~120,000", context.label())
        self.assertIn("window assumed", context.label())
        self.assertEqual(context.label(compact=True), "ctx ~120%*")

    def test_narrow_terminal_and_unicode_model_names(self):
        context = ContextStatus(200_000, tokens=100_000)
        for width in (40, 60, 80):
            fragments = context.toolbar("漢字-model-" * 10, width)
            label = "".join(text for _, text in fragments)
            self.assertLessEqual(get_cwidth(label), width)
            self.assertIn("50%", label)

    def test_banner_embeds_logo_and_model_without_asset_files(self):
        output = io.StringIO()
        console = Console(file=output, width=90, color_system=None)
        print_banner(
            console,
            model="sample-model",
            cwd=Path("/workspace"),
            window=200_000,
            window_assumed=True,
        )
        self.assertIn("▄███", output.getvalue())
        self.assertIn("sample-model", output.getvalue())
        self.assertIn("200,000 tokens (assumed)", output.getvalue())
        output.truncate(0)
        output.seek(0)
        console = Console(file=output, width=40, color_system=None)
        print_banner(
            console,
            model="sample-model",
            cwd=Path("/workspace"),
            window=200_000,
            window_assumed=False,
        )
        self.assertIn("TonCode", output.getvalue())
        self.assertNotIn("▄███", output.getvalue())
