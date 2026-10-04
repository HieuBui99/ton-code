import io
import unittest

from rich.console import Console

from src.entities import events
from src.tui.renderer import (
    Notice,
    TranscriptRenderer,
    UserSubmitted,
    clean_text,
    preview,
)


class RendererTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        self.renderer = TranscriptRenderer(
            Console(file=self.output, width=80, color_system=None)
        )

    def test_split_lines_and_fragment_boundary(self):
        self.renderer.render(events.AssistantTextDelta("hel"))
        self.assertEqual(self.output.getvalue(), "")
        self.renderer.render(events.AssistantTextDelta("lo\nlast"))
        self.assertEqual(self.output.getvalue(), "Assistant\nhello\n")
        self.renderer.render(events.ToolCallStarted("id", "bash", "{}"))
        self.assertEqual(
            self.output.getvalue(), "Assistant\nhello\nlast\nTool bash [id]: {}\n"
        )
        self.renderer.render(events.AssistantTextDelta("done"))
        self.renderer.render(events.TurnFinished(1))
        self.assertIn("Assistant\ndone\nTurn complete.", self.output.getvalue())

    def test_literal_text_and_split_terminal_escape(self):
        self.renderer.render(events.AssistantTextDelta("[red]漢字[/red]\x1b["))
        self.renderer.render(events.AssistantTextDelta("2Jsafe\n"))
        self.assertEqual(self.output.getvalue(), "Assistant\n[red]漢字[/red]safe\n")
        self.assertEqual(clean_text("a\x1b]0;title\x07b\r\b\x00"), "ab")

    def test_preview_limits_and_verbose(self):
        result = events.ToolResult(
            "id", "read", "\n".join(f"line {n}" for n in range(20))
        )
        self.renderer.render(result)
        self.assertIn("line 9\n… output omitted", self.output.getvalue())
        self.assertNotIn("line 10", self.output.getvalue())
        self.assertTrue(preview("x" * 2001).startswith("x" * 2000 + "\n…"))
        self.output.truncate(0)
        self.output.seek(0)
        self.renderer.verbose = True
        self.renderer.render(events.ThinkingDelta("reason\n"))
        self.renderer.render(result)
        self.assertIn("Thinking\nreason", self.output.getvalue())
        self.assertIn("line 19", self.output.getvalue())

    def test_hidden_thinking_and_event_order(self):
        self.renderer.render(UserSubmitted("hello"))
        self.renderer.render(events.TurnStarted(1, "hello"))
        self.renderer.render(events.ThinkingDelta("private\n"))
        self.renderer.render(events.AssistantTextDelta("answer"))
        self.renderer.render(Notice("notice"))
        self.renderer.render(events.AgentError("bad"))
        self.assertEqual(
            self.output.getvalue(),
            "You\nhello\nAssistant\nanswer\nnotice\nError: bad\n",
        )

    def test_parallel_tool_results_are_correlated(self):
        self.renderer.render(events.ToolCallStarted("a", "read", "{}"))
        self.renderer.render(events.ToolCallStarted("b", "bash", "{}"))
        self.renderer.render(events.ToolResult("b", "", "failed", ok=False))
        self.renderer.render(events.ToolResult("a", "", "ok"))
        self.assertIn("FAILED bash [b]", self.output.getvalue())
        self.assertIn("OK read [a]", self.output.getvalue())

    def test_context_status_does_not_split_a_streamed_line(self):
        self.renderer.render(events.AssistantTextDelta("hel"))
        self.renderer.render(events.ContextUsage(1234))
        self.assertEqual(self.output.getvalue(), "")
        self.renderer.render(events.AssistantTextDelta("lo\n"))
        self.assertEqual(self.output.getvalue(), "Assistant\nhello\n")
