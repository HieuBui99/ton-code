import tempfile
import unittest
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import DeltaToolCall, FunctionModel

from src.agent.deps import AgentDeps
from src.agent.loop import AgentTurnHandler
from src.context.session_log import SessionLog, load
from src.entities import events
from src.tools.registry import register_tools
from src.tui.runner import SessionRunner


class AgentIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_real_loop_streams_tool_results_and_persists_turns(self):
        calls = 0

        async def model_stream(messages, info):
            nonlocal calls
            calls += 1
            if calls == 1:
                yield {
                    0: DeltaToolCall(
                        name="read",
                        json_args='{"path":"sample.txt"}',
                        tool_call_id="read-1",
                    )
                }
            else:
                yield "hello "
                yield "world\n"

        with tempfile.TemporaryDirectory() as directory:
            cwd = Path(directory)
            (cwd / "sample.txt").write_text("fixture data\n")
            emitted = []
            deps = AgentDeps(cwd=cwd, emit=emitted.append)
            agent = Agent(
                FunctionModel(stream_function=model_stream), deps_type=AgentDeps
            )
            register_tools(agent)
            session_log = SessionLog.create(cwd / "sessions", cwd=cwd)
            handler = AgentTurnHandler(agent, deps=deps, session_log=session_log)
            runner = SessionRunner(handler, emitted.append)
            runner.submit("read sample.txt")
            await runner.wait_idle()
            self.assertFalse(
                any(isinstance(event, events.AgentError) for event in emitted), emitted
            )
            self.assertEqual(
                sum(isinstance(event, events.ToolCallStarted) for event in emitted), 1
            )
            results = [
                event for event in emitted if isinstance(event, events.ToolResult)
            ]
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].ok)
            self.assertIn("fixture data", results[0].output)
            self.assertEqual(
                "".join(
                    event.text
                    for event in emitted
                    if isinstance(event, events.AssistantTextDelta)
                ),
                "hello world\n",
            )
            persisted = load(session_log.path)
            self.assertTrue(
                any(
                    isinstance(part, ToolCallPart)
                    for msg in persisted
                    for part in msg.parts
                )
            )
            self.assertTrue(
                any(
                    isinstance(part, ToolReturnPart)
                    for msg in persisted
                    for part in msg.parts
                )
            )
            runner.submit("next turn")
            await runner.wait_idle()
            self.assertGreater(len(load(session_log.path)), len(persisted))
            context_updates = [
                event for event in emitted if isinstance(event, events.ContextUsage)
            ]
            self.assertGreater(context_updates[-1].input_tokens, 0)
            self.assertEqual(
                context_updates[-1].input_tokens, handler.last_input_tokens
            )
            self.assertFalse(context_updates[-1].estimated)
            handler.clear()
            self.assertEqual(emitted[-1], events.ContextUsage(0))
            await runner.close()
