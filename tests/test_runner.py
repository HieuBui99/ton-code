import asyncio
import unittest

from src.agent.loop import Boundary
from src.context.compaction import CompactOutcome
from src.entities import events
from src.tui.runner import SessionRunner


class FakeHandler:
    def __init__(self):
        self.model_ready = asyncio.Event()
        self.stop_ready = asyncio.Event()
        self.finish = asyncio.Event()
        self.received = []
        self.closed = False
        self.fail = False
        self.clear_count = 0

    async def __call__(self, ctx):
        try:
            self.received.append((yield Boundary.MODEL_REQUEST))
            self.model_ready.set()
            await self.stop_ready.wait()
            if self.fail:
                raise RuntimeError("model failed")
            self.received.append((yield Boundary.MODEL_REQUEST))
            await self.finish.wait()
            followup = yield Boundary.WOULD_STOP
            self.received.append(followup)
            if followup:
                self.received.append((yield Boundary.MODEL_REQUEST))
                yield Boundary.WOULD_STOP
        finally:
            self.closed = True

    def clear(self):
        self.clear_count += 1

    async def compact(self):
        return CompactOutcome.NOTHING_TO_COMPACT


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.events = []
        self.handler = FakeHandler()
        self.runner = SessionRunner(self.handler, self.events.append)

    async def asyncTearDown(self):
        await self.runner.close()

    async def test_steering_is_ordered_and_drained_at_model_request(self):
        self.runner.submit("initial")
        await self.handler.model_ready.wait()
        self.runner.submit("first")
        self.runner.submit("second")
        self.assertIn("2 pending", self.runner.status)
        self.handler.stop_ready.set()
        await asyncio.sleep(0)
        self.assertIn(["first", "second"], self.handler.received)
        self.handler.finish.set()
        await self.runner.wait_idle()
        self.assertFalse(self.runner.busy)
        self.assertIsInstance(self.events[-1], events.TurnFinished)

    async def test_would_stop_drains_late_steering(self):
        self.runner.submit("initial")
        await self.handler.model_ready.wait()
        self.handler.stop_ready.set()
        await asyncio.sleep(0)
        self.runner.submit("late")
        self.handler.finish.set()
        await self.runner.wait_idle()
        self.assertIn(["late"], self.handler.received)

    async def test_interrupt_closes_generator_and_discards_pending(self):
        self.runner.submit("initial")
        await self.handler.model_ready.wait()
        self.runner.submit("pending")
        self.runner.interrupt()
        self.runner.interrupt()
        await self.runner.wait_idle()
        self.assertTrue(self.handler.closed)
        self.assertTrue(self.events[-1].aborted)
        self.assertIn("0 pending", self.runner.status)

    async def test_close_immediately_after_submit(self):
        self.runner.submit("initial")
        await self.runner.close()
        self.assertFalse(self.runner.busy)
        self.assertTrue(self.handler.closed)

    async def test_error_allows_next_turn(self):
        self.handler.fail = True
        self.handler.stop_ready.set()
        self.runner.submit("first")
        await self.runner.wait_idle()
        self.assertTrue(any(isinstance(e, events.AgentError) for e in self.events))
        self.handler.fail = False
        self.handler.finish.set()
        self.runner.submit("second")
        await self.runner.wait_idle()
        self.assertEqual(
            sum(isinstance(e, events.TurnFinished) for e in self.events), 2
        )

    async def test_commands_require_idle(self):
        await self.runner.command("/clear")
        self.assertEqual(self.handler.clear_count, 1)
        self.runner.submit("work")
        await self.runner.command("/clear")
        self.assertEqual(self.handler.clear_count, 1)
        self.assertTrue(await self.runner.command("/help"))
        self.assertFalse(await self.runner.command("/quit"))

    async def test_manual_compaction_queues_messages_then_runs_them(self):
        ready = asyncio.Event()
        release = asyncio.Event()

        async def compact():
            ready.set()
            await release.wait()
            return CompactOutcome.COMPACTED

        self.handler.compact = compact
        self.handler.stop_ready.set()
        self.handler.finish.set()
        await self.runner.command("/compact")
        await ready.wait()
        self.runner.submit("after compaction")
        release.set()
        await self.runner.wait_idle()
        self.assertFalse(self.runner.busy)
        starts = [
            event for event in self.events if isinstance(event, events.TurnStarted)
        ]
        self.assertEqual([event.prompt for event in starts], ["after compaction"])

    async def test_interrupted_compaction_discards_pending_input(self):
        ready = asyncio.Event()

        async def compact():
            ready.set()
            await asyncio.Event().wait()

        self.handler.compact = compact
        await self.runner.command("/compact")
        await ready.wait()
        self.runner.submit("discard this")
        self.runner.interrupt()
        await self.runner.wait_idle()
        self.assertFalse(
            any(isinstance(event, events.TurnStarted) for event in self.events)
        )
        self.assertIn("0 pending", self.runner.status)
