"""Offline terminal fixture used by the PTY integration test."""

import asyncio
from pathlib import Path

from rich.console import Console

from src.agent.loop import Boundary
from src.context.compaction import CompactOutcome
from src.entities.events import AssistantTextDelta, ContextUsage
from src.tui.app import TerminalApp


class DemoHandler:
    async def __call__(self, ctx):
        steering = yield Boundary.MODEL_REQUEST
        prompt = ctx.prompt
        while True:
            ctx.emit(AssistantTextDelta(f"REPLY {prompt}\n"))
            for index in range(35):
                await asyncio.sleep(0.02)
                ctx.emit(AssistantTextDelta(f"stream line {index}: 漢字 café\n"))
                ctx.emit(ContextUsage((index + 1) * 1000))
            prompt = "\n".join((yield Boundary.WOULD_STOP))
            if not prompt:
                return
            steering = yield Boundary.MODEL_REQUEST
            if steering:
                prompt += "\n" + "\n".join(steering)

    def clear(self):
        pass

    async def compact(self):
        return CompactOutcome.NOTHING_TO_COMPACT


if __name__ == "__main__":
    asyncio.run(
        TerminalApp(
            Console(color_system=None), model="offline-test", cwd=Path.cwd()
        ).run(DemoHandler())
    )
