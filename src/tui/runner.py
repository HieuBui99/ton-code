from __future__ import annotations

import asyncio
import traceback
from collections import deque
from collections.abc import AsyncGenerator, Callable
from typing import Protocol

from src.agent.loop import Boundary, TurnContext
from src.context.compaction import CompactOutcome
from src.entities import events
from src.tui.renderer import Notice, TranscriptEvent, UserSubmitted


class TurnHandler(Protocol):
    def __call__(self, ctx: TurnContext) -> AsyncGenerator[Boundary, list[str]]: ...
    def clear(self) -> None: ...
    async def compact(self) -> CompactOutcome: ...


class SessionRunner:
    """Own one active turn and inject submitted messages at safe boundaries."""

    def __init__(
        self,
        handler: TurnHandler,
        emit: Callable[[TranscriptEvent], None],
        *,
        invalidate: Callable[[], None] = lambda: None,
        verbose: bool = False,
    ) -> None:
        self.handler = handler
        self.emit = emit
        self.invalidate = invalidate
        self.verbose = verbose
        self._pending: deque[str] = deque()
        self._task: asyncio.Task[None] | None = None
        self._turn_id = 0
        self._cancelling = False
        self._closed = False

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    @property
    def status(self) -> str:
        state = "stopping" if self._cancelling else "working" if self.busy else "idle"
        return f"{state} · {len(self._pending)} pending"

    def submit(self, prompt: str) -> None:
        if self._closed or not prompt.strip():
            return
        if self._cancelling:
            self.emit(
                Notice("Wait for interruption to finish before submitting.", "yellow")
            )
            return
        steering = self.busy
        self.emit(UserSubmitted(prompt, steering=steering))
        if steering:
            self._pending.append(prompt)
        else:
            self._turn_id += 1
            self._task = asyncio.create_task(self._run(prompt, self._turn_id))
        self.invalidate()

    def _drain(self) -> list[str]:
        result = list(self._pending)
        self._pending.clear()
        self.invalidate()
        return result

    async def _run(self, prompt: str, turn_id: int) -> None:
        self.emit(events.TurnStarted(turn_id, prompt))
        ctx = TurnContext(turn_id, prompt, self.emit)
        turn = self.handler(ctx)
        aborted = False
        try:
            boundary = await anext(turn)
            while True:
                if boundary not in (Boundary.MODEL_REQUEST, Boundary.WOULD_STOP):
                    raise RuntimeError(f"Unknown turn boundary: {boundary}")
                # No await between draining and resuming: no submission can get lost
                # in a finishing turn's empty WOULD_STOP decision.
                boundary = await turn.asend(self._drain())
        except StopAsyncIteration:
            pass
        except asyncio.CancelledError:
            aborted = True
        except Exception as exc:  # noqa: BLE001 - keep the interactive session alive
            self.emit(
                events.AgentError(traceback.format_exc() if self.verbose else str(exc))
            )
        finally:
            try:
                await turn.aclose()
            finally:
                if self._pending:
                    self._pending.clear()
                    self.emit(
                        Notice(
                            "Pending steering discarded; resubmit it to continue.",
                            "yellow",
                        )
                    )
                self.emit(events.TurnFinished(turn_id, aborted=aborted))
                self._task = None
                self._cancelling = False
                self.invalidate()

    def interrupt(self) -> None:
        if self.busy and not self._cancelling:
            self._cancelling = True
            assert self._task is not None
            # Let a newly created task enter its try/finally before cancellation.
            asyncio.get_running_loop().call_soon(self._task.cancel)
            self.invalidate()

    async def wait_idle(self) -> None:
        while self._task is not None:
            await self._task

    async def close(self) -> None:
        self._closed = True
        self.interrupt()
        await self.wait_idle()

    async def command(self, command: str) -> bool:
        """Return False when the input loop should exit."""
        if command == "/quit":
            return False
        if command == "/help":
            self.emit(
                Notice(
                    "/help  /clear  /compact  /quit\nEnter: submit · Alt+Enter: newline · Ctrl+C: interrupt/clear draft · Ctrl+D: quit on empty draft\nMessages submitted while working steer the next model request."
                )
            )
        elif command in ("/clear", "/compact"):
            if self.busy:
                self.emit(Notice(f"{command} requires an idle agent.", "yellow"))
            elif command == "/clear":
                self.handler.clear()
                self.emit(Notice("Conversation cleared."))
            else:
                # Manual compaction is cancellable and uses the same busy guard.
                self._task = asyncio.create_task(self._compact())
                self.invalidate()
        else:
            self.emit(Notice(f"Unknown command: {command}. Use /help.", "yellow"))
        return True

    async def _compact(self) -> None:
        completed = False
        try:
            result = await self.handler.compact()
            completed = True
            self.emit(Notice(f"Compaction: {result.value.replace('_', ' ')}."))
        except asyncio.CancelledError:
            self.emit(Notice("Compaction interrupted."))
        except Exception as exc:  # noqa: BLE001 - keep the interactive session alive
            self.emit(
                events.AgentError(traceback.format_exc() if self.verbose else str(exc))
            )
        finally:
            self._task = None
            self._cancelling = False
            # Steering cannot be injected into a standalone compaction call.
            if self._pending and not self._closed and completed:
                prompt = "\n".join(self._drain())
                self._turn_id += 1
                self._task = asyncio.create_task(self._run(prompt, self._turn_id))
            else:
                if self._pending:
                    self.emit(
                        Notice(
                            "Pending steering discarded; resubmit it to continue.",
                            "yellow",
                        )
                    )
                self._pending.clear()
            self.invalidate()
