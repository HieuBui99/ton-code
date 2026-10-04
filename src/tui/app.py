from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.styles import Style
from rich.console import Console

from src.entities import events
from src.tui.branding import print_banner
from src.tui.renderer import Notice, TranscriptEvent, TranscriptRenderer
from src.tui.runner import SessionRunner, TurnHandler
from src.tui.status import ContextStatus


class TranscriptLogHandler(logging.Handler):
    def __init__(self, emit) -> None:
        super().__init__()
        self.emit_event = emit

    def emit(self, record: logging.LogRecord) -> None:
        self.emit_event(
            Notice(
                self.format(record),
                "yellow" if record.levelno >= logging.WARNING else "dim",
            )
        )


class TerminalApp:
    """Normal-screen prompt with one serialized append-only Rich output task."""

    def __init__(
        self,
        console: Console,
        *,
        model: str,
        cwd: Path,
        verbose: bool = False,
        context_window: int = 200_000,
        window_assumed: bool = False,
    ) -> None:
        self.console = console
        self.model = model
        self.cwd = cwd
        self.verbose = verbose
        self.context = ContextStatus(context_window, window_assumed)
        self.queue: asyncio.Queue[TranscriptEvent] = asyncio.Queue()
        self.renderer = TranscriptRenderer(console, verbose=verbose)
        self.runner: SessionRunner
        self.prompt: PromptSession[str]
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: int | None = None

    def emit(self, event: TranscriptEvent) -> None:
        if self._loop is not None and threading.get_ident() != self._loop_thread:
            self._loop.call_soon_threadsafe(self.queue.put_nowait, event)
        else:
            self.queue.put_nowait(event)

    def _invalidate(self) -> None:
        self.prompt.app.invalidate()

    def _message(self):
        return [
            *self._toolbar(),
            ("", "\n"),
            ("class:prompt", "ton> "),
            ("class:status", f"[{self.runner.status}] "),
        ]

    def _toolbar(self):
        return self.context.toolbar(
            self.model, self.prompt.app.output.get_size().columns
        )

    def _bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def submit(event) -> None:
            event.current_buffer.validate_and_handle()

        @bindings.add("escape", "enter")
        def newline(event) -> None:
            event.current_buffer.insert_text("\n")

        @bindings.add("c-c")
        def interrupt(event) -> None:
            if self.runner.busy:
                self.runner.interrupt()
            else:
                event.current_buffer.reset()

        @bindings.add("c-d")
        def eof(event) -> None:
            if not event.current_buffer.text:
                event.app.exit(exception=EOFError())
            else:
                event.current_buffer.delete()

        return bindings

    async def _render_events(self) -> None:
        while True:
            event = await self.queue.get()
            try:
                if isinstance(event, events.ContextUsage):
                    self.context.tokens = event.input_tokens
                    self.context.estimated = event.estimated
                    self._invalidate()
                    continue
                # Collect text first so incomplete deltas and hidden thinking do
                # not redraw the prompt. Only completed output touches the terminal.
                with self.console.capture() as capture:
                    self.renderer.render(event)
                text = capture.get()
                if text:
                    await run_in_terminal(lambda text=text: self._write(text))
            finally:
                self.queue.task_done()

    def _write(self, text: str) -> None:
        self.console.file.write(text)
        self.console.file.flush()

    async def _input_loop(self) -> None:
        while True:
            try:
                text = await self.prompt.prompt_async()
            except EOFError:
                return
            except KeyboardInterrupt:
                self.runner.interrupt()
                continue
            if not text.strip():
                continue
            if text.strip().startswith("/"):
                if not await self.runner.command(text.strip()):
                    return
            else:
                self.runner.submit(text)

    async def run(self, handler: TurnHandler) -> None:
        self._loop = asyncio.get_running_loop()
        self._loop_thread = threading.get_ident()
        self.runner = SessionRunner(
            handler, self.emit, invalidate=self._invalidate, verbose=self.verbose
        )
        self.prompt = PromptSession(
            message=self._message,
            multiline=True,
            prompt_continuation="… ",
            key_bindings=self._bindings(),
            history=InMemoryHistory(),
            erase_when_done=True,
            style=Style.from_dict(
                {
                    "prompt": "bold ansigreen",
                    "status": "ansibrightblack",
                    "model": "ansicyan",
                    "context": "ansibrightblack",
                    "context.warning": "ansiyellow",
                    "context.danger": "ansired",
                }
            ),
            reserve_space_for_menu=0,
        )
        print_banner(
            self.console,
            model=self.model,
            cwd=self.cwd,
            window=self.context.window,
            window_assumed=self.context.window_assumed,
        )

        root_logger = logging.getLogger()
        old_handlers, old_level = root_logger.handlers[:], root_logger.level
        log_handler = TranscriptLogHandler(self.emit)
        log_handler.setFormatter(
            logging.Formatter("%(levelname)s %(name)s: %(message)s")
        )
        root_logger.handlers = [log_handler]
        root_logger.setLevel(logging.INFO if self.verbose else logging.WARNING)
        renderer = asyncio.create_task(self._render_events())
        reader = asyncio.create_task(self._input_loop())
        try:
            done, _ = await asyncio.wait(
                {reader, renderer}, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
        finally:
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
            try:
                await self.runner.close()
                # Deliver any thread-originated events scheduled on this loop.
                await asyncio.sleep(0)
                if not renderer.done():
                    drained = asyncio.create_task(self.queue.join())
                    try:
                        await asyncio.wait(
                            {drained, renderer}, return_when=asyncio.FIRST_COMPLETED
                        )
                        if renderer.done():
                            renderer.result()
                        await drained
                    finally:
                        drained.cancel()
                        await asyncio.gather(drained, return_exceptions=True)
                    await run_in_terminal(self.renderer.flush)
            finally:
                renderer.cancel()
                await asyncio.gather(renderer, return_exceptions=True)
                root_logger.handlers, root_logger.level = old_handlers, old_level
                self._loop = None
                self._loop_thread = None
