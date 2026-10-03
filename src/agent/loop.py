from __future__ import annotations

import enum
import logging
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent, DeferredToolRequests, ToolDenied
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    PartDeltaEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.tools import DeferredToolResults
from pydantic_ai.usage import RunUsage

from src.agent.deps import AgentDeps
from src.config.settings import settings
from src.context.compaction import (
    CompactOutcome,
    build_summary_message,
    estimate_history_tokens,
    microcompact,
    should_compact,
    split_tail,
    summarize_for_compaction,
)
from src.entities import events

if TYPE_CHECKING:
    from pydantic_ai.agent import AbstractAgent
    from pydantic_ai.models import Model

    from src.context.session_log import SessionLog

logger = logging.getLogger(__name__)
EventSink = Callable[[events.Event], None]


def _leg_input_tokens(messages: list[ModelMessage]) -> int:
    for message in reversed(messages):
        if isinstance(message, ModelResponse) and message.usage.input_tokens > 0:
            return message.usage.input_tokens + message.usage.cache_read_tokens
    return 0


class Boundary(enum.Enum):
    """A point in a turn where the runner may inject queued input or stop.

    ``MODEL_REQUEST`` — the runner drains **steering** and sends it back into the handler.
    ``WOULD_STOP`` — the runner drains **follow-up**; non-empty continues the turn, empty ends it.
    """

    MODEL_REQUEST = "model_request"
    WOULD_STOP = "would_stop"


class TurnContext:
    """What a turn handler may do while the runner drives it — currently just :meth:`emit`."""

    __slots__ = ("_sink", "prompt", "turn_id")

    def __init__(self, turn_id: int, prompt: str, sink: EventSink) -> None:
        self.turn_id = turn_id
        self.prompt = prompt
        self._sink = sink

    def emit(self, event: events.Event) -> None:
        """Stream one event to the TUI."""
        self._sink(event)


class AgentTurnHandler:
    """
    Drive ``agent.iter()`` as the harness turn handler, carrying history across turns.
    """

    def __init__(
        self,
        agent: AbstractAgent[AgentDeps, str | DeferredToolRequests],
        *,
        deps: AgentDeps,
        session_log: SessionLog | None = None,
        session_id: str | None = None,
        message_history: list[ModelMessage] | None = None,
        compaction_model: Model | None = None,
    ) -> None:
        self._agent = agent
        self._deps = deps
        # Opik Thread id for this session's turns; ``None`` makes the root span a nullcontext.
        self._session_id = session_id
        # The running conversation, carried across turns; ``--resume`` seeds the replayed history.
        self.message_history: list[ModelMessage] = list(message_history or [])
        # Append-only JSONL session log
        self._session_log = session_log
        # Persisted-count cursor: the seeded prefix counts as persisted so resume never re-writes it.
        self._persisted_count = len(self.message_history)
        # Compaction summarizer: 
        # ``None`` disables the cascade.
        self._compaction_model = compaction_model

        self._last_input_tokens = 0

        self._announced_tool_calls: set[str] = set()

    @property
    def last_input_tokens(self) -> int:
        """Input-token occupancy of the most recent leg — the TUI footer gauge + compaction trigger
        read this single number.

        Normally the last populated ``ModelResponse.usage`` of the leg (``input_tokens +
        cache_read_tokens``), NOT the leg's cumulative usage: it tracks real context occupancy
        instead of overcounting ~Nx across N tool rounds. The one exception is the instant a full
        compaction lands: the gauge is reseeded with the chars≈/4 estimate of the kept
        ``[summary, *tail]`` so the footer drops immediately (understating, never inflating), and
        the next leg's provider number overwrites it. ``0`` before any leg (and when
        no response reported usage).
        """
        return self._last_input_tokens

    async def __call__(self, ctx: TurnContext) -> AsyncGenerator[Boundary, list[str]]:
        next_prompt: str | None = ctx.prompt
        pending_results: DeferredToolResults | None = None

        try:
            while True:
                steering = yield Boundary.MODEL_REQUEST

                if pending_results is not None:
                    # Deferred resume leg: no new prompt, so steering rides the history instead.
                    self._append_steering(steering)
                    output = await self._run_turn(ctx, deferred_results=pending_results)
                    pending_results = None
                else:
                    prompt = self._compose_prompt(next_prompt, steering)
                    next_prompt = None
                    if prompt is None:
                        # Nothing to ask the model: stop cleanly (drain follow-up below).
                        output = ""
                        self._persist_turn()
                        follow_ups = yield Boundary.WOULD_STOP
                        if not follow_ups:
                            return
                        next_prompt = "\n".join(follow_ups)
                        continue
                    self._heal_dangling_tool_calls()
                    output = await self._run_turn(ctx, prompt=prompt)

                if isinstance(output, DeferredToolRequests):
                    # A gated tool paused the run: resolve approvals and loop back to resume.
                    pending_results = await self._resolve_deferred(ctx, output)
                    continue

                # --- would-stop boundary: persist, compaction cascade, drain follow-up ---
                self._persist_turn()
                await self._maybe_auto_compact()

                follow_ups = yield Boundary.WOULD_STOP
                if not follow_ups:
                    return
                next_prompt = "\n".join(follow_ups)
        finally:
            self._persist_turn()

    @staticmethod
    def _compose_prompt(base: str | None, steering: list[str]) -> str | None:
        parts = [*steering]
        if base is not None:
            parts.append(base)
        if not parts:
            return None
        return "\n".join(parts)

    def _heal_dangling_tool_calls(self) -> None:
        """Heal a history ending in unprocessed tool calls before a new-prompt.

        A crashed leg (a tool's ``ModelRetry`` budget exhausted mid-resume) or an Esc-abort while
        a permission prompt is pending leaves the history's last message a ``ModelResponse`` whose
        tool calls never got results — pydantic-ai then rejects EVERY later prompt ("unprocessed
        tool calls"), bricking the session. Synthesize one interrupted-tool return per dangling
        call so the next leg is accepted; a deferred *resume* leg never comes through here (it
        needs the dangling call for its ``DeferredToolResults``). Covers a ``--resume`` of a
        crash-persisted log too — the seeded history heals on the first prompt.
        """
        if not self.message_history:
            return
        last = self.message_history[-1]
        if not isinstance(last, ModelResponse) or not last.tool_calls:
            return
        logger.warning(
            "healing %d unprocessed tool call(s) left by a crashed or aborted turn",
            len(last.tool_calls),
        )
        returns = [
            ToolReturnPart(
                tool_name=call.tool_name,
                tool_call_id=call.tool_call_id,
                content=(
                    "Tool call interrupted before completing (the turn crashed or was aborted); "
                    "re-issue it if still needed."
                ),
            )
            for call in last.tool_calls
        ]
        self.message_history.append(ModelRequest(parts=returns))

    def _append_steering(self, steering: list[str]) -> None:
        """Append steering to the history as a user message before a deferred resume.

        A resume leg carries no new ``user_prompt``, so steering cannot ride the prompt.
        """
        if not steering:
            return
        content = "\n".join(steering)
        logger.debug(
            "appending steering to history before deferred resume: %r", content
        )
        self.message_history.append(
            ModelRequest(parts=[UserPromptPart(content=content)])
        )

    def _persist_turn(self) -> None:
        """
        Append the messages beyond the persisted-count cursor to the session log
        """
        if self._session_log is None:
            return
        new_messages = self.message_history[self._persisted_count :]
        if not new_messages:
            return
        try:
            self._session_log.append_turn(new_messages)
        except OSError:
            logger.warning("failed to persist turn to session log", exc_info=True)
            return
        self._persisted_count = len(self.message_history)

    async def _maybe_auto_compact(self) -> None:
        """
        Run the two-tier compaction cascade at would-stop
        """
        if self._compaction_model is None:
            return
        usage = RunUsage(input_tokens=self._last_input_tokens)

        window = (
            self._deps.context_window_tokens
            or settings.compaction_context_window_tokens
        )
        full = should_compact(
            usage,
            window=window,
            reserve=settings.compaction_reserve_fraction,
            enabled=settings.compaction_enabled,
        )
        micro = should_compact(
            usage,
            window=window,
            reserve=settings.microcompaction_reserve_fraction,
            enabled=settings.compaction_enabled,
        )
        if full:
            outcome = await self.compact()
            if outcome is not CompactOutcome.COMPACTED:

                logger.info(
                    "full compaction trigger fired but did not land: outcome=%s",
                    outcome.name,
                )
        elif micro:
            elided = self._microcompact()
            if elided == 0:
                logger.info(
                    "microcompaction trigger fired but elided nothing "
                    "(no eligible tool output in the compactable prefix)"
                )

    async def compact(self) -> CompactOutcome:
        """Full compaction: replace history with ``[summary, *tail]``"""
        split = split_tail(
            self.message_history,
            keep_recent_tokens=settings.compaction_keep_recent_tokens,
        )
        if split == 0:
            return CompactOutcome.NOTHING_TO_COMPACT
        skeleton = await summarize_for_compaction(
            self.message_history, model=self._compaction_model
        )
        if skeleton is None:
            return CompactOutcome.SUMMARIZER_FAILED
        before_tokens = self._last_input_tokens
        summary_message = build_summary_message(skeleton)
        tail = self.message_history[split:]
        if self._session_log is not None:
            try:
                self._session_log.append_compaction(summary_message, tail)
            except OSError:
                logger.warning("failed to persist compaction checkpoint", exc_info=True)
        self.message_history = [summary_message, *tail]

        self._last_input_tokens = estimate_history_tokens(self.message_history)
        self._persisted_count = len(self.message_history)
        self._deps.emit(
            events.ContextCompacted(
                before_tokens=before_tokens, kept_messages=len(tail)
            )
        )
        return CompactOutcome.COMPACTED

    def clear(self) -> None:
        """Reset the conversation to empty — the body of ``/clear``.

        Appends a ``clear`` marker FIRST (when a log is wired) so ``--resume`` replays to the
        post-clear state; ``OSError`` is logged and swallowed. The summarize-before-wipe memory
        write-back lives at the call site.
        """
        if self._session_log is not None:
            try:
                self._session_log.append_clear()
            except OSError:
                logger.warning("failed to persist the clear marker", exc_info=True)
        self.message_history = []
        self._persisted_count = 0
        self._last_input_tokens = 0
        self._announced_tool_calls.clear()

    def _microcompact(self) -> int:
        """
        Microcompaction: blank old tool-output bodies, in memory only
        """
        new_messages, elided = microcompact(
            self.message_history,
            keep_recent_tokens=settings.compaction_keep_recent_tokens,
        )
        if elided == 0:
            return 0
        self.message_history = new_messages
        self._deps.emit(
            events.ContextMicrocompacted(
                elided_count=elided, before_tokens=self._last_input_tokens
            )
        )
        return elided

    async def _run_turn(
        self,
        ctx: TurnContext,
        *,
        prompt: str | None = None,
        deferred_results: DeferredToolResults | None = None,
    ) -> str | DeferredToolRequests:
        async with self._agent.iter(
            prompt,
            deps=self._deps,
            message_history=self.message_history,
            deferred_tool_results=deferred_results,
        ) as run:
            try:
                async for node in run:
                    if Agent.is_model_request_node(node):
                        await self._stream_model_node(ctx, node, run)
                    elif Agent.is_call_tools_node(node):
                        await self._stream_tool_node(ctx, node, run)
            finally:
                self.message_history = run.all_messages()

                self._last_input_tokens = _leg_input_tokens(self.message_history)
        # pydantic-ai may coalesce adjacent same-role prior messages (notably the two ModelRequests
        # a full compaction leaves), shrinking the persisted prefix. Clamp the cursor to the count
        # preceding this leg's new messages so the next persist never drops a fresh message.
        persisted_floor = len(self.message_history) - len(run.result.new_messages())
        self._persisted_count = min(self._persisted_count, persisted_floor)
        return run.result.output

    async def _resolve_deferred(
        self, ctx: TurnContext, requests: DeferredToolRequests
    ) -> DeferredToolResults:
        """Turn a deferred pause into ``DeferredToolResults`` via the gate + the resolver (ADR-0003 §3).

        Allow maps to ``True``; deny to ``ToolDenied`` so the model sees the reason on the resume leg.
        """
        approvals: dict[str, bool | ToolDenied] = {}
        for call in requests.approvals:
            approvals[call.tool_call_id] = True
        return requests.build_results(approvals=approvals)

    async def _stream_model_node(
        self, ctx: TurnContext, node: object, run: object
    ) -> None:
        """Stream one model-request node, emitting text/thinking deltas as they arrive."""
        async with node.stream(run.ctx) as request_stream:  # type: ignore[attr-defined]
            async for event in request_stream:
                self._emit_for_stream_event(ctx, event)

    async def _stream_tool_node(
        self, ctx: TurnContext, node: object, run: object
    ) -> None:
        async with node.stream(run.ctx) as tool_stream:  # type: ignore[attr-defined]
            async for event in tool_stream:
                self._emit_for_tool_event(ctx, event)

    def _emit_for_tool_event(self, ctx: TurnContext, event: object) -> None:
        """Map one Pydantic AI tool-stream event to a canonical decode tool event (or ignore it).

        ``ok`` is keyed off ``ToolReturnPart.outcome`` (not a bare isinstance check): a gate deny
        arrives as a ``ToolReturnPart`` with ``outcome == "denied"``, which would otherwise render
        as a green success panel.
        """
        if isinstance(event, FunctionToolCallEvent):
            call = event.part
            if call.tool_call_id in self._announced_tool_calls:
                return
            self._announced_tool_calls.add(call.tool_call_id)
            ctx.emit(
                events.ToolCallStarted(
                    tool_call_id=call.tool_call_id,
                    name=call.tool_name,
                    args=call.args_as_json_str(),
                )
            )
        elif isinstance(event, FunctionToolResultEvent):
            result = event.part
            if isinstance(result, ToolReturnPart):
                # outcome ∈ {"success", "failed", "denied"}; only "success" is ok.
                ok = result.outcome == "success"
                output = result.model_response_str()
            elif isinstance(result, RetryPromptPart):
                ok = False
                output = result.model_response()
            else:  # pragma: no cover - defensive: the union is exhausted above
                ok = False
                output = str(getattr(result, "content", ""))
            ctx.emit(
                events.ToolResult(
                    tool_call_id=result.tool_call_id,
                    name=result.tool_name or "",
                    output=output,
                    ok=ok,
                )
            )

    def _emit_for_stream_event(self, ctx: TurnContext, event: object) -> None:
        """Map one Pydantic AI stream event (text/thinking start + delta) to a decode event."""
        if isinstance(event, PartStartEvent):
            part = event.part
            if isinstance(part, TextPart) and part.content:
                ctx.emit(events.AssistantTextDelta(text=part.content))
            elif isinstance(part, ThinkingPart) and part.content:
                ctx.emit(events.ThinkingDelta(text=part.content))
        elif isinstance(event, PartDeltaEvent):
            delta = event.delta
            if isinstance(delta, TextPartDelta) and delta.content_delta:
                ctx.emit(events.AssistantTextDelta(text=delta.content_delta))
            elif isinstance(delta, ThinkingPartDelta) and delta.content_delta:
                ctx.emit(events.ThinkingDelta(text=delta.content_delta))
