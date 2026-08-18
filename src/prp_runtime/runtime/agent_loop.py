"""Protocol-neutral bounded Provider -> Tool -> Result agent loop.

The loop owns only public turn history and control bounds. Policy, approval and
tool persistence remain in the injected executor; a model response can never
change runtime state directly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from enum import StrEnum, unique
from typing import Protocol

from pydantic import Field, model_validator

from prp_runtime.domain.enums import AgentMode, ToolCallStatus
from prp_runtime.domain.errors import DomainValidationError
from prp_runtime.domain.models import (
    MAX_AGENT_HISTORY_ITEMS,
    AgentHistoryItem,
    AgentToolCall,
    AgentToolResult,
    AgentTurn,
    DomainModel,
    ErrorCategory,
    ErrorInfo,
    ProviderToolDescriptor,
    Usage,
)
from prp_runtime.domain.values import utc_now
from prp_runtime.policy.models import DevScope, serialize_dev_evidence
from prp_runtime.providers.base import ModelProfile, ProviderAdapter, ProviderRequest

__all__ = [
    "AgentHistoryItem",
    "AgentHistoryWriter",
    "AgentLoop",
    "AgentLoopResult",
    "AgentLoopStatus",
    "AgentToolCall",
    "AgentToolContext",
    "AgentToolExecution",
    "AgentToolExecutor",
    "AgentToolResult",
    "AgentTurn",
]

DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_MAX_TOOL_ATTEMPTS = 16

AgentHistoryWriter = Callable[[int, AgentHistoryItem], Awaitable[None]]


@unique
class AgentLoopStatus(StrEnum):
    """Terminal or resumable state of one bounded agent turn sequence."""

    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    EXHAUSTED = "EXHAUSTED"


class AgentToolContext(DomainModel):
    """The bounded runtime context exposed to one tool authorization step."""

    run_id: str
    work_unit_id: str
    mode: AgentMode
    round_index: int = Field(ge=1)
    attempt_index: int = Field(ge=1)


class AgentToolExecution(DomainModel):
    """One executor decision, including an approval pause without fake failure."""

    call: AgentToolCall
    result: AgentToolResult | None = None
    awaiting_approval: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _shape_is_closed(self) -> AgentToolExecution:
        if self.awaiting_approval == (self.result is not None):
            raise ValueError(
                "agent tool execution must contain a terminal result or approval pause"
            )
        if self.result is not None and self.result.call_id != self.call.call_id:
            raise ValueError("agent tool result call_id does not match the tool call")
        if self.awaiting_approval and self.reason is None:
            raise ValueError("approval pause requires a reason")
        return self


class AgentToolExecutor(Protocol):
    """Policy-gated tool boundary consumed by :class:`AgentLoop`."""

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        """Authorize and execute one public tool call, or pause for approval."""


class AgentLoopResult(DomainModel):
    """Bounded facts returned by one agent loop invocation."""

    status: AgentLoopStatus
    text: str | None = None
    history: tuple[AgentHistoryItem, ...] = ()
    usage: Usage | None = None
    provider_request_id: str | None = None
    tool_rounds: int = Field(default=0, ge=0)
    attempts: int = Field(default=0, ge=0)
    pending_call_ids: tuple[str, ...] = ()
    error: ErrorInfo | None = None

    @model_validator(mode="after")
    def _result_shape_is_consistent(self) -> AgentLoopResult:
        if self.status is AgentLoopStatus.COMPLETED:
            if self.text is None or self.error is not None or self.pending_call_ids:
                raise ValueError("completed agent loop must contain only final text")
        elif self.status is AgentLoopStatus.PAUSED:
            if self.text is not None or self.error is not None or not self.pending_call_ids:
                raise ValueError("paused agent loop must contain pending tool calls")
        elif self.status is AgentLoopStatus.EXHAUSTED:
            if self.text is not None or self.error is None or self.pending_call_ids:
                raise ValueError("exhausted agent loop must contain a structured error")
        return self

    @property
    def succeeded(self) -> bool:
        """Whether a verifier may receive a final text artifact."""
        return self.status is AgentLoopStatus.COMPLETED

    def serialize_dev_evidence(self, *, scope: DevScope) -> dict[str, object]:
        """Publish loop facts with scope-derived DEV-only evidence metadata."""
        return serialize_dev_evidence(self.model_dump(mode="json"), scope=scope)


class AgentLoop:
    """Run bounded provider turns and policy-controlled tool calls."""

    def __init__(
        self,
        adapter: ProviderAdapter,
        profile: ModelProfile,
        *,
        tool_executor: AgentToolExecutor | None = None,
        tool_catalog: tuple[ProviderToolDescriptor, ...] | None = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        max_attempts: int = DEFAULT_MAX_TOOL_ATTEMPTS,
    ) -> None:
        if max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be at least one")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least one")
        self._adapter = adapter
        self._profile = profile
        self._tool_executor = tool_executor
        self._tool_catalog = tuple(tool_catalog or ())
        self._catalog_names = (
            None
            if tool_catalog is None
            else frozenset(descriptor.name for descriptor in self._tool_catalog)
        )
        self._max_tool_rounds = max_tool_rounds
        self._max_attempts = max_attempts

    async def execute(
        self,
        *,
        input: str,
        instructions: str | None = None,
        json_schema: str | None = None,
        history: tuple[AgentHistoryItem, ...] = (),
        attempt_id: str | None = None,
        run_id: str = "run_agent_loop",
        work_unit_id: str = "wu_agent_loop",
        mode: AgentMode = AgentMode.NORMAL,
        deadline: datetime | None = None,
        max_tool_rounds: int | None = None,
        max_attempts: int | None = None,
        history_writer: AgentHistoryWriter | None = None,
        resume_pending_call: AgentToolCall | None = None,
        replay_results: Mapping[str, AgentToolResult] | None = None,
    ) -> AgentLoopResult:
        """Execute one bounded sequence, returning facts suitable for settlement."""
        rounds_limit = self._bounded_limit(max_tool_rounds, self._max_tool_rounds)
        attempts_limit = self._bounded_limit(max_attempts, self._max_attempts)
        current_history = list(history)
        if len(current_history) > MAX_AGENT_HISTORY_ITEMS:
            return self._exhausted(
                current_history,
                None,
                None,
                0,
                0,
                ErrorInfo(
                    category=ErrorCategory.PROVIDER_ERROR,
                    message="agent loop history limit exhausted",
                ),
            )
        history_error = self._validate_history(current_history)
        if history_error is not None:
            return self._exhausted(
                current_history,
                None,
                None,
                0,
                0,
                ErrorInfo(category=ErrorCategory.INVALID_REQUEST, message=history_error),
            )
        seen_results: dict[str, AgentToolResult] = {
            item.call_id: item
            for item in current_history
            if isinstance(item, AgentToolResult)
        }
        persisted_replays = dict(replay_results or {})
        if any(
            not isinstance(call_id, str)
            or not isinstance(result, AgentToolResult)
            or result.call_id != call_id
            for call_id, result in persisted_replays.items()
        ):
            return self._exhausted(
                current_history,
                None,
                None,
                0,
                0,
                ErrorInfo(
                    category=ErrorCategory.INVALID_REQUEST,
                    message="replay result key does not match its tool call",
                ),
            )
        for call_id, persisted_result in persisted_replays.items():
            history_result = seen_results.get(call_id)
            if history_result is not None and history_result != persisted_result:
                return self._exhausted(
                    current_history,
                    None,
                    None,
                    0,
                    0,
                    ErrorInfo(
                        category=ErrorCategory.INVALID_REQUEST,
                        message="replay result conflicts with persisted history",
                    ),
                )
        seen_calls: dict[str, AgentToolCall] = {
            call.call_id: call
            for item in current_history
            if isinstance(item, AgentTurn)
            for call in item.tool_calls
        }
        if any(call_id not in seen_calls for call_id in persisted_replays):
            return self._exhausted(
                current_history,
                None,
                None,
                0,
                0,
                ErrorInfo(
                    category=ErrorCategory.INVALID_REQUEST,
                    message="replay result does not match a known tool call",
                ),
            )
        if any(
            item.status is ToolCallStatus.UNKNOWN
            for item in current_history
            if isinstance(item, AgentToolResult)
        ) or any(
            item.status is ToolCallStatus.UNKNOWN for item in persisted_replays.values()
        ):
            return self._exhausted(
                current_history,
                None,
                None,
                0,
                0,
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="unconfirmed tool outcome cannot be replayed",
                ),
            )
        failed_tool_names = {
            seen_calls[item.call_id].tool_name
            for item in current_history
            if (
                isinstance(item, AgentToolResult)
                and item.status is ToolCallStatus.FAILED
                and item.call_id in seen_calls
            )
        }
        total_usage: Usage | None = None
        last_request_id: str | None = None
        tool_rounds = 0
        attempts = 0

        if resume_pending_call is not None:
            previous_call = seen_calls.get(resume_pending_call.call_id)
            if previous_call is None or previous_call != resume_pending_call:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.INVALID_REQUEST,
                        message="resume tool call is absent or conflicts with history",
                    ),
                )
            if resume_pending_call.call_id not in seen_results:
                if self._tool_executor is None:
                    return self._exhausted(
                        current_history,
                        total_usage,
                        last_request_id,
                        tool_rounds,
                        attempts,
                        ErrorInfo(
                            category=ErrorCategory.PROVIDER_ERROR,
                            message="provider requested a tool but no tool executor is configured",
                        ),
                    )
                execution = await self._tool_executor.execute(
                    resume_pending_call,
                    context=AgentToolContext(
                        run_id=run_id,
                        work_unit_id=work_unit_id,
                        mode=mode,
                        round_index=1,
                        attempt_index=1,
                    ),
                )
                if execution.awaiting_approval:
                    return AgentLoopResult(
                        status=AgentLoopStatus.PAUSED,
                        history=tuple(current_history),
                        usage=total_usage,
                        provider_request_id=last_request_id,
                        tool_rounds=tool_rounds,
                        attempts=attempts,
                        pending_call_ids=(resume_pending_call.call_id,),
                    )
                assert execution.result is not None
                seen_results[resume_pending_call.call_id] = execution.result
                if execution.result.status is ToolCallStatus.FAILED:
                    failed_tool_names.add(resume_pending_call.tool_name)
                else:
                    failed_tool_names.discard(resume_pending_call.tool_name)
                current_history.append(execution.result)
                if history_writer is not None:
                    await history_writer(len(current_history), execution.result)
                if execution.result.status is ToolCallStatus.UNKNOWN:
                    return self._exhausted(
                        current_history,
                        total_usage,
                        last_request_id,
                        tool_rounds,
                        attempts,
                        ErrorInfo(
                            category=ErrorCategory.UNKNOWN,
                            message="unconfirmed tool outcome cannot be replayed",
                        ),
                    )
                tool_rounds = 1

        while True:
            if self._deadline_reached(deadline):
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.DEADLINE_EXCEEDED,
                        message="agent loop deadline exceeded before provider dispatch",
                    ),
                )
            if attempts >= attempts_limit:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.BUDGET_EXCEEDED,
                        message="agent loop attempt limit exhausted",
                    ),
                )

            try:
                request = ProviderRequest.for_profile(
                    self._profile,
                    input=input,
                    instructions=instructions,
                    json_schema=json_schema,
                    history=tuple(current_history),
                    tools=self._tool_catalog,
                )
            except (DomainValidationError, ValueError):
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.INVALID_REQUEST,
                        message="provider request violates the native contract",
                    ),
                )
            response = await self._adapter.complete(request)
            attempts += 1
            if response.usage is not None:
                total_usage = (
                    response.usage
                    if total_usage is None
                    else total_usage + response.usage
                )
            if response.provider_request_id is not None:
                last_request_id = response.provider_request_id

            try:
                turn = response.turn
            except ValueError:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.PROVIDER_ERROR,
                        message="provider response violates the native contract",
                    ),
                )
            current_history.append(turn)
            if history_writer is not None:
                await history_writer(len(current_history), turn)
            if turn.text is not None:
                if failed_tool_names:
                    return self._exhausted(
                        current_history,
                        total_usage,
                        last_request_id,
                        tool_rounds,
                        attempts,
                        ErrorInfo(
                            category=ErrorCategory.VERIFICATION_FAILED,
                            message="a failed tool call was not successfully recovered",
                        ),
                    )
                return AgentLoopResult(
                    status=AgentLoopStatus.COMPLETED,
                    text=turn.text,
                    history=tuple(current_history),
                    usage=total_usage,
                    provider_request_id=last_request_id,
                    tool_rounds=tool_rounds,
                    attempts=attempts,
                )

            tool_rounds += 1
            if tool_rounds > rounds_limit:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.BUDGET_EXCEEDED,
                        message="agent loop tool round limit exhausted",
                    ),
                )
            if self._tool_executor is None:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.PROVIDER_ERROR,
                        message="provider requested a tool but no tool executor is configured",
                    ),
                )
            if self._catalog_names is not None:
                unknown_tools = sorted(
                    {call.tool_name for call in turn.tool_calls}
                    - self._catalog_names
                )
                if unknown_tools:
                    return self._exhausted(
                        current_history,
                        total_usage,
                        last_request_id,
                        tool_rounds,
                        attempts,
                        ErrorInfo(
                            category=ErrorCategory.INVALID_REQUEST,
                            message="provider requested a tool outside the public catalog",
                        ),
                    )
            if attempts >= attempts_limit:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.BUDGET_EXCEEDED,
                        message="agent loop attempt limit exhausted before tool execution",
                    ),
                )

            for call in turn.tool_calls:
                previous_call = seen_calls.get(call.call_id)
                if previous_call is not None:
                    if previous_call.model_dump(mode="json") != call.model_dump(mode="json"):
                        return self._exhausted(
                            current_history,
                            total_usage,
                            last_request_id,
                            tool_rounds,
                            attempts,
                            ErrorInfo(
                                category=ErrorCategory.INVALID_REQUEST,
                                message="tool call id was reused with different arguments",
                            ),
                        )
                    replay = seen_results.get(call.call_id) or persisted_replays.get(
                        call.call_id
                    )
                    if replay is None:
                        return self._exhausted(
                            current_history,
                            total_usage,
                            last_request_id,
                            tool_rounds,
                            attempts,
                            ErrorInfo(
                                category=ErrorCategory.UNKNOWN,
                                message="duplicate tool call has no persisted result",
                            ),
                        )
                    current_history.append(replay)
                    if history_writer is not None:
                        await history_writer(len(current_history), replay)
                    if replay.status is ToolCallStatus.FAILED:
                        failed_tool_names.add(call.tool_name)
                    else:
                        failed_tool_names.discard(call.tool_name)
                    continue

                seen_calls[call.call_id] = call
                execution = await self._tool_executor.execute(
                    call,
                    context=AgentToolContext(
                        run_id=run_id,
                        work_unit_id=work_unit_id,
                        mode=mode,
                        round_index=tool_rounds,
                        attempt_index=attempts,
                    ),
                )
                if execution.awaiting_approval:
                    return AgentLoopResult(
                        status=AgentLoopStatus.PAUSED,
                        history=tuple(current_history),
                        usage=total_usage,
                        provider_request_id=last_request_id,
                        tool_rounds=tool_rounds,
                        attempts=attempts,
                        pending_call_ids=(call.call_id,),
                    )
                assert execution.result is not None
                seen_results[call.call_id] = execution.result
                if execution.result.status is ToolCallStatus.FAILED:
                    failed_tool_names.add(call.tool_name)
                else:
                    failed_tool_names.discard(call.tool_name)
                current_history.append(execution.result)
                if history_writer is not None:
                    await history_writer(len(current_history), execution.result)
                if execution.result.status is ToolCallStatus.UNKNOWN:
                    return self._exhausted(
                        current_history,
                        total_usage,
                        last_request_id,
                        tool_rounds,
                        attempts,
                        ErrorInfo(
                            category=ErrorCategory.UNKNOWN,
                            message="unconfirmed tool outcome cannot be replayed",
                        ),
                    )
            if len(current_history) > MAX_AGENT_HISTORY_ITEMS:
                return self._exhausted(
                    current_history,
                    total_usage,
                    last_request_id,
                    tool_rounds,
                    attempts,
                    ErrorInfo(
                        category=ErrorCategory.PROVIDER_ERROR,
                        message="agent loop history limit exhausted",
                    ),
                )

    @staticmethod
    def _bounded_limit(value: int | None, default: int) -> int:
        if value is None:
            return default
        if value < 1:
            raise ValueError("agent loop limits must be at least one")
        return min(value, default)

    @staticmethod
    def _deadline_reached(deadline: datetime | None) -> bool:
        return deadline is not None and utc_now() >= deadline

    @staticmethod
    def _validate_history(history: list[AgentHistoryItem]) -> str | None:
        """Reject orphaned or conflicting public history before provider dispatch."""
        calls: dict[str, AgentToolCall] = {}
        results: dict[str, AgentToolResult] = {}
        for item in history:
            if not isinstance(item, (AgentTurn, AgentToolResult)):
                return "history contains an unsupported item"
            if isinstance(item, AgentTurn):
                for call in item.tool_calls:
                    previous = calls.get(call.call_id)
                    if previous is not None and previous != call:
                        return "history reuses a tool call id with different arguments"
                    calls[call.call_id] = call
                continue
            matching_call = calls.get(item.call_id)
            if matching_call is None:
                return "history contains a tool result without a matching tool call"
            previous_result = results.get(item.call_id)
            if previous_result is not None and previous_result != item:
                return "history contains conflicting results for one tool call"
            results[item.call_id] = item
        return None

    @staticmethod
    def _exhausted(
        history: list[AgentHistoryItem],
        usage: Usage | None,
        provider_request_id: str | None,
        tool_rounds: int,
        attempts: int,
        error: ErrorInfo,
    ) -> AgentLoopResult:
        return AgentLoopResult(
            status=AgentLoopStatus.EXHAUSTED,
            history=tuple(history),
            usage=usage,
            provider_request_id=provider_request_id,
            tool_rounds=tool_rounds,
            attempts=attempts,
            error=error,
        )
