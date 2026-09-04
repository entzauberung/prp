"""Protocol-neutral bounded Provider -> Tool -> Result agent loop.

The loop owns only public turn history and control bounds. Policy, approval and
tool persistence remain in the injected executor; a model response can never
change runtime state directly.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable, Mapping, Sequence
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
from prp_runtime.runtime.context import (
    StaticFact,
    render_static_facts,
    select_relevant_facts,
)

__all__ = [
    "AgentHistoryItem",
    "AgentHistoryWriter",
    "AgentLoop",
    "AgentLoopResult",
    "AgentLoopStatus",
    "REMOTE_ASSIGNMENT_PENDING",
    "AgentToolCall",
    "AgentToolContext",
    "AgentToolExecution",
    "AgentToolExecutor",
    "AgentToolResult",
    "AgentTurn",
]

DEFAULT_MAX_TOOL_ROUNDS = 8
DEFAULT_MAX_TOOL_ATTEMPTS = 16
_ROOT_RE = re.compile(
    r"(?:[A-Za-z]:)?(?:/|\\)(?:home|tmp|Users|root|var|etc|usr)(?:/|\\)[^\s'\"]+"
)
_SECRET_ASSIGN_RE = re.compile(
    r"(?i)\b(api_key|apikey|authorization|credential|password|secret|token)\s*[:=]\s*\S+"
)
_PRIVATE_REASONING_RE = re.compile(
    r"(?i)\b(chain_of_thought|private_reasoning|hidden_cot)\b"
)
_FORBIDDEN_JSON_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "chain_of_thought",
        "cot",
        "credential",
        "password",
        "provider_body",
        "raw_provider_body",
        "secret",
        "token",
    }
)

AgentHistoryWriter = Callable[[int, AgentHistoryItem], Awaitable[None]]


@unique
class AgentLoopStatus(StrEnum):
    """Terminal or resumable state of one bounded agent turn sequence."""

    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    WAITING_REMOTE = "WAITING_REMOTE"
    EXHAUSTED = "EXHAUSTED"


class AgentToolContext(DomainModel):
    """The bounded runtime context exposed to one tool authorization step."""

    run_id: str
    work_unit_id: str
    mode: AgentMode
    round_index: int = Field(ge=1)
    attempt_index: int = Field(ge=1)


REMOTE_ASSIGNMENT_PENDING = "remote_assignment_pending"


class AgentToolExecution(DomainModel):
    """One executor decision: result, approval pause, or remote-result wait."""

    call: AgentToolCall
    result: AgentToolResult | None = None
    awaiting_approval: bool = False
    awaiting_remote_result: bool = False
    reason: str | None = None

    @model_validator(mode="after")
    def _shape_is_closed(self) -> AgentToolExecution:
        states = (
            int(self.result is not None),
            int(self.awaiting_approval),
            int(self.awaiting_remote_result),
        )
        if sum(states) != 1:
            raise ValueError(
                "agent tool execution must contain a terminal result, approval pause "
                "or remote wait"
            )
        if self.result is not None and self.result.call_id != self.call.call_id:
            raise ValueError("agent tool result call_id does not match the tool call")
        if self.awaiting_approval and self.reason is None:
            raise ValueError("approval pause requires a reason")
        if self.awaiting_remote_result and self.reason is None:
            raise ValueError("remote wait requires a reason")
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
        elif self.status in (AgentLoopStatus.PAUSED, AgentLoopStatus.WAITING_REMOTE):
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
        static_facts: Sequence[StaticFact | Mapping[str, object]] = (),
    ) -> AgentLoopResult:
        """Execute one bounded sequence, returning facts suitable for settlement."""
        rounds_limit = self._bounded_limit(max_tool_rounds, self._max_tool_rounds)
        attempts_limit = self._bounded_limit(max_attempts, self._max_attempts)
        input = self._public_model_input(input, static_facts, work_unit_id=work_unit_id)
        instructions = _sanitize_public_text(instructions) if instructions else None
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
        tool_catalog = self._tool_catalog
        catalog_names = self._catalog_names
        catalog_loader = (
            None
            if self._tool_executor is None
            else getattr(self._tool_executor, "provider_catalog", None)
        )
        if catalog_names is None and callable(catalog_loader):
            try:
                tool_catalog = tuple(await catalog_loader())
                catalog_names = frozenset(tool.name for tool in tool_catalog)
            except Exception:
                # A text-only turn does not need a workspace runtime. Keep the
                # catalog empty and let any attempted tool call fail closed.
                tool_catalog = ()
                catalog_names = frozenset()
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

        for call_id, persisted_result in persisted_replays.items():
            if call_id in seen_results:
                continue
            seen_results[call_id] = persisted_result
            current_history.append(persisted_result)
            if history_writer is not None:
                await history_writer(len(current_history), persisted_result)
            if persisted_result.status is ToolCallStatus.FAILED:
                failed_tool_names.add(seen_calls[call_id].tool_name)
            else:
                failed_tool_names.discard(seen_calls[call_id].tool_name)

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
                execution = await self._execute_public_tool(
                    resume_pending_call,
                    context=AgentToolContext(
                        run_id=run_id,
                        work_unit_id=work_unit_id,
                        mode=mode,
                        round_index=1,
                        attempt_index=1,
                    ),
                )
                paused = self._pause_if_waiting(
                    execution,
                    current_history=current_history,
                    total_usage=total_usage,
                    last_request_id=last_request_id,
                    tool_rounds=tool_rounds,
                    attempts=attempts,
                    call_id=resume_pending_call.call_id,
                )
                if paused is not None:
                    return paused
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
                    history=tuple(
                        _sanitize_history_item(item) for item in current_history
                    ),
                    tools=tool_catalog,
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
            if catalog_names is not None:
                unknown_tools = sorted(
                    {call.tool_name for call in turn.tool_calls}
                    - catalog_names
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
                execution = await self._execute_public_tool(
                    call,
                    context=AgentToolContext(
                        run_id=run_id,
                        work_unit_id=work_unit_id,
                        mode=mode,
                        round_index=tool_rounds,
                        attempt_index=attempts,
                    ),
                )
                paused = self._pause_if_waiting(
                    execution,
                    current_history=current_history,
                    total_usage=total_usage,
                    last_request_id=last_request_id,
                    tool_rounds=tool_rounds,
                    attempts=attempts,
                    call_id=call.call_id,
                )
                if paused is not None:
                    return paused
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
    def _public_model_input(
        raw_input: str,
        static_facts: Sequence[StaticFact | Mapping[str, object]],
        *,
        work_unit_id: str,
    ) -> str:
        selected = select_relevant_facts(static_facts, work_unit_id=work_unit_id)
        rendered = render_static_facts(selected)
        text = raw_input
        if rendered and rendered not in text:
            text = f"{text}\n\n{rendered}"
        return _sanitize_public_text(text)

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

    async def _execute_public_tool(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        """Execute one public call and map a remote assignment into a wait."""

        assert self._tool_executor is not None
        try:
            return await self._tool_executor.execute(call, context=context)
        except Exception as error:
            from prp_runtime.tools.executor import RemoteToolAssignmentPending

            if isinstance(error, RemoteToolAssignmentPending):
                return AgentToolExecution(
                    call=call,
                    awaiting_remote_result=True,
                    reason=REMOTE_ASSIGNMENT_PENDING,
                )
            raise

    @staticmethod
    def _pause_if_waiting(
        execution: AgentToolExecution,
        *,
        current_history: list[AgentHistoryItem],
        total_usage: Usage | None,
        last_request_id: str | None,
        tool_rounds: int,
        attempts: int,
        call_id: str,
    ) -> AgentLoopResult | None:
        """Return a distinct pause for approval or remote wait, never a fake result."""

        if execution.awaiting_approval:
            status = AgentLoopStatus.PAUSED
        elif execution.awaiting_remote_result:
            status = AgentLoopStatus.WAITING_REMOTE
        else:
            return None
        return AgentLoopResult(
            status=status,
            history=tuple(current_history),
            usage=total_usage,
            provider_request_id=last_request_id,
            tool_rounds=tool_rounds,
            attempts=attempts,
            pending_call_ids=(call_id,),
        )

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


def _sanitize_public_text(text: str) -> str:
    """Strip host roots, credentials and private reasoning from model text."""
    redacted = _ROOT_RE.sub("[redacted-root]", text)
    redacted = _SECRET_ASSIGN_RE.sub(lambda match: f"{match.group(1)}=[redacted]", redacted)
    return _PRIVATE_REASONING_RE.sub("[redacted]", redacted)


def _sanitize_json(value: object) -> object:
    if isinstance(value, dict):
        cleaned: dict[str, object] = {}
        for key, nested in value.items():
            if str(key).lower() in _FORBIDDEN_JSON_KEYS:
                cleaned[key] = "[redacted]"
            else:
                cleaned[key] = _sanitize_json(nested)
        return cleaned
    if isinstance(value, list):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, str):
        return _sanitize_public_text(value)
    return value


def _sanitize_history_item(item: AgentHistoryItem) -> AgentHistoryItem:
    if isinstance(item, AgentTurn):
        updates: dict[str, object] = {}
        if item.text is not None:
            updates["text"] = _sanitize_public_text(item.text)
        if item.tool_calls:
            updates["tool_calls"] = tuple(
                call.model_copy(
                    update={"arguments": _sanitize_json(call.arguments)}  # type: ignore[arg-type]
                )
                for call in item.tool_calls
            )
        return item.model_copy(update=updates) if updates else item
    updates = {}
    if item.output:
        updates["output"] = _sanitize_public_text(item.output)
    if item.result is not None:
        updates["result"] = _sanitize_json(item.result)
    return item.model_copy(update=updates) if updates else item

