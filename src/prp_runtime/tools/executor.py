"""Policy-gated tool execution and lifecycle orchestration."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel, ConfigDict

from prp_runtime.domain.enums import AgentMode, ExecutionLocation, IsolationMode, ToolCallStatus
from prp_runtime.domain.models import DomainModel, ErrorCategory, ErrorInfo
from prp_runtime.domain.values import utc_now
from prp_runtime.settings import Settings
from prp_runtime.tools.models import ToolCall, ToolResult
from prp_runtime.tools.registry import ToolRegistry

if TYPE_CHECKING:
    from prp_runtime.policy.engine import PolicyDecision
    from prp_runtime.policy.models import CommandClass, Lease
else:
    PolicyDecision = Any
    CommandClass = Any
    Lease = Any

__all__ = [
    "ExecutionContext",
    "ToolExecutionError",
    "ToolExecutionOutcome",
    "ToolExecutor",
    "ToolStore",
    "uses_in_process_tool_settlement",
]


class ToolStore(Protocol):
    """The narrow persistence surface required by ``ToolExecutor``."""

    async def create_tool_call(
        self,
        call: ToolCall,
        *,
        workspace_id: str,
        idempotency_key: str,
    ) -> ToolCall:
        """Persist or replay one requested call."""

    async def await_tool_call(
        self,
        call_id: str,
        *,
        reason: str = "approval required",
        timestamp: datetime | None = None,
    ) -> ToolCall:
        """Record that a call is waiting for approval."""

    async def start_tool_call(
        self,
        call_id: str,
        *,
        approved: bool | None = None,
        started_at: datetime | None = None,
    ) -> ToolCall:
        """Record the transition into execution."""

    async def complete_tool_call(
        self,
        call_or_result: str | ToolResult,
        result: ToolResult | None = None,
    ) -> ToolResult:
        """Persist one terminal result and its event, idempotently."""

    async def reject_tool_call(
        self,
        call_id: str,
        *,
        reason: str,
        completed_at: datetime | None = None,
    ) -> ToolResult:
        """Persist a pre-execution rejection and its event, idempotently."""

    async def get_tool_result(self, call_id: str) -> ToolResult:
        """Read a previously persisted terminal result."""

    async def mark_tool_call_unknown(
        self,
        call_id: str,
        *,
        completed_at: datetime | None = None,
        message: str = "tool outcome is unconfirmed after restart",
    ) -> ToolResult:
        """Close an interrupted running call as UNKNOWN."""


class ToolExecutionError(RuntimeError):
    """A call cannot safely enter or repeat the execution lifecycle."""


class ExecutionContext(BaseModel):
    """The immutable, bounded input exposed to one handler invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)

    call: ToolCall
    arguments: BaseModel
    workspace_id: str
    resolved_paths: tuple[str, ...] = ()
    command_class: CommandClass | None = None


class ToolExecutionOutcome(DomainModel):
    """The auditable projection of one policy and handler attempt."""

    decision: PolicyDecision
    call: ToolCall
    result: ToolResult | None = None

    @property
    def completed(self) -> bool:
        return self.result is not None


def uses_in_process_tool_settlement(location: ExecutionLocation) -> bool:
    """Return whether this process settles the tool through registered handlers.

    ``LOCAL`` and ``CLOUD`` complete in-process. ``BRIDGE`` claim create, settle
    and submit remain on the Native Agent store path and are never invoked here.
    """

    return location is not ExecutionLocation.BRIDGE


class ToolExecutor:
    """Run registered handlers only after deterministic policy authorization."""

    def __init__(
        self,
        registry: ToolRegistry,
        store: ToolStore,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._registry = registry
        self._store = store
        self._timeout_seconds = timeout_seconds

    async def execute(
        self,
        call: ToolCall,
        mode: AgentMode,
        *,
        workspace_id: str | None = None,
        idempotency_key: str | None = None,
        lease: Lease | None = None,
        resolved_paths: tuple[str, ...] | None = None,
        command_class: CommandClass | None = None,
        isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
        execution_location: ExecutionLocation = ExecutionLocation.CLOUD,
        user_explicit_host_yolo: bool = False,
        settings: Settings | None = None,
        approved: bool | None = None,
        now: datetime | None = None,
    ) -> ToolExecutionOutcome:
        """Validate, authorize, persist, invoke, and complete one tool call."""
        from prp_runtime.policy.engine import PolicyOutcome, decide_tool_call

        decision = decide_tool_call(
            call,
            mode,
            known_tools=self._registry.names,
            lease=lease,
            workspace_id=workspace_id,
            resolved_paths=resolved_paths,
            command_class=command_class,
            isolation_mode=isolation_mode,
            execution_location=execution_location,
            user_explicit_host_yolo=user_explicit_host_yolo,
            settings=settings,
            now=now,
        )
        if workspace_id is None:
            raise ValueError("workspace_id is required for an executable tool call")

        persisted = await self._store.create_tool_call(
            call,
            workspace_id=workspace_id,
            idempotency_key=idempotency_key or call.call_id,
        )
        if persisted.status.is_terminal:
            result = await self._store.get_tool_result(persisted.call_id)
            return ToolExecutionOutcome(
                decision=decision,
                call=persisted,
                result=result,
            )

        if decision.outcome is PolicyOutcome.DENY:
            rejected = await self._store.reject_tool_call(
                persisted.call_id,
                reason=decision.reason_code.value,
            )
            return ToolExecutionOutcome(
                decision=decision,
                call=persisted.model_copy(update={"status": rejected.status}),
                result=rejected,
            )

        definition = self._registry[call.tool_name]
        if call.effect is not definition.effect:
            raise ToolExecutionError(
                f"tool call effect {call.effect.value} does not match registered effect "
                f"{definition.effect.value}"
            )
        arguments = definition.validate_arguments(call.arguments)

        if decision.outcome is PolicyOutcome.ASK and approved is not True:
            if persisted.status is ToolCallStatus.REQUESTED:
                persisted = await self._store.await_tool_call(
                    persisted.call_id,
                    reason=decision.reason_code.value,
                )
            return ToolExecutionOutcome(decision=decision, call=persisted)

        if (
            decision.outcome is PolicyOutcome.ASK
            and approved is True
            and persisted.status is ToolCallStatus.REQUESTED
        ):
            persisted = await self._store.await_tool_call(
                persisted.call_id,
                reason=decision.reason_code.value,
            )

        if persisted.status is ToolCallStatus.REQUESTED:
            persisted = await self._store.start_tool_call(
                persisted.call_id,
                approved=True if decision.outcome is PolicyOutcome.ASK else None,
            )
        elif persisted.status is ToolCallStatus.AWAITING_APPROVAL:
            if approved is not True:
                return ToolExecutionOutcome(decision=decision, call=persisted)
            persisted = await self._store.start_tool_call(
                persisted.call_id,
                approved=True,
            )
        elif persisted.status is not ToolCallStatus.RUNNING:
            raise ToolExecutionError(
                f"tool call cannot execute from {persisted.status.value} state"
            )

        if uses_in_process_tool_settlement(execution_location):
            return await self._invoke_registered_handler(
                persisted,
                definition=definition,
                arguments=arguments,
                workspace_id=workspace_id,
                resolved_paths=resolved_paths or (),
                command_class=command_class,
                decision=decision,
            )
        # BRIDGE preserves the same registered-handler completion contract.
        # Claim create/settle/submit stay outside this executor.
        return await self._invoke_registered_handler(
            persisted,
            definition=definition,
            arguments=arguments,
            workspace_id=workspace_id,
            resolved_paths=resolved_paths or (),
            command_class=command_class,
            decision=decision,
        )

    async def _invoke_registered_handler(
        self,
        persisted: ToolCall,
        *,
        definition: Any,
        arguments: Any,
        workspace_id: str,
        resolved_paths: tuple[str, ...],
        command_class: CommandClass | None,
        decision: PolicyDecision,
    ) -> ToolExecutionOutcome:
        context = ExecutionContext(
            call=persisted,
            arguments=arguments,
            workspace_id=workspace_id,
            resolved_paths=resolved_paths,
            command_class=command_class,
        )
        try:
            raw_result = await asyncio.wait_for(
                definition.handler(context), timeout=self._timeout_seconds
            )
            result = self._success_result(persisted, raw_result, definition.max_output_bytes)
        except TimeoutError:
            result = self._failure_result(
                persisted,
                ErrorCategory.TIMEOUT,
                "tool handler timed out",
            )
        except asyncio.CancelledError:
            await self._store.mark_tool_call_unknown(
                persisted.call_id,
                completed_at=utc_now(),
                message="tool outcome is unconfirmed after cancellation",
            )
            raise
        except Exception:
            result = self._failure_result(
                persisted,
                ErrorCategory.INTERNAL,
                "tool handler failed",
            )
        completed = await self._store.complete_tool_call(result)
        terminal_call = persisted.model_copy(update={"status": completed.status})
        return ToolExecutionOutcome(
            decision=decision,
            call=terminal_call,
            result=completed,
        )

    @staticmethod
    def _failure_result(
        call: ToolCall,
        category: ErrorCategory,
        message: str,
    ) -> ToolResult:
        return ToolResult.from_call(
            call,
            status=ToolCallStatus.FAILED,
            error=ErrorInfo(category=category, message=message),
            completed_at=utc_now(),
        )

    @classmethod
    def _success_result(
        cls,
        call: ToolCall,
        raw_result: Any,
        output_limit: int,
    ) -> ToolResult:
        if isinstance(raw_result, ToolResult):
            if raw_result.call_id != call.call_id:
                raise ValueError("handler result call_id does not match the tool call")
            output, was_truncated = cls._bound_text(raw_result.output, output_limit)
            return ToolResult(
                call_id=call.call_id,
                status=raw_result.status,
                result=None if was_truncated else raw_result.result,
                output=output,
                truncated=raw_result.truncated or was_truncated,
                changed_paths=raw_result.changed_paths,
                exit_code=raw_result.exit_code,
                error=raw_result.error,
                completed_at=raw_result.completed_at,
            )

        if raw_result is None:
            return ToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.SUCCEEDED,
                completed_at=utc_now(),
            )
        if isinstance(raw_result, str):
            payload: dict[str, Any] | None = None
            text = raw_result
        elif isinstance(raw_result, Mapping):
            payload = dict(raw_result)
            text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        elif isinstance(raw_result, BaseModel):
            payload = raw_result.model_dump(mode="json")
            text = json.dumps(payload, ensure_ascii=True, sort_keys=True)
        else:
            raise ValueError("tool handler returned an unsupported result type")
        bounded, truncated = cls._bound_text(text, output_limit)
        return ToolResult(
            call_id=call.call_id,
            status=ToolCallStatus.SUCCEEDED,
            result=None if truncated else payload,
            output=bounded,
            truncated=truncated,
            completed_at=utc_now(),
        )

    @staticmethod
    def _bound_text(value: str, limit: int) -> tuple[str, bool]:
        encoded = value.encode("utf-8")
        if len(encoded) <= limit:
            return value, False
        return encoded[:limit].decode("utf-8", errors="ignore"), True
