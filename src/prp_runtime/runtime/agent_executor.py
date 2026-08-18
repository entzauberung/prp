"""Production adapter between public Agent tool calls and ToolWorker."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import Final, Protocol

from pydantic import BaseModel, ValidationError

from prp_runtime.domain.enums import ResourceAccess, ToolCallStatus, ToolEffect
from prp_runtime.domain.models import (
    MAX_AGENT_RESULT_BYTES,
    AgentToolCall,
    AgentToolResult,
    ExecutionScope,
)
from prp_runtime.domain.values import SnapshotId, utc_now
from prp_runtime.policy.engine import PolicyOutcome, PolicyReasonCode
from prp_runtime.policy.models import (
    ApprovalDecision,
    ApprovalOutcome,
    ApprovalRequest,
    CapabilityBudget,
    CapabilityScope,
    CommandClass,
    Lease,
    LeaseStatus,
)
from prp_runtime.runtime.agent_loop import (
    AgentToolContext,
    AgentToolExecution,
)
from prp_runtime.runtime.tool_worker import ToolWorker
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import MissingEntityError
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES, ToolCall, ToolResult
from prp_runtime.tools.registry import ToolRegistry

__all__ = [
    "AgentToolExecutor",
    "ProductionAgentToolExecutor",
    "WorkspaceAgentExecutor",
]

_REJECTION_OUTPUT: Final = "tool request rejected"
_FAILURE_OUTPUT: Final = "tool execution failed"
_PATH_ARGUMENT_KEYS: Final = frozenset({"path", "root", "paths", "target", "targets"})
_DEFAULT_COMMAND_CLASSES: Final = {"run_targeted_test": CommandClass.TEST}
_APPROVAL_MAX_WALL_CLOCK_MS: Final = 30_000
_APPROVAL_MAX_DURATION: Final = timedelta(minutes=5)


def _patch_paths(value: object) -> tuple[str, ...]:
    """Extract only validated unified-diff header paths for approval scope."""
    if not isinstance(value, str):
        raise ValueError("patch unified diff must be text")
    paths: list[str] = []
    lines = value.splitlines()
    for index, line in enumerate(lines[:-1]):
        if not line.startswith("--- ") or not lines[index + 1].startswith("+++ "):
            continue
        headers = (line[4:], lines[index + 1][4:])
        candidates: list[str | None] = []
        for header in headers:
            header_path = header.split("\t", 1)[0]
            if header_path == "/dev/null":
                candidates.append(None)
                continue
            if not header_path.startswith(("a/", "b/")):
                raise ValueError("patch path must use a relative diff prefix")
            candidate = header_path[2:]
            AgentToolExecutor._assert_relative_path(candidate)
            candidates.append(candidate)
        old_path, new_path = candidates
        if old_path is None and new_path is None:
            raise ValueError("patch must name a changed workspace path")
        if old_path is not None and new_path is not None and old_path != new_path:
            raise ValueError("patch renames are not supported")
        changed_path = new_path if new_path is not None else old_path
        assert changed_path is not None
        paths.append(changed_path)
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("patch must contain unique file paths")
    return tuple(paths)


class ApprovalStore(Protocol):
    """The owner-scoped persistence surface needed for ASK continuation."""

    async def create_approval(
        self, request: ApprovalRequest, *, owner_id: str
    ) -> ApprovalRequest:
        """Persist or replay one approval request."""

    async def get_approval(
        self, request_id: str, *, owner_id: str
    ) -> ApprovalRequest:
        """Read one owner-scoped approval request."""

    async def get_tool_call(self, call_id: str) -> ToolCall:
        """Read the persisted call to preserve its original timestamp."""

    async def get_approval_decision(
        self, request_id: str, *, owner_id: str
    ) -> ApprovalDecision:
        """Read the immutable approval decision."""

    async def create_lease(self, lease: Lease, *, owner_id: str) -> Lease:
        """Persist or replay one owner-scoped lease."""

    async def get_lease(self, lease_id: str, *, owner_id: str) -> Lease:
        """Read one owner-scoped lease."""

    async def reject_tool_call(
        self,
        call_id: str,
        *,
        reason: str,
        completed_at: datetime | None = None,
    ) -> ToolResult:
        """Close a pending call as a persisted rejection."""


class AgentToolExecutor:
    """Adapt one public Agent tool call to the policy-gated ToolWorker.

    Provider-owned fields are used only as a tool name and JSON arguments. The
    registry supplies the effect, while the authenticated execution scope and
    adapter snapshot supply all persistence and workspace identity.
    """

    def __init__(
        self,
        worker: ToolWorker,
        registry: ToolRegistry,
        scope: ExecutionScope,
        *,
        snapshot_id: SnapshotId,
        approved: bool | None = None,
        command_classes: Mapping[str, CommandClass] | None = None,
        settings: Settings | None = None,
        approval_store: ApprovalStore | None = None,
    ) -> None:
        self._worker = worker
        self._registry = registry
        self._scope = scope
        self._snapshot_id = snapshot_id
        classes = dict(_DEFAULT_COMMAND_CLASSES)
        classes.update(command_classes or {})
        if not all(isinstance(value, CommandClass) for value in classes.values()):
            raise ValueError("command_classes must contain CommandClass values")
        self._command_classes = MappingProxyType(classes)
        self._approved = approved
        self._settings = settings
        self._approval_store = approval_store

    @property
    def scope(self) -> ExecutionScope:
        """Return the immutable authenticated scope used by this adapter."""
        return self._scope

    @property
    def snapshot_id(self) -> SnapshotId:
        """Return the base snapshot bound to newly mapped calls."""
        return self._snapshot_id

    @property
    def approval_store(self) -> ApprovalStore | None:
        """Return the explicitly injected approval store, if configured."""
        return self._approval_store

    def build_approval_request(
        self,
        call: ToolCall,
        *,
        reason: PolicyReasonCode,
        resolved_paths: tuple[str, ...],
        command_class: CommandClass | None = None,
        requested_at: datetime | None = None,
    ) -> ApprovalRequest:
        """Build one bounded approval request from server-resolved facts."""
        if call.run_id != self._scope.run_id:
            raise ValueError("approval call does not belong to the execution scope")
        if reason is not PolicyReasonCode.APPROVAL_REQUIRED:
            raise ValueError("approval request requires the approval policy reason")
        if not resolved_paths:
            raise ValueError("approval request requires resolved paths")
        paths = tuple(dict.fromkeys(resolved_paths))
        for path in paths:
            self._assert_relative_path(path)
        required_access = (
            ResourceAccess.READ
            if call.effect is ToolEffect.READ
            else ResourceAccess.WRITE
        )
        if required_access not in self._scope.grant.access:
            raise ValueError("approval request exceeds the execution grant")
        if call.effect is ToolEffect.NETWORK:
            raise ValueError("network approval is not supported")
        if call.effect is ToolEffect.COMMAND:
            if command_class is None:
                raise ValueError("command approval requires a command class")
            command_classes: tuple[CommandClass, ...] = (command_class,)
        elif command_class is not None:
            raise ValueError("command class requires a command effect")
        else:
            command_classes = ()

        issued_at = requested_at or utc_now()
        grant_expiry = self._scope.grant.expires_at
        if grant_expiry is not None and grant_expiry <= issued_at:
            raise ValueError("approval request grant is expired")
        expires_at = min(
            grant_expiry or (issued_at + _APPROVAL_MAX_DURATION),
            issued_at + _APPROVAL_MAX_DURATION,
        )
        capability = CapabilityScope(
            tools=(call.tool_name,),
            effects=(call.effect,),
            workspace_id=self._scope.workspace_id,
            paths=paths,
            command_classes=command_classes,
            budget=CapabilityBudget(
                max_calls=1,
                max_output_bytes=MAX_TOOL_OUTPUT_BYTES,
                max_wall_clock_ms=_APPROVAL_MAX_WALL_CLOCK_MS,
            ),
            expires_at=expires_at,
        )
        return ApprovalRequest.from_tool_call(
            call,
            workspace_id=self._scope.workspace_id,
            scope=capability,
            reason=reason.value,
            requested_at=issued_at,
        )

    async def _existing_approval(
        self,
        call: ToolCall,
        *,
        resolved_paths: tuple[str, ...],
        command_class: CommandClass | None,
    ) -> ApprovalRequest | None:
        store = self._approval_store
        if store is None:
            return None
        approval_call = call
        try:
            approval_call = await store.get_tool_call(call.call_id)
        except MissingEntityError:
            pass
        request = self.build_approval_request(
            approval_call,
            reason=PolicyReasonCode.APPROVAL_REQUIRED,
            resolved_paths=resolved_paths,
            command_class=command_class,
            requested_at=approval_call.requested_at,
        )
        try:
            return await store.get_approval(
                request.request_id,
                owner_id=self._scope.principal_id,
            )
        except MissingEntityError:
            return None

    @staticmethod
    def _lease_id(request: ApprovalRequest) -> str:
        material = request.request_id + "\x1f" + request.call_id
        return "lease_" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    async def _ensure_lease(self, request: ApprovalRequest) -> Lease:
        store = self._approval_store
        if store is None:
            raise RuntimeError("approval store is unavailable")
        now = utc_now()
        lease_id = self._lease_id(request)
        existing: Lease | None = None
        try:
            existing = await store.get_lease(
                lease_id,
                owner_id=self._scope.principal_id,
            )
        except MissingEntityError:
            pass
        if existing is not None:
            if existing.status is not LeaseStatus.ACTIVE or not existing.is_active_at(now):
                raise RuntimeError("approval lease is expired or revoked")
            return existing
        if request.scope.expires_at <= now:
            raise RuntimeError("approval request is expired")
        return await store.create_lease(
            Lease(
                lease_id=lease_id,
                approval_request_id=request.request_id,
                call_id=request.call_id,
                scope=request.scope,
                issuer=request.issuer,
                issued_at=now,
                expires_at=request.scope.expires_at,
            ),
            owner_id=self._scope.principal_id,
        )

    async def _approval_decision(
        self, request: ApprovalRequest
    ) -> ApprovalDecision | None:
        store = self._approval_store
        if store is None:
            return None
        try:
            return await store.get_approval_decision(
                request.request_id,
                owner_id=self._scope.principal_id,
            )
        except MissingEntityError:
            return None

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        """Map, authorize and execute a public call without trusting its metadata."""
        options = self._scope.agent_options
        if context.run_id != self._scope.run_id or context.mode is not options.agent_mode:
            return self._rejected(call, "scope_mismatch")

        try:
            definition = self._registry.get(call.tool_name)
            # Validate against the server-owned argument model before creating
            # a persistent call. ToolExecutor repeats this check at execution.
            validated_arguments = definition.validate_arguments(call.arguments)
            resolved_paths = self._resolved_paths(validated_arguments)
            command_class = self._command_classes.get(call.tool_name)
            if definition.effect is ToolEffect.COMMAND and command_class is None:
                return self._rejected(call, "unknown_command_class")
            if not self._grant_allows(definition.effect):
                return self._rejected(call, "grant_denied")
            persisted_call = ToolCall(
                call_id=self._internal_call_id(
                    run_id=self._scope.run_id,
                    work_unit_id=context.work_unit_id,
                    snapshot_id=self._snapshot_id,
                    provider_call_id=call.call_id,
                    tool_name=call.tool_name,
                ),
                run_id=self._scope.run_id,
                work_unit_id=context.work_unit_id,
                tool_name=call.tool_name,
                effect=definition.effect,
                arguments=dict(call.arguments),
                snapshot_id=self._snapshot_id,
                requested_at=utc_now(),
            )
        except (KeyError, ValidationError, ValueError):
            return self._rejected(call, "invalid_tool_request")

        approved_for_worker = None
        existing_approval: ApprovalRequest | None = None
        approval_lease_failed = False
        if self._approval_store is not None:
            try:
                existing_approval = await self._existing_approval(
                    persisted_call,
                    resolved_paths=resolved_paths,
                    command_class=command_class,
                )
                if existing_approval is not None:
                    decision = await self._approval_decision(existing_approval)
                    if (
                        decision is not None
                        and decision.outcome is ApprovalOutcome.ALLOW
                        and self._approved is not False
                    ):
                        await self._ensure_lease(existing_approval)
                        approved_for_worker = True
            except Exception:
                # The worker still records the pending call; the ASK branch below
                # closes it as a safe rejection if the approval facts are unusable.
                approved_for_worker = None
                approval_lease_failed = (
                    existing_approval is not None and self._approved is not False
                )

        try:
            outcome = await self._worker.execute(
                persisted_call,
                options.agent_mode,
                workspace_id=self._scope.workspace_id,
                idempotency_key=persisted_call.call_id,
                resolved_paths=resolved_paths,
                command_class=command_class,
                isolation_mode=options.isolation_mode,
                execution_location=options.execution_location,
                user_explicit_host_yolo=options.user_explicit,
                settings=self._settings,
                approved=approved_for_worker,
            )
        except Exception:
            return self._failed(call)
        if outcome.result is None:
            if outcome.decision.outcome is PolicyOutcome.ASK:
                store = self._approval_store
                if store is None:
                    return AgentToolExecution(
                        call=call,
                        awaiting_approval=True,
                        reason=outcome.decision.reason_code.value,
                    )
                try:
                    approval = existing_approval or self.build_approval_request(
                        outcome.call,
                        reason=outcome.decision.reason_code,
                        resolved_paths=resolved_paths,
                        command_class=command_class,
                        requested_at=outcome.call.requested_at,
                    )
                    if existing_approval is None:
                        approval = await store.create_approval(
                            approval,
                            owner_id=self._scope.principal_id,
                        )
                    decision = await self._approval_decision(approval)
                except Exception:
                    rejected = await store.reject_tool_call(
                        outcome.call.call_id,
                        reason="approval_unavailable",
                    )
                    return AgentToolExecution(
                        call=call,
                        result=self._public_result(call, rejected),
                    )
                if decision is not None and decision.outcome is ApprovalOutcome.DENY:
                    rejected = await store.reject_tool_call(
                        outcome.call.call_id,
                        reason="approval_denied",
                    )
                    return AgentToolExecution(
                        call=call,
                        result=self._public_result(call, rejected),
                    )
                if (
                    decision is not None
                    and decision.outcome is ApprovalOutcome.ALLOW
                    and approval_lease_failed
                ):
                    rejected = await store.reject_tool_call(
                        outcome.call.call_id,
                        reason="lease_inactive",
                    )
                    return AgentToolExecution(
                        call=call,
                        result=self._public_result(call, rejected),
                    )
                return AgentToolExecution(
                    call=call,
                    awaiting_approval=True,
                    reason=outcome.decision.reason_code.value,
                )
            return self._rejected(call, "policy_denied")

        return AgentToolExecution(
            call=call,
            result=self._public_result(call, outcome.result),
        )

    @staticmethod
    def _internal_call_id(
        *,
        run_id: str,
        work_unit_id: str,
        snapshot_id: str,
        provider_call_id: str,
        tool_name: str,
    ) -> str:
        """Create a stable domain id without exposing or trusting provider ids."""
        material = "\x1f".join(
            (run_id, work_unit_id, snapshot_id, provider_call_id, tool_name)
        )
        return f"tc_{hashlib.sha256(material.encode('utf-8')).hexdigest()}"

    @staticmethod
    def public_result(call: AgentToolCall, result: ToolResult) -> AgentToolResult:
        """Project one persisted result back to the provider call identity."""
        output = result.output
        truncated = result.truncated
        encoded_output = output.encode("utf-8")
        if len(encoded_output) > MAX_AGENT_RESULT_BYTES:
            output = encoded_output[:MAX_AGENT_RESULT_BYTES].decode("utf-8", errors="ignore")
            truncated = True
        payload = result.result
        if payload is not None:
            try:
                encoded_payload = json.dumps(
                    payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            except (TypeError, ValueError):
                payload = None
                truncated = True
            else:
                if len(encoded_payload) > MAX_AGENT_RESULT_BYTES:
                    payload = None
                    truncated = True
        return AgentToolResult(
            call_id=call.call_id,
            status=result.status,
            result=payload,
            output=output,
            truncated=truncated,
        )

    @staticmethod
    def _public_result(call: AgentToolCall, result: ToolResult) -> AgentToolResult:
        """Backward-compatible private alias for the bounded projection."""
        return AgentToolExecutor.public_result(call, result)

    def _grant_allows(self, effect: ToolEffect) -> bool:
        expires_at = self._scope.grant.expires_at
        if expires_at is not None and expires_at <= utc_now():
            return False
        required = ResourceAccess.READ if effect is ToolEffect.READ else ResourceAccess.WRITE
        return required in self._scope.grant.access

    @classmethod
    def _resolved_paths(cls, arguments: BaseModel) -> tuple[str, ...]:
        paths: list[str] = []

        def visit(value: object, key: str | None = None) -> None:
            if key == "unified_diff":
                paths.extend(_patch_paths(value))
                return
            if key in _PATH_ARGUMENT_KEYS:
                values: tuple[object, ...]
                if isinstance(value, str):
                    values = (value,)
                elif isinstance(value, (list, tuple)):
                    values = tuple(value)
                else:
                    raise ValueError("path argument must be a string or string list")
                for candidate in values:
                    if not isinstance(candidate, str):
                        raise ValueError("path argument must contain only strings")
                    if candidate:
                        cls._assert_relative_path(candidate)
                        paths.append(candidate)
                return
            if isinstance(value, Mapping):
                for nested_key, nested_value in value.items():
                    visit(nested_value, str(nested_key))
            elif isinstance(value, (list, tuple)):
                for nested_value in value:
                    visit(nested_value)

        visit(arguments.model_dump(mode="python"))
        return tuple(dict.fromkeys(paths))

    @staticmethod
    def _assert_relative_path(path: str) -> None:
        if (
            path.startswith(("/", "\\"))
            or (len(path) >= 2 and path[1] == ":")
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            raise ValueError("resolved path is outside the workspace")

    @staticmethod
    def _failed(call: AgentToolCall) -> AgentToolExecution:
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.FAILED,
                result={"error": "internal_tool_failure"},
                output=_FAILURE_OUTPUT,
            ),
        )

    @staticmethod
    def _rejected(call: AgentToolCall, reason: str) -> AgentToolExecution:
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.REJECTED,
                result={"error": reason},
                output=_REJECTION_OUTPUT,
            ),
        )


ProductionAgentToolExecutor = AgentToolExecutor
WorkspaceAgentExecutor = AgentToolExecutor
