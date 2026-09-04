"""In-process LocalRuntime over the shared composition stack."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum, unique
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from prp_runtime.domain.enums import (
    AgentMode,
    ExecutionLocation,
    ExecutionStrategy,
    IsolationMode,
    ResourceAccess,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
)
from prp_runtime.domain.errors import BudgetError, DomainValidationError, ErrorCode
from prp_runtime.policy.models import ApprovalOutcome, ApprovalRequest
from prp_runtime.domain.models import (
    AgentRequestOptions,
    Budget,
    NativeRunRequest,
    Session,
    WorkspaceGrant,
)
from prp_runtime.domain.values import new_session_id, new_snapshot_id, utc_now
from prp_runtime.providers.base import ProviderAdapter, ProviderRequest, ProviderResponse
from prp_runtime.runtime.agent_executor import bind_local_approval_continuation
from prp_runtime.runtime.assembler import RunResult, assemble_run_result
from prp_runtime.runtime.composition import RuntimeComposition
from prp_runtime.settings import ProcessResourceEnvelope, Settings
from prp_runtime.storage.sqlite import MissingEntityError, SqliteStore
from prp_runtime.workspace.isolation import (
    ExecutionCopyMode,
    select_execution_copy_mode,
)
from prp_runtime.workspace.local import (
    LocalWorkspaceHandle,
    canonicalize_local_root,
    local_workspace_id,
    resolve_local_workspace,
)
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotStatus,
    Workspace,
    WorkspaceSource,
    WorkspaceSourceType,
)

__all__ = ["LocalRuntime", "LocalRuntimeState", "ProcessResourceLease"]

_ENVELOPE_FIELDS = (
    ("slots", "max_slots", 0, ErrorCode.RESOURCE_BUDGET_EXCEEDED),
    ("copied_bytes", "max_copied_bytes", 0, ErrorCode.RESOURCE_BUDGET_EXCEEDED),
    ("concurrency", "max_concurrency", 1, ErrorCode.RESOURCE_BUDGET_EXCEEDED),
    ("attempts", "max_attempts", 1, ErrorCode.ATTEMPT_BUDGET_EXCEEDED),
    ("tokens", "max_total_tokens", 0, ErrorCode.TOKEN_BUDGET_EXCEEDED),
)


@dataclass(frozen=True, slots=True)
class ProcessResourceLease:
    """One process-local envelope reservation."""

    lease_id: str
    slots: int
    copied_bytes: int
    concurrency: int
    attempts: int
    tokens: int


@unique
class LocalRuntimeState(StrEnum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"


class _ProcessEnvelopeAdapter:
    """Charge measured provider usage onto the in-flight local lease."""

    def __init__(self, inner: ProviderAdapter, runtime: "LocalRuntime") -> None:
        self._inner = inner
        self._runtime = runtime

    @property
    def name(self) -> str:
        return self._inner.name

    async def aclose(self) -> None:
        await self._inner.aclose()

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        response = await self._inner.complete(request)
        tokens = 0 if response.usage is None else response.usage.total_tokens
        self._runtime._accept_measured_tokens(tokens)
        return response


class LocalRuntime:
    """Open, recover and close the in-process runtime without HTTP or Bridge."""

    __slots__ = (
        "settings",
        "_composition",
        "_state",
        "_defaults",
        "_workspace_ids",
        "last_snapshot_id",
        "last_run_id",
        "last_workspace_id",
        "last_copy_mode",
        "_held",
        "_released_leases",
        "_run_leases",
        "_lease_seq",
        "_inflight_run_id",
        "_token_overflow",
    )

    def __init__(
        self,
        settings: Settings,
        *,
        adapters: Mapping[str, ProviderAdapter] | None = None,
        store: SqliteStore | None = None,
    ) -> None:
        self.settings = settings
        self._composition = RuntimeComposition(
            settings,
            adapters=adapters,
            store=store,
            execution_location=ExecutionLocation.LOCAL,
            isolation_mode=IsolationMode.HOST,
        )
        self._state = LocalRuntimeState.CLOSED
        self._workspace_ids: dict[str, str] = {}
        self.last_snapshot_id: str | None = None
        self.last_run_id: str | None = None
        self.last_workspace_id: str | None = None
        self.last_copy_mode: ExecutionCopyMode | None = None
        self._held: dict[str, ProcessResourceLease] = {}
        self._released_leases: set[str] = set()
        self._run_leases: dict[str, ProcessResourceLease] = {}
        self._lease_seq = 0
        self._inflight_run_id: str | None = None
        self._token_overflow: BudgetError | None = None
        self._defaults = AgentRequestOptions(
            isolation_mode=IsolationMode.HOST,
            execution_location=ExecutionLocation.LOCAL,
        )

    @property
    def state(self) -> LocalRuntimeState:
        return self._state

    @property
    def defaults(self) -> AgentRequestOptions:
        return self._defaults

    @property
    def composition(self) -> RuntimeComposition:
        self._require_open()
        return self._composition

    @property
    def store(self) -> SqliteStore:
        self._require_open()
        store = self._composition.store
        if store is None:
            raise RuntimeError("local runtime store is unavailable")
        return store

    @property
    def controller(self):
        self._require_open()
        controller = self._composition.controller
        if controller is None:
            raise RuntimeError("local runtime controller is unavailable")
        return controller

    @property
    def recovery(self):
        self._require_open()
        return self._composition.recovery

    @property
    def resource_envelope(self) -> ProcessResourceEnvelope:
        """Return the process-local envelope; model input cannot raise it."""
        return self.settings.resource_envelope

    def held_capacity(self) -> dict[str, int]:
        """Return currently reserved envelope occupancy."""
        held = {
            "max_slots": 0,
            "max_copied_bytes": 0,
            "max_concurrency": 0,
            "max_attempts": 0,
            "max_total_tokens": 0,
        }
        for lease in self._held.values():
            held["max_slots"] += lease.slots
            held["max_copied_bytes"] += lease.copied_bytes
            held["max_concurrency"] += lease.concurrency
            held["max_attempts"] += lease.attempts
            held["max_total_tokens"] += lease.tokens
        return held

    def admit(
        self,
        *,
        slots: int = 0,
        copied_bytes: int = 0,
        concurrency: int = 1,
        attempts: int = 1,
        tokens: int = 0,
    ) -> ProcessResourceLease:
        """Reserve envelope capacity or reject before work starts."""
        envelope = self.resource_envelope
        requested = {
            "slots": slots,
            "copied_bytes": copied_bytes,
            "concurrency": concurrency,
            "attempts": attempts,
            "tokens": tokens,
        }
        held = self.held_capacity()
        for attr, field, minimum, exceeded in _ENVELOPE_FIELDS:
            value = requested[attr]
            ceiling = getattr(envelope, field)
            if value < minimum:
                raise DomainValidationError(
                    f"{field} must be at least {minimum}",
                    code=ErrorCode.INVALID_BUDGET,
                    field=field,
                )
            if value > ceiling:
                raise DomainValidationError(
                    f"requested {field} exceeds the process envelope",
                    code=ErrorCode.INVALID_BUDGET,
                    field=field,
                )
            if held[field] + value > ceiling:
                raise BudgetError(
                    f"process {field} capacity exceeded",
                    code=exceeded,
                    field=field,
                )
        self._lease_seq += 1
        lease = ProcessResourceLease(
            lease_id=f"lease_{self._lease_seq}",
            slots=slots,
            copied_bytes=copied_bytes,
            concurrency=concurrency,
            attempts=attempts,
            tokens=tokens,
        )
        self._held[lease.lease_id] = lease
        return lease

    def release(self, lease: ProcessResourceLease | str | None) -> None:
        """Release one reservation exactly once."""
        if lease is None:
            return
        lease_id = lease.lease_id if isinstance(lease, ProcessResourceLease) else lease
        if lease_id in self._released_leases:
            return
        self._held.pop(lease_id, None)
        self._released_leases.add(lease_id)

    def public_facts(self) -> dict[str, Any]:
        """Return lifecycle facts without credentials or host paths."""
        facts: dict[str, Any] = {
            "state": self._state.value,
            "execution_location": self._defaults.execution_location.value,
            "isolation_mode": self._defaults.isolation_mode.value,
            "store_open": self._state is LocalRuntimeState.OPEN
            and self._composition.store is not None
            and self._composition.store.is_open,
            "controller_present": self._state is LocalRuntimeState.OPEN
            and self._composition.controller is not None,
        }
        facts.update(self.resource_envelope.public_facts())
        return facts

    async def open(self) -> LocalRuntime:
        """Open the shared composition, recover, and start the supervisor."""
        if self._state is LocalRuntimeState.OPEN:
            return self
        try:
            await self._composition.open()
        except BaseException:
            self._state = LocalRuntimeState.CLOSED
            raise
        self._state = LocalRuntimeState.OPEN
        self._hold_recovery_until_explicit_resume()
        self._install_envelope_adapters()
        return self

    def bind_workspace(self, root: Path | str) -> str:
        """Bind a local root to the in-process tool runtime without copying it."""
        self._require_open()
        root_path = canonicalize_local_root(root)
        principal_id = self.settings.service_principal
        workspace_id = local_workspace_id(owner_id=principal_id, root=root_path)
        self._workspace_ids[str(root_path)] = workspace_id
        self.last_workspace_id = workspace_id
        provider = self.composition.tool_runtime_provider
        if provider is not None:
            provider.bind_local_workspace(workspace_id, root_path)
        return workspace_id

    def _hold_recovery_until_explicit_resume(self) -> None:
        """Keep recovered local runs off the supervisor until bind+approve."""
        supervisor = self._composition.supervisor
        recovery = self._composition.recovery
        if supervisor is None or recovery is None:
            return
        supervisor.hold_runs(recovery.recoverable_run_ids)

    def _release_held_recovery_run(self, run_id: str) -> None:
        """Release one recovered run only after a local workspace is bound."""
        if not self._workspace_ids:
            raise RuntimeError("local workspace root is not bound")
        supervisor = self._composition.supervisor
        if supervisor is None:
            raise RuntimeError("local runtime is not open")
        supervisor.release_held_run(run_id)

    async def run(
        self,
        prompt: str,
        *,
        instructions: str | None = None,
        workspace: Path | str | None = None,
        agent_mode: AgentMode | None = None,
        isolation_mode: IsolationMode | None = None,
        user_explicit: bool | None = None,
        strategy: ExecutionStrategy = ExecutionStrategy.DIRECT,
        concurrency: int = 1,
        budget: Budget | None = None,
    ) -> RunResult:
        """Create and execute one DIRECT run through the shared controller."""
        self._require_open()
        self._reject_if_no_token_headroom()
        lease = self.admit(
            slots=0,
            copied_bytes=0,
            concurrency=concurrency,
            attempts=1,
            tokens=0,
        )
        run_id: str | None = None
        temporary: TemporaryDirectory[str] | None = None
        handle: LocalWorkspaceHandle | None = None
        try:
            root = Path(workspace) if workspace is not None else None
            if root is None:
                temporary = TemporaryDirectory()
                root = Path(temporary.name)
            principal_id = self.settings.service_principal
            handle = resolve_local_workspace(root, owner_id=principal_id)
            options = self._defaults
            updates: dict[str, object] = {}
            if agent_mode is not None:
                updates["agent_mode"] = agent_mode
            if isolation_mode is not None:
                updates["isolation_mode"] = isolation_mode
            if user_explicit is not None:
                updates["user_explicit"] = user_explicit
            if updates:
                options = options.model_copy(update=updates)
            copy_mode = select_execution_copy_mode(
                execution_location=options.execution_location,
                isolation_mode=options.isolation_mode,
                strategy=strategy,
                concurrency=concurrency,
            )
            self.last_copy_mode = copy_mode
            if copy_mode is not ExecutionCopyMode.IN_PLACE:
                raise DomainValidationError(
                    "local sequential runtime cannot execute copy-backed work",
                    code=ErrorCode.INVALID_AGENT_OPTIONS,
                    field="execution_copy_mode",
                )
            created_at = utc_now()
            workspace_id = self.bind_workspace(root)
            try:
                await self.store.get_workspace(workspace_id, owner_id=principal_id)
            except MissingEntityError:
                await self.store.create_workspace(
                    Workspace(
                        workspace_id=workspace_id,
                        owner_id=principal_id,
                        alias=workspace_id,
                        source=WorkspaceSource(
                            source_type=WorkspaceSourceType.SERVER_ALIAS,
                            server_alias="local-workspace",
                        ),
                        created_at=created_at,
                    )
                )
            manifest, file_contents = handle.backend.capture_snapshot()
            snapshot = await self.store.create_snapshot(
                Snapshot(
                    snapshot_id=new_snapshot_id(),
                    workspace_id=workspace_id,
                    status=SnapshotStatus.READY,
                    created_at=created_at,
                    completed_at=created_at,
                    file_count=len(manifest.entries),
                    total_size=manifest.total_size,
                ),
                manifest,
                owner_id=principal_id,
                file_contents=file_contents,
            )
            self.last_snapshot_id = snapshot.snapshot_id
            session = Session(
                session_id=new_session_id(),
                principal_id=principal_id,
                workspace_id=workspace_id,
                grant=WorkspaceGrant(
                    principal_id=principal_id,
                    workspace_id=workspace_id,
                    access=(ResourceAccess.READ, ResourceAccess.WRITE),
                ),
                agent_options=options,
                created_at=created_at,
            )
            await self.store.create_session(session)
            request = NativeRunRequest(
                input=prompt,
                instructions=instructions,
                routing_policy=RoutingPolicy.MANUAL,
                strategy=strategy,
                agent_options=options,
                budget=budget or Budget(),
            )
            created = await self.controller.create_run(request)
            run_id = created.run_id
            self._run_leases[run_id] = lease
            await self.store.attach_run_to_session(
                session.session_id,
                created.run_id,
                principal_id=principal_id,
            )
            self.last_run_id = created.run_id
            self.last_workspace_id = workspace_id
            supervisor = self.composition.supervisor
            bus = self.composition.event_bus
            if supervisor is None or bus is None:
                raise RuntimeError("local runtime is not open")
            subscription = await bus.subscribe(created.run_id)
            self._inflight_run_id = created.run_id
            self._token_overflow = None
            try:
                await supervisor.enqueue(created.run_id)
                result = await self._wait_for_run(
                    created.run_id,
                    subscription,
                    deadline=self._wait_deadline(
                        None if budget is None else budget.deadline
                    ),
                )
            finally:
                await subscription.close()
                self._inflight_run_id = None
            overflow = self._token_overflow
            self._token_overflow = None
            if overflow is not None:
                await self._settle_lease(run_id)
                raise overflow
            await self._account_usage(run_id, result)
            await self._settle_lease(run_id)
            return result
        except BaseException:
            self._inflight_run_id = None
            if run_id is None:
                self.release(lease)
            else:
                await self._settle_lease(run_id)
            raise
        finally:
            if handle is not None:
                handle.close()
            if temporary is not None:
                temporary.cleanup()

    async def pending_approvals(
        self,
        *,
        run_id: str | None = None,
        principal_id: str | None = None,
    ) -> tuple[ApprovalRequest, ...]:
        """Return durable ASK requests that still have no decision."""
        owner_id = principal_id or self.settings.service_principal
        scoped_run_id = run_id or self.last_run_id
        approvals = await self.store.list_approvals(
            owner_id=owner_id,
            run_id=scoped_run_id,
        )
        pending: list[ApprovalRequest] = []
        for approval in approvals:
            try:
                await self.store.get_approval_decision(
                    approval.request_id, owner_id=owner_id
                )
            except MissingEntityError:
                pending.append(approval)
        return tuple(pending)

    async def approve(
        self,
        request_id: str,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
        workspace_id: str | None = None,
        reason: str | None = None,
    ) -> RunResult:
        """Allow one owner-scoped ASK request and resume the same run."""
        return await self._continue_approval(
            request_id,
            outcome=ApprovalOutcome.ALLOW,
            principal_id=principal_id,
            run_id=run_id,
            workspace_id=workspace_id,
            reason=reason,
        )

    async def deny(
        self,
        request_id: str,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
        workspace_id: str | None = None,
        reason: str = "denied by operator",
    ) -> RunResult:
        """Reject one owner-scoped ASK request without applying the write."""
        return await self._continue_approval(
            request_id,
            outcome=ApprovalOutcome.DENY,
            principal_id=principal_id,
            run_id=run_id,
            workspace_id=workspace_id,
            reason=reason,
        )

    async def replay(
        self,
        request_id: str,
        *,
        principal_id: str | None = None,
        run_id: str | None = None,
        workspace_id: str | None = None,
    ) -> RunResult:
        """Replay an already-recorded decision without creating a new one."""
        owner_id = principal_id or self.settings.service_principal
        approval = await self.store.get_approval(request_id, owner_id=owner_id)
        if run_id is not None and approval.run_id != run_id:
            raise MissingEntityError(f"approval request {request_id} is not persisted")
        if workspace_id is not None and approval.workspace_id != workspace_id:
            raise MissingEntityError(f"approval request {request_id} is not persisted")
        await self.store.get_approval_decision(request_id, owner_id=owner_id)
        if approval.workspace_id not in self._workspace_ids.values():
            raise RuntimeError("local workspace root is not bound")
        return await self._resume_run(approval.run_id)

    async def _continue_approval(
        self,
        request_id: str,
        *,
        outcome: ApprovalOutcome,
        principal_id: str | None,
        run_id: str | None,
        workspace_id: str | None,
        reason: str | None,
    ) -> RunResult:
        self._require_open()
        owner_id = principal_id or self.settings.service_principal
        approval, _decision = await bind_local_approval_continuation(
            self.store,
            request_id,
            owner_id=owner_id,
            outcome=outcome,
            reason=reason,
            run_id=run_id,
            workspace_id=workspace_id,
        )
        if approval.workspace_id not in self._workspace_ids.values():
            raise RuntimeError("local workspace root is not bound")
        return await self._resume_run(approval.run_id)

    async def _resume_run(self, run_id: str) -> RunResult:
        supervisor = self.composition.supervisor
        bus = self.composition.event_bus
        if supervisor is None or bus is None:
            raise RuntimeError("local runtime is not open")
        self._release_held_recovery_run(run_id)
        subscription = await bus.subscribe(run_id)
        try:
            await supervisor.enqueue(run_id)
            result = await self._wait_for_run(run_id, subscription)
            await self._account_usage(run_id, result)
            await self._settle_lease(run_id)
            return result
        except BaseException:
            await self._settle_lease(run_id)
            raise
        finally:
            await subscription.close()

    def _wait_deadline(self, budget_deadline: datetime | None = None) -> datetime:
        """Return the nearer of the local wait ceiling and an explicit deadline."""
        ceiling = utc_now() + timedelta(seconds=float(self.settings.local_wait_seconds))
        if budget_deadline is None:
            return ceiling
        return min(budget_deadline, ceiling)

    async def _wait_for_run(
        self,
        run_id: str,
        subscription: Any,
        *,
        deadline: datetime | None = None,
    ) -> RunResult:
        limit = deadline or self._wait_deadline()
        while True:
            current = await self.store.get_run(run_id)
            if current.status.is_terminal or await self._paused_for_approval(run_id):
                break
            remaining = (limit - utc_now()).total_seconds()
            if remaining <= 0:
                current = await self.store.get_run(run_id)
                if current.status.is_terminal or await self._paused_for_approval(run_id):
                    break
                await self.cancel(run_id)
                self.release(self._run_leases.pop(run_id, None))
                raise BudgetError(
                    "local run exceeded the wait ceiling",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    field="deadline",
                )
            try:
                hint = await asyncio.wait_for(
                    subscription.get(), timeout=min(0.25, remaining)
                )
            except TimeoutError:
                continue
            if hint is None:
                current = await self.store.get_run(run_id)
                if current.status.is_terminal or await self._paused_for_approval(run_id):
                    break
                await self.cancel(run_id)
                self.release(self._run_leases.pop(run_id, None))
                raise BudgetError(
                    "local run exceeded the wait ceiling",
                    code=ErrorCode.DEADLINE_EXCEEDED,
                    field="deadline",
                )
        return await assemble_run_result(self.store, run_id)

    async def _paused_for_approval(self, run_id: str) -> bool:
        current = await self.store.get_run(run_id)
        if current.status is not RunStatus.RUNNING:
            return False
        pending_calls = await self.store.list_tool_calls(
            run_id,
            statuses=[ToolCallStatus.AWAITING_APPROVAL],
        )
        if not pending_calls:
            return False
        owner_id = self.settings.service_principal
        for approval in await self.store.list_approvals(owner_id=owner_id, run_id=run_id):
            try:
                await self.store.get_approval_decision(
                    approval.request_id, owner_id=owner_id
                )
            except MissingEntityError:
                return True
        return False

    async def cancel(self, run_id: str):
        """Cancel one run and release its envelope reservation if terminal."""
        self._require_open()
        run = await self.controller.cancel(run_id)
        await self._settle_lease(run_id)
        return run

    def _install_envelope_adapters(self) -> None:
        """Wrap outbound adapters so measured tokens are charged before success."""
        controller = self._composition.controller
        if controller is None:
            return
        wrapped: dict[str, ProviderAdapter] = {}
        for alias, adapter in controller._adapters.items():
            if isinstance(adapter, _ProcessEnvelopeAdapter):
                wrapped[alias] = adapter
            else:
                wrapped[alias] = _ProcessEnvelopeAdapter(adapter, self)
        controller._adapters = wrapped
        self._composition.adapters = wrapped

    def _accept_measured_tokens(self, tokens: int) -> None:
        """Charge one provider response onto the in-flight lease, or reject."""
        if tokens <= 0:
            return
        run_id = self._inflight_run_id
        if run_id is None:
            return
        lease = self._run_leases.get(run_id)
        if lease is None:
            return
        try:
            self._run_leases[run_id] = self._charge_tokens(
                lease, lease.tokens + tokens
            )
        except BudgetError as error:
            self._token_overflow = error
            raise

    def _reject_if_no_token_headroom(self) -> None:
        """Reject a new local run when the token envelope is already full."""
        envelope = self.resource_envelope
        if self.held_capacity()["max_total_tokens"] >= envelope.max_total_tokens:
            raise BudgetError(
                "process max_total_tokens capacity exceeded",
                code=ErrorCode.TOKEN_BUDGET_EXCEEDED,
                field="max_total_tokens",
            )

    def _charge_tokens(
        self, lease: ProcessResourceLease, tokens: int
    ) -> ProcessResourceLease:
        """Grow one held lease to measured provider usage, or reject overflow."""
        if lease.lease_id not in self._held or tokens <= lease.tokens:
            return lease
        extra = tokens - lease.tokens
        ceiling = self.resource_envelope.max_total_tokens
        if self.held_capacity()["max_total_tokens"] + extra > ceiling:
            raise BudgetError(
                "process max_total_tokens capacity exceeded",
                code=ErrorCode.TOKEN_BUDGET_EXCEEDED,
                field="max_total_tokens",
            )
        updated = replace(lease, tokens=tokens)
        self._held[lease.lease_id] = updated
        return updated

    async def _account_usage(self, run_id: str, result: RunResult) -> None:
        """Charge measured run usage onto the process envelope lease."""
        lease = self._run_leases.get(run_id)
        if lease is None:
            return
        self._run_leases[run_id] = self._charge_tokens(lease, result.usage.total_tokens)

    async def _settle_lease(self, run_id: str) -> None:
        lease = self._run_leases.get(run_id)
        if lease is None:
            return
        try:
            current = await self.store.get_run(run_id)
        except MissingEntityError:
            self.release(self._run_leases.pop(run_id, None))
            return
        if current.status.is_terminal:
            self.release(self._run_leases.pop(run_id, None))

    async def close(self) -> None:
        """Stop owned background work; repeated close is harmless."""
        if self._state is LocalRuntimeState.CLOSED and not self._composition.public_facts()["opened"]:
            return
        for lease in tuple(self._held.values()):
            self.release(lease)
        self._run_leases.clear()
        await self._composition.close()
        self._state = LocalRuntimeState.CLOSED

    def _require_open(self) -> None:
        if self._state is not LocalRuntimeState.OPEN:
            raise RuntimeError("local runtime is closed")

    async def __aenter__(self) -> LocalRuntime:
        return await self.open()

    async def __aexit__(self, *exc_info: object) -> None:
        await self.close()

    def __repr__(self) -> str:
        facts = self.public_facts()
        return (
            "LocalRuntime("
            f"state={facts['state']!r}, "
            f"execution_location={facts['execution_location']!r}, "
            f"isolation_mode={facts['isolation_mode']!r})"
        )
