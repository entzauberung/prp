"""Workspace-scoped composition contracts for production tools."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Collection
from enum import StrEnum, unique
from pathlib import Path

from prp_runtime.domain.enums import (
    ExecutionLocation,
    ExecutionStrategy,
    ResourceAccess,
    ToolCallStatus,
    ToolEffect,
)
from prp_runtime.domain.models import (
    AgentToolCall,
    AgentToolResult,
    BridgeClientStatus,
    ClientCapabilityDescriptor,
    ExecutionScope,
    ProviderToolDescriptor,
    RegisteredBridgeClient,
)
from prp_runtime.domain.values import SnapshotId, utc_now
from prp_runtime.policy.models import CommandClass
from prp_runtime.runtime.agent_executor import AgentToolExecutor
from prp_runtime.runtime.agent_loop import AgentToolContext, AgentToolExecution
from prp_runtime.runtime.tool_worker import ToolWorker
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.tools.command import (
    DEFAULT_COMMAND_REGISTRY,
    CommandRegistry,
    CommandRunner,
    build_targeted_test_definition,
)
from prp_runtime.tools.diff import (
    DiffManifestMismatchError,
    DiffResult,
    DiffToolRunner,
    build_diff_definitions,
)
from prp_runtime.policy.engine import PolicyOutcome
from prp_runtime.tools.executor import RemoteToolAssignmentPending, ToolExecutor
from prp_runtime.tools.filesystem import build_filesystem_registry
from prp_runtime.tools.models import ToolCall
from prp_runtime.tools.patch import (
    PatchApplyResult,
    PatchRequest,
    PatchRunner,
    build_patch_definition,
)
from prp_runtime.tools.registry import ToolDefinition, ToolRegistry
from prp_runtime.tools.search import SearchRunner, build_search_definition
from prp_runtime.workspace.backend import WorkspaceBackend
from prp_runtime.workspace.changes import ChangeSet
from prp_runtime.workspace.local import canonicalize_local_root
from prp_runtime.workspace.isolation import (
    ExecutionCopyMode,
    SlotContext,
    select_execution_copy_mode,
)
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotManifest,
    SnapshotStatus,
    WorkspaceSourceType,
)
from prp_runtime.workspace.resolver import (
    ResolvedWorkspace,
    WorkspaceResolveError,
    WorkspaceResolver,
)
from prp_runtime.workspace.sandbox import SandboxBackend, SandboxUnavailableError

__all__ = [
    "BridgeRemoteToolExecutor",
    "ToolRuntimeError",
    "ToolRuntimeState",
    "ScopedAgentToolExecutor",
    "ScopeToolRuntimeProvider",
    "WorkspaceToolRuntime",
    "WorkspaceToolRuntimeFactory",
    "catalog_from_bridge_capabilities",
]


@unique
class ToolRuntimeState(StrEnum):
    """Lifecycle state of one workspace-bound tool composition."""

    OPEN = "OPEN"
    CLOSED = "CLOSED"


class ToolRuntimeError(ValueError):
    """A workspace tool runtime cannot be safely composed or used."""


class _DeferredDiffRunner(DiffToolRunner):
    """Expose a stable tool catalog, then bind the approved patch result."""

    def __init__(
        self,
        base_manifest: SnapshotManifest,
        manifest_provider: Callable[[], SnapshotManifest],
    ) -> None:
        self._base_manifest = base_manifest
        self._manifest_provider = manifest_provider
        self._bound_change_set: ChangeSet | None = None
        self._runner: DiffToolRunner | None = None

    def bind(self, change_set: ChangeSet) -> None:
        if self._bound_change_set is not None:
            if self._bound_change_set != change_set:
                raise ToolRuntimeError("diff runtime is already bound to another ChangeSet")
            return
        self._bound_change_set = change_set
        self._runner = DiffToolRunner(
            change_set,
            base_manifest=self._base_manifest,
            manifest_provider=self._manifest_provider,
        )

    def get_diff(self) -> DiffResult:
        if self._runner is None:
            raise DiffManifestMismatchError("diff is unavailable before a successful patch")
        return self._runner.get_diff()

    def get_status(self) -> DiffResult:
        if self._runner is None:
            raise DiffManifestMismatchError("status is unavailable before a successful patch")
        return self._runner.get_status()


class _DiffBindingPatchRunner(PatchRunner):
    """Bind the persisted ChangeSet only after PatchRunner commits atomically."""

    def __init__(
        self,
        runner: PatchRunner,
        store: SqliteStore,
        diff_runner: _DeferredDiffRunner,
    ) -> None:
        self._runner = runner
        self._store = store
        self._diff_runner = diff_runner

    async def apply(self, call: ToolCall, request: PatchRequest) -> PatchApplyResult:
        result = await self._runner.apply(call, request)
        self._diff_runner.bind(await self._store.get_change_set(result.change_set_id))
        return result


class WorkspaceToolRuntime:
    """Own one registry/executor pair for one resolved workspace snapshot."""

    __slots__ = (
        "_scope",
        "_workspace_id",
        "_server_alias",
        "_snapshot_id",
        "_registry",
        "_catalog",
        "_executor",
        "_close_workspace",
        "_state",
    )

    def __init__(
        self,
        *,
        scope: ExecutionScope,
        resolved_workspace: ResolvedWorkspace,
        snapshot: Snapshot,
        registry: ToolRegistry,
        executor: AgentToolExecutor,
        close_workspace: Callable[[], None] | None = None,
    ) -> None:
        if scope.workspace_id != resolved_workspace.workspace_id:
            raise ToolRuntimeError("resolved workspace does not match execution scope")
        if scope.principal_id != resolved_workspace.owner_id:
            raise ToolRuntimeError("resolved workspace owner does not match execution scope")
        if snapshot.workspace_id != scope.workspace_id:
            raise ToolRuntimeError("snapshot does not belong to execution scope workspace")
        if snapshot.status is not SnapshotStatus.READY:
            raise ToolRuntimeError("tool runtime requires a ready snapshot")
        if resolved_workspace.server_alias.strip() == "":
            raise ToolRuntimeError("resolved workspace alias is missing")
        self._scope = scope
        self._workspace_id = resolved_workspace.workspace_id
        self._server_alias = resolved_workspace.server_alias
        self._snapshot_id = snapshot.snapshot_id
        self._registry = registry
        self._catalog = registry.provider_catalog
        self._executor = executor
        self._close_workspace = close_workspace or resolved_workspace.close
        self._state = ToolRuntimeState.OPEN

    @property
    def scope(self) -> ExecutionScope:
        """Return the immutable scope without a host root."""
        self._require_open()
        return self._scope

    @property
    def workspace_id(self) -> str:
        """Return the logical workspace identity."""
        return self._workspace_id

    @property
    def server_alias(self) -> str:
        """Return the logical server alias, never its configured root."""
        return self._server_alias

    @property
    def snapshot_id(self) -> SnapshotId:
        """Return the immutable base snapshot identity."""
        return self._snapshot_id

    @property
    def registry(self) -> ToolRegistry:
        """Return the workspace-bound registry while the handle is open."""
        self._require_open()
        return self._registry

    @property
    def catalog(self) -> tuple[ProviderToolDescriptor, ...]:
        """Return the public, non-executable provider catalog."""
        return self._catalog

    @property
    def executor(self) -> AgentToolExecutor:
        """Return the production Agent adapter while the handle is open."""
        self._require_open()
        return self._executor

    @property
    def state(self) -> ToolRuntimeState:
        """Return the idempotent lifecycle state."""
        return self._state

    def close(self) -> None:
        """Close the resolved workspace exactly once."""
        if self._state is ToolRuntimeState.CLOSED:
            return
        try:
            self._close_workspace()
        finally:
            self._state = ToolRuntimeState.CLOSED

    def _require_open(self) -> None:
        if self._state is ToolRuntimeState.CLOSED:
            raise ToolRuntimeError("tool runtime is closed")

    def __enter__(self) -> WorkspaceToolRuntime:
        self._require_open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return (
            "WorkspaceToolRuntime("
            f"workspace_id={self._workspace_id!r}, "
            f"server_alias={self._server_alias!r}, "
            f"snapshot_id={self._snapshot_id!r}, state={self._state.value!r})"
        )


class WorkspaceToolRuntimeFactory:
    """Build non-cached workspace tool runtimes from server-owned handles."""

    def build(
        self,
        *,
        scope: ExecutionScope,
        resolved_workspace: ResolvedWorkspace,
        snapshot: Snapshot,
        registry: ToolRegistry | None = None,
        executor: AgentToolExecutor | None = None,
        source_type: WorkspaceSourceType = WorkspaceSourceType.SERVER_ALIAS,
        store: SqliteStore | None = None,
        snapshot_manifest: SnapshotManifest | None = None,
        command_registry: CommandRegistry = DEFAULT_COMMAND_REGISTRY,
        rg_path: Path | None = None,
        sandbox_backend: SandboxBackend | None = None,
        runtime_roots: Collection[Path] | None = None,
        change_set: ChangeSet | None = None,
        manifest_provider: Callable[[], SnapshotManifest] | None = None,
        approved: bool | None = None,
        settings: Settings | None = None,
        slot_context: SlotContext | None = None,
    ) -> WorkspaceToolRuntime:
        """Compose one runtime or bind all server-owned tool builders."""
        if scope.agent_options.execution_location is ExecutionLocation.BRIDGE:
            raise ToolRuntimeError("BRIDGE does not create a server workspace tool runtime")
        if source_type is not WorkspaceSourceType.SERVER_ALIAS:
            raise WorkspaceResolveError("bridge workspace is not available to the cloud")
        if registry is not None or executor is not None:
            if slot_context is not None:
                raise ToolRuntimeError(
                    "slot context requires server-owned tool composition"
                )
            if registry is None or executor is None:
                raise ToolRuntimeError("registry and executor must be supplied together")
            return WorkspaceToolRuntime(
                scope=scope,
                resolved_workspace=resolved_workspace,
                snapshot=snapshot,
                registry=registry,
                executor=executor,
            )
        if store is None:
            raise ToolRuntimeError("tool runtime store is required")
        return self._build_cloud(
            scope=scope,
            resolved_workspace=resolved_workspace,
            snapshot=snapshot,
            store=store,
            snapshot_manifest=snapshot_manifest,
            command_registry=command_registry,
            rg_path=rg_path,
            sandbox_backend=sandbox_backend,
            runtime_roots=runtime_roots,
            change_set=change_set,
            manifest_provider=manifest_provider,
            approved=approved,
            settings=settings,
            slot_context=slot_context,
        )

    def _build_cloud(
        self,
        *,
        scope: ExecutionScope,
        resolved_workspace: ResolvedWorkspace,
        snapshot: Snapshot,
        store: SqliteStore,
        snapshot_manifest: SnapshotManifest | None,
        command_registry: CommandRegistry,
        rg_path: Path | None,
        sandbox_backend: SandboxBackend | None,
        runtime_roots: Collection[Path] | None,
        change_set: ChangeSet | None,
        manifest_provider: Callable[[], SnapshotManifest] | None,
        approved: bool | None,
        settings: Settings | None,
        slot_context: SlotContext | None,
    ) -> WorkspaceToolRuntime:
        """Bind each handler to this non-cached backend and immutable snapshot."""
        copy_mode = select_execution_copy_mode(
            execution_location=scope.agent_options.execution_location,
            isolation_mode=scope.agent_options.isolation_mode,
            strategy=ExecutionStrategy.DIRECT,
            concurrency=1,
        )
        if copy_mode is ExecutionCopyMode.IN_PLACE and slot_context is not None:
            raise ToolRuntimeError("in-place local DIRECT cannot use isolation slots")
        slot_backend: WorkspaceBackend | None = None
        slot_acquired = False
        try:
            if slot_context is None:
                backend = resolved_workspace.backend
                workspace_cwd = resolved_workspace._tool_workspace_cwd()
            else:
                slot_context.acquire()
                slot_acquired = True
                slot_backend = WorkspaceBackend(slot_context.path)
                backend = slot_backend
                workspace_cwd = slot_context.path
            base_manifest = snapshot_manifest or resolved_workspace.snapshot_manifest()
            definitions: list[ToolDefinition] = []
            if ResourceAccess.READ in scope.grant.access:
                definitions.extend(build_filesystem_registry(backend).definitions)
                definitions.append(
                    build_search_definition(
                        SearchRunner(backend, workspace_cwd=workspace_cwd, rg_path=rg_path)
                    )
                )

            deferred_diff_runner: _DeferredDiffRunner | None = None
            if ResourceAccess.READ in scope.grant.access:
                current_manifest = manifest_provider or (
                    resolved_workspace.snapshot_manifest
                    if slot_context is None
                    else backend.snapshot_manifest
                )
                if change_set is not None:
                    if (
                        change_set.workspace_id != scope.workspace_id
                        or change_set.base_snapshot_id != snapshot.snapshot_id
                    ):
                        raise ToolRuntimeError("ChangeSet does not match tool runtime snapshot")
                    current_manifest = manifest_provider or resolved_workspace.snapshot_manifest
                    definitions.extend(
                        build_diff_definitions(
                            DiffToolRunner(
                                change_set,
                                base_manifest=base_manifest,
                                manifest_provider=current_manifest,
                            )
                        )
                    )
                elif ResourceAccess.WRITE in scope.grant.access:
                    deferred_diff_runner = _DeferredDiffRunner(
                        base_manifest,
                        current_manifest,
                    )
                    definitions.extend(build_diff_definitions(deferred_diff_runner))

            if ResourceAccess.WRITE in scope.grant.access:
                patch_runner = PatchRunner(
                    backend,
                    store,
                    owner_id=scope.principal_id,
                    base_snapshot=snapshot,
                    base_manifest=base_manifest,
                )
                if deferred_diff_runner is not None:
                    patch_runner = _DiffBindingPatchRunner(
                        patch_runner,
                        store,
                        deferred_diff_runner,
                    )
                definitions.append(build_patch_definition(patch_runner))
                try:
                    command_runner = CommandRunner(
                        command_registry,
                        workspace_cwd=workspace_cwd,
                        test_only=True,
                        isolation_mode=scope.agent_options.isolation_mode,
                        sandbox_backend=sandbox_backend,
                        runtime_roots=runtime_roots,
                    )
                except SandboxUnavailableError:
                    # A missing L3 gate must not block non-command tools such as
                    # patch approval. The command capability remains absent and
                    # therefore cannot spawn on the host by accident.
                    command_runner = None
                if command_runner is not None:
                    definitions.append(build_targeted_test_definition(command_runner))

            registry = ToolRegistry(definitions)
            tool_executor = ToolExecutor(registry, store)
            agent_executor = AgentToolExecutor(
                ToolWorker(tool_executor),
                registry,
                scope,
                snapshot_id=snapshot.snapshot_id,
                approved=approved,
                command_classes={"run_targeted_test": CommandClass.TEST},
                settings=settings,
                approval_store=store,
            )
            return WorkspaceToolRuntime(
                scope=scope,
                resolved_workspace=resolved_workspace,
                snapshot=snapshot,
                registry=registry,
                executor=agent_executor,
                close_workspace=(
                    self._close_slot_runtime(
                        resolved_workspace,
                        slot_backend,
                        slot_context,
                    )
                    if slot_context is not None
                    else None
                ),
            )
        except BaseException:
            try:
                if slot_backend is not None:
                    slot_backend.close()
            finally:
                try:
                    if slot_acquired and slot_context is not None:
                        slot_context.cleanup()
                finally:
                    resolved_workspace.close()
            raise

    @staticmethod
    def _close_slot_runtime(
        resolved_workspace: ResolvedWorkspace,
        slot_backend: WorkspaceBackend | None,
        slot_context: SlotContext | None,
    ) -> Callable[[], None]:
        """Close slot tools and their private slot before the source handle."""
        closed = False

        def close() -> None:
            nonlocal closed
            if closed:
                return
            closed = True
            try:
                if slot_backend is not None:
                    slot_backend.close()
            finally:
                try:
                    if slot_context is not None:
                        slot_context.cleanup()
                finally:
                    resolved_workspace.close()

        return close


class ScopedAgentToolExecutor:
    """Lazy Agent executor that is permanently bound to one run scope."""

    def __init__(
        self,
        scope: ExecutionScope,
        runtime_provider: Callable[[ExecutionScope], Awaitable[WorkspaceToolRuntime]],
    ) -> None:
        self._scope = scope
        self._runtime_provider = runtime_provider
        self._runtime: WorkspaceToolRuntime | None = None

    async def provider_catalog(self) -> tuple[ProviderToolDescriptor, ...]:
        """Load the same workspace-bound public catalog used for execution."""
        if self._runtime is None:
            self._runtime = await self._runtime_provider(self._scope)
        return self._runtime.catalog

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        """Create the scope runtime on demand and keep failures public-safe."""
        if context.run_id != self._scope.run_id:
            return self._failed(call, "scope_mismatch")
        try:
            if self._runtime is None:
                self._runtime = await self._runtime_provider(self._scope)
            return await self._runtime.executor.execute(call, context=context)
        except Exception:
            return self._failed(call, "tool_runtime_unavailable")

    @staticmethod
    def _failed(call: AgentToolCall, reason: str) -> AgentToolExecution:
        return AgentToolExecution(
            call=call,
            result=AgentToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.FAILED,
                result={"error": reason},
                output="tool runtime unavailable",
            ),
        )


_BRIDGE_TOOL_EFFECTS: dict[str, ToolEffect] = {
    "apply_patch": ToolEffect.WRITE,
    "get_diff": ToolEffect.READ,
    "get_status": ToolEffect.READ,
    "list_files": ToolEffect.READ,
    "read_file": ToolEffect.READ,
    "run_targeted_test": ToolEffect.COMMAND,
    "search_text": ToolEffect.READ,
}

_BRIDGE_TOOL_DESCRIPTIONS: dict[str, str] = {
    "apply_patch": "Apply one validated patch to the authorized workspace.",
    "get_diff": "Inspect the bounded diff for the authorized ChangeSet.",
    "get_status": "Inspect the current status of the authorized ChangeSet.",
    "list_files": "List entries below an authorized relative directory.",
    "read_file": "Read bounded text from one authorized relative file.",
    "run_targeted_test": "Run one targeted verification command.",
    "search_text": "Search authorized workspace text with bounded results.",
}


def _bridge_argument_model(name: str) -> type:
    from prp_runtime.tools.command import CommandInvocation
    from prp_runtime.tools.diff import DiffRequest
    from prp_runtime.tools.filesystem import ListFilesArguments, ReadFileArguments
    from prp_runtime.tools.patch import PatchRequest
    from prp_runtime.tools.search import SearchRequest

    models = {
        "apply_patch": PatchRequest,
        "get_diff": DiffRequest,
        "get_status": DiffRequest,
        "list_files": ListFilesArguments,
        "read_file": ReadFileArguments,
        "run_targeted_test": CommandInvocation,
        "search_text": SearchRequest,
    }
    return models[name]


def catalog_from_bridge_capabilities(
    capabilities: ClientCapabilityDescriptor,
) -> tuple[ProviderToolDescriptor, ...]:
    """Project one durable client's tools into a non-executable provider catalog."""

    descriptors: list[ProviderToolDescriptor] = []
    for name in capabilities.tools:
        effect = _BRIDGE_TOOL_EFFECTS[name]
        if effect not in capabilities.effects:
            continue
        descriptors.append(
            ProviderToolDescriptor(
                name=name,
                description=_BRIDGE_TOOL_DESCRIPTIONS[name],
                input_schema=_bridge_argument_model(name).model_json_schema(mode="validation"),
            )
        )
    return tuple(descriptors)


def _bridge_sentinel_registry(capabilities: ClientCapabilityDescriptor) -> ToolRegistry:
    """Build a catalog registry whose handlers must never run on the server."""

    async def sentinel(context: object) -> None:
        del context
        raise RuntimeError("BRIDGE must not invoke a server-local tool handler")

    definitions: list[ToolDefinition] = []
    for name in capabilities.tools:
        effect = _BRIDGE_TOOL_EFFECTS[name]
        if effect not in capabilities.effects:
            continue
        definitions.append(
            ToolDefinition(
                name=name,
                description=_BRIDGE_TOOL_DESCRIPTIONS[name],
                effect=effect,
                argument_model=_bridge_argument_model(name),
                handler=sentinel,
            )
        )
    return ToolRegistry(definitions)


class BridgeRemoteToolExecutor:
    """Persist BRIDGE assignments without a server workspace runtime."""

    def __init__(self, scope: ExecutionScope, store: SqliteStore, settings: Settings) -> None:
        self._scope = scope
        self._store = store
        self._settings = settings
        self._client: RegisteredBridgeClient | None = None

    async def provider_catalog(self) -> tuple[ProviderToolDescriptor, ...]:
        """Return the assigned client's public catalog without opening a root."""

        client = await self._assigned_client()
        return catalog_from_bridge_capabilities(client.capabilities)

    async def execute(
        self,
        call: AgentToolCall,
        *,
        context: AgentToolContext,
    ) -> AgentToolExecution:
        """Persist a remote assignment without invoking a server handler."""

        if context.run_id != self._scope.run_id:
            return ScopedAgentToolExecutor._failed(call, "scope_mismatch")
        try:
            client = await self._assigned_client()
            registry = _bridge_sentinel_registry(client.capabilities)
            definition = registry.get(call.tool_name)
            definition.validate_arguments(call.arguments)
            snapshot_id = await self._ready_snapshot_id()
            persisted_call = ToolCall(
                call_id=AgentToolExecutor._internal_call_id(
                    run_id=self._scope.run_id,
                    work_unit_id=context.work_unit_id,
                    snapshot_id=snapshot_id,
                    provider_call_id=call.call_id,
                    tool_name=call.tool_name,
                ),
                run_id=self._scope.run_id,
                work_unit_id=context.work_unit_id,
                tool_name=call.tool_name,
                effect=definition.effect,
                arguments=dict(call.arguments),
                snapshot_id=snapshot_id,
                requested_at=utc_now(),
            )
            outcome = await ToolExecutor(registry, self._store).execute(
                persisted_call,
                self._scope.agent_options.agent_mode,
                workspace_id=self._scope.workspace_id,
                idempotency_key=persisted_call.call_id,
                isolation_mode=self._scope.agent_options.isolation_mode,
                execution_location=ExecutionLocation.BRIDGE,
                user_explicit_host_yolo=self._scope.agent_options.user_explicit,
                settings=self._settings,
            )
        except Exception:
            return ScopedAgentToolExecutor._failed(call, "tool_runtime_unavailable")
        if (
            outcome.assignment is not None
            and outcome.result is None
            and outcome.decision.outcome is PolicyOutcome.ALLOW
        ):
            raise RemoteToolAssignmentPending(
                outcome.assignment.model_copy(update={"client_id": client.client_id})
            )
        if outcome.result is None:
            if outcome.decision.outcome is PolicyOutcome.ASK:
                return AgentToolExecution(
                    call=call,
                    awaiting_approval=True,
                    reason=outcome.decision.reason_code.value,
                )
            return ScopedAgentToolExecutor._failed(call, "policy_denied")
        return AgentToolExecution(
            call=call,
            result=AgentToolExecutor.public_result(call, outcome.result),
        )

    async def _assigned_client(self) -> RegisteredBridgeClient:
        if self._client is not None:
            return self._client
        clients = await self._store.list_bridge_clients(
            principal_id=self._scope.principal_id,
            workspace_id=self._scope.workspace_id,
        )
        active = tuple(
            client
            for client in clients
            if client.status is BridgeClientStatus.ACTIVE
        )
        if len(active) != 1:
            raise ToolRuntimeError("BRIDGE requires one assigned durable client")
        self._client = active[0]
        return self._client

    async def _ready_snapshot_id(self) -> SnapshotId:
        snapshots = await self._store.list_snapshots(
            self._scope.workspace_id, owner_id=self._scope.principal_id
        )
        ready = tuple(
            snapshot
            for snapshot in snapshots
            if snapshot.status is SnapshotStatus.READY
        )
        if not ready:
            raise ToolRuntimeError("workspace has no ready snapshot")
        return ready[-1].snapshot_id


class ScopeToolRuntimeProvider:
    """Own lazy, non-shared workspace runtimes for one application lifespan."""

    def __init__(
        self,
        store: SqliteStore,
        settings: Settings,
        *,
        factory: WorkspaceToolRuntimeFactory | None = None,
        enable_server_resolver: bool = True,
    ) -> None:
        self._store = store
        self._settings = settings
        self._resolver = (
            WorkspaceResolver(settings.workspace_roots) if enable_server_resolver else None
        )
        self._factory = factory or WorkspaceToolRuntimeFactory()
        self._runtimes: dict[str, WorkspaceToolRuntime] = {}
        self._local_roots: dict[str, Path] = {}
        self._lock = asyncio.Lock()

    def bind_local_workspace(self, workspace_id: str, root: Path) -> None:
        """Bind a process-local directory to a workspace identity."""
        self._local_roots[workspace_id] = canonicalize_local_root(root)

    def executor_for(
        self, scope: ExecutionScope
    ) -> ScopedAgentToolExecutor | BridgeRemoteToolExecutor:
        """Return a lazy adapter without opening a workspace during construction."""
        if scope.agent_options.execution_location is ExecutionLocation.BRIDGE:
            return BridgeRemoteToolExecutor(scope, self._store, self._settings)
        return ScopedAgentToolExecutor(scope, self._runtime_for_scope)

    async def _runtime_for_scope(self, scope: ExecutionScope) -> WorkspaceToolRuntime:
        if scope.agent_options.execution_location is ExecutionLocation.BRIDGE:
            raise ToolRuntimeError("BRIDGE does not resolve a server workspace root")
        existing = self._runtimes.get(scope.run_id)
        if existing is not None:
            return existing
        async with self._lock:
            existing = self._runtimes.get(scope.run_id)
            if existing is not None:
                return existing
            workspace = await self._store.get_workspace(
                scope.workspace_id, owner_id=scope.principal_id
            )
            snapshots = await self._store.list_snapshots(
                scope.workspace_id, owner_id=scope.principal_id
            )
            ready = tuple(
                snapshot
                for snapshot in snapshots
                if snapshot.status is SnapshotStatus.READY
            )
            if not ready:
                raise ToolRuntimeError("workspace has no ready snapshot")
            snapshot = ready[-1]
            manifest = await self._store.get_snapshot_manifest(
                snapshot.snapshot_id, owner_id=scope.principal_id
            )
            copy_mode = select_execution_copy_mode(
                execution_location=scope.agent_options.execution_location,
                isolation_mode=scope.agent_options.isolation_mode,
                strategy=ExecutionStrategy.DIRECT,
                concurrency=1,
            )
            if scope.agent_options.execution_location is ExecutionLocation.LOCAL:
                if copy_mode is not ExecutionCopyMode.IN_PLACE:
                    raise ToolRuntimeError(
                        "copy-backed local execution requires isolation slots"
                    )
                from prp_runtime.workspace.backend import WorkspaceBackend
                from prp_runtime.workspace.resolver import (
                    ResolvedWorkspace,
                    WorkspaceResolver,
                )

                root = self._local_roots.get(scope.workspace_id)
                if root is None:
                    raise ToolRuntimeError("local workspace root is unavailable")
                WorkspaceResolver._validate_root_path(root, kind="local")
                resolved = ResolvedWorkspace(
                    workspace_id=scope.workspace_id,
                    owner_id=scope.principal_id,
                    server_alias="local-workspace",
                    backend=WorkspaceBackend(root),
                    workspace_root=root,
                )
            else:
                if self._resolver is None:
                    raise ToolRuntimeError("server workspace resolver is unavailable")
                resolved = self._resolver.resolve(workspace, owner_id=scope.principal_id)
            try:
                runtime = self._factory.build(
                    scope=scope,
                    resolved_workspace=resolved,
                    snapshot=snapshot,
                    store=self._store,
                    snapshot_manifest=manifest,
                    settings=self._settings,
                )
            except BaseException:
                resolved.close()
                raise
            self._runtimes[scope.run_id] = runtime
            return runtime

    def close(self) -> None:
        """Close every scope runtime in reverse creation order, once."""
        close_error: BaseException | None = None
        for runtime in reversed(tuple(self._runtimes.values())):
            try:
                runtime.close()
            except BaseException as error:
                if close_error is None:
                    close_error = error
        self._runtimes.clear()
        if close_error is not None:
            raise close_error
