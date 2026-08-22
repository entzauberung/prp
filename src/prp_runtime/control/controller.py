"""The single run controller.

One controller drives every strategy. It is the only writer of run and work unit
state, and every state change it makes is paired with an event in the same
transaction. A worker may produce facts; only the controller decides what those
facts mean for the run.

DIRECT uses one work unit and one attempt. CASCADE reuses that same lifecycle
with a bounded, ordered worker profile chain; the verifier evaluates every
produced artifact and the controller decides acceptance.
"""

import hashlib
import tempfile
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path

from pydantic import JsonValue

from prp_runtime.control.budget import (
    check_attempt_budget,
    check_deadline,
    check_token_budget_postflight,
    check_token_budget_preflight,
)
from prp_runtime.control.cascade import (
    CascadeChain,
    CascadeDisposition,
    build_cascade_chain,
    decide_cascade,
    provider_failure_is_retryable,
)
from prp_runtime.control.progressive import (
    ProgressiveRound,
    ReuseDecision,
    ReuseDisposition,
    ReuseReason,
    RevisionDecision,
    RevisionDisposition,
    RevisionStopReason,
    RoundComparison,
    RoundStatus,
    compare_rounds,
    decide_reuse,
    decide_revision,
    new_round_id,
)
from prp_runtime.control.reservations import ReservationRequest
from prp_runtime.control.routing import RoutingFacts, StrategyDecision, route
from prp_runtime.domain.enums import (
    AttemptStatus,
    ExecutionStrategy,
    MergeLedgerStatus,
    ModelRole,
    ReservationStatus,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    ToolEffect,
    WorkUnitStatus,
)
from prp_runtime.domain.errors import (
    DomainValidationError,
    ErrorCode,
    ProviderError,
    StateError,
)
from prp_runtime.domain.events import EventType, payload_from_model
from prp_runtime.domain.models import (
    Artifact,
    Attempt,
    ControllerAction,
    ControllerDecision,
    ErrorCategory,
    ErrorInfo,
    Evidence,
    ExecutionScope,
    GlobalVerificationReport,
    MergeLedger,
    NativeRunRequest,
    Run,
    VerificationResult,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.transitions import (
    AttemptNotAllowedError,
    resolve_run_outcome,
    transition_attempt,
    transition_run,
    transition_work_unit,
)
from prp_runtime.domain.values import (
    new_attempt_id,
    new_merge_id,
    new_run_id,
    new_snapshot_id,
    new_work_unit_id,
    new_workspace_id,
    utc_now,
)
from prp_runtime.planning.compiler import CompiledPlan, compile_plan
from prp_runtime.planning.frontier import compute_frontier
from prp_runtime.planning.models import (
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.planning.planner import (
    Planner,
    PlannerCallResult,
    new_planning_work_unit,
)
from prp_runtime.providers.base import ModelProfile, ProviderAdapter
from prp_runtime.runtime.agent_loop import AgentToolExecutor
from prp_runtime.runtime.context import DependencyArtifact, build_worker_context
from prp_runtime.runtime.coordinator import (
    Coordinator,
    MergeResult,
    StagedChangeSet,
    resolve_work_unit_effect,
)
from prp_runtime.runtime.scheduler import (
    PlannedExecutor,
    Scheduler,
    SlotAwarePlannedExecutor,
    SlotDispatcher,
    StartDecision,
    WaveOutcome,
    WaveResult,
    WaveStatus,
)
from prp_runtime.runtime.worker import ResumeAction, Worker, WorkerResult
from prp_runtime.settings import Settings
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.rules import plan_for_output
from prp_runtime.verification.verifier import (
    GlobalCheckKind,
    RuleVerifier,
    verify_global_round,
)
from prp_runtime.workspace.changes import ChangeSet
from prp_runtime.workspace.merge import (
    MergeError,
    MergeStatus,
    merge_candidate_manifest,
    merge_input_digest,
)
from prp_runtime.workspace.merge import promote_merge as promote_merge_result
from prp_runtime.workspace.models import (
    Snapshot,
    SnapshotStatus,
    Workspace,
    WorkspaceRootMapping,
    WorkspaceSource,
    WorkspaceSourceType,
)
from prp_runtime.workspace.resolver import WorkspaceResolver

__all__ = ["DIRECT_WORK_UNIT_NAME", "SUPPORTED_STRATEGIES", "RunController"]

DIRECT_WORK_UNIT_NAME = "direct"

#: Strategies this version can execute. A request for any other strategy is
#: refused explicitly instead of being silently downgraded.
SUPPORTED_STRATEGIES: frozenset[ExecutionStrategy] = frozenset(
    {
        ExecutionStrategy.DIRECT,
        ExecutionStrategy.CASCADE,
        ExecutionStrategy.PLANNED,
        ExecutionStrategy.PROGRESSIVE,
    }
)

ProgressiveMergeRoots = Callable[
    [ProgressiveRound],
    Awaitable[tuple[Path, Mapping[str, Path], Path]],
]

_TERMINAL_RUN_EVENTS: dict[RunStatus, EventType] = {
    RunStatus.SUCCEEDED: EventType.RUN_SUCCEEDED,
    RunStatus.FAILED: EventType.RUN_FAILED,
    RunStatus.CANCELLED: EventType.RUN_CANCELLED,
}

_WORK_UNIT_EVENTS: dict[WorkUnitStatus, EventType] = {
    WorkUnitStatus.READY: EventType.WORK_UNIT_READY,
    WorkUnitStatus.RUNNING: EventType.WORK_UNIT_STARTED,
    WorkUnitStatus.SUCCEEDED: EventType.WORK_UNIT_SUCCEEDED,
    WorkUnitStatus.FAILED: EventType.WORK_UNIT_FAILED,
    WorkUnitStatus.CANCELLED: EventType.WORK_UNIT_CANCELLED,
    WorkUnitStatus.BLOCKED: EventType.WORK_UNIT_BLOCKED,
    WorkUnitStatus.INVALIDATED: EventType.WORK_UNIT_INVALIDATED,
}


class RunController:
    """Creates, executes and cancels runs."""

    def __init__(
        self,
        store: SqliteStore,
        settings: Settings,
        adapters: Mapping[str, ProviderAdapter],
        *,
        tool_executor: AgentToolExecutor | None = None,
        tool_executor_provider: Callable[[ExecutionScope], AgentToolExecutor] | None = None,
        progressive_merge_roots: ProgressiveMergeRoots | None = None,
        max_tool_rounds: int = 8,
    ) -> None:
        self._store = store
        self._settings = settings
        self._adapters = dict(adapters)
        self._tool_executor = tool_executor
        self._tool_executor_provider = tool_executor_provider
        self._progressive_merge_roots = progressive_merge_roots
        self._max_tool_rounds = max_tool_rounds
        for profile in settings.profiles:
            store.register_capacity_limit(profile.alias, profile.max_concurrency)

    async def create_run(self, request: NativeRunRequest) -> Run:
        """Persist a new run in PENDING and record its creation."""
        run = Run(run_id=new_run_id(), request=request, created_at=utc_now())
        async with self._store.transaction():
            await self._store.create_run(run)
            await self._store.append_event(
                run.run_id,
                EventType.RUN_CREATED,
                {"request": request.model_dump(mode="json")},
            )
        return run

    def merge_candidate(
        self,
        base_root: Path,
        changes: Sequence[StagedChangeSet],
        *,
        staging_root: Path,
        verify: Callable[[Path], None] | None = None,
    ) -> MergeResult:
        """Create a bounded merge candidate; promotion remains a separate action."""
        return Coordinator().merge_candidate(
            base_root,
            changes,
            staging_root=staging_root,
            verify=verify,
        )

    async def merge_progressive_round(
        self,
        round_id: str,
        *,
        base_root: Path,
        staged_roots: Mapping[str, Path],
        staging_root: Path,
        verify: Callable[[Path], None] | None = None,
    ) -> MergeResult:
        """Merge one persisted round into a durable candidate without promotion."""
        progressive_round = await self._store.get_progressive_round(round_id)
        if not progressive_round.change_set_ids:
            raise MergeError("Progressive round has no ChangeSets to merge")
        snapshot = await self._store.get_snapshot(
            progressive_round.base_snapshot_id,
            owner_id=self._settings.service_principal,
        )
        staged: list[StagedChangeSet] = []
        for change_set_id in progressive_round.change_set_ids:
            root = staged_roots.get(change_set_id)
            if root is None:
                raise MergeError("Progressive round is missing an isolated ChangeSet root")
            change_set = await self._store.get_change_set(change_set_id)
            if (
                change_set.run_id != progressive_round.run_id
                or change_set.workspace_id != snapshot.workspace_id
                or change_set.base_snapshot_id != progressive_round.base_snapshot_id
            ):
                raise MergeError("Progressive ChangeSet lineage does not match its round")
            staged.append(StagedChangeSet(change_set=change_set, root=root))

        input_digest = merge_input_digest(
            progressive_round.base_snapshot_id,
            progressive_round.change_set_ids,
        )
        planned = MergeLedger(
            merge_id=new_merge_id(),
            run_id=progressive_round.run_id,
            workspace_id=snapshot.workspace_id,
            base_snapshot_id=progressive_round.base_snapshot_id,
            change_set_ids=progressive_round.change_set_ids,
            input_digest=input_digest,
            status=MergeLedgerStatus.PLANNED,
        )
        ledger = await self._store.create_merge_ledger(planned)
        if ledger.status is not MergeLedgerStatus.PLANNED:
            raise MergeError("merge input already has a non-replayable lifecycle")
        running = ledger.model_copy(update={"status": MergeLedgerStatus.RUNNING})
        await self._store.update_merge_ledger(running)
        result = self.merge_candidate(
            base_root,
            tuple(staged),
            staging_root=staging_root,
            verify=verify,
        )
        if result.status is MergeStatus.MERGED:
            if result.merged_snapshot_id is None or result.merged_content_hash is None:
                raise MergeError("merged candidate omitted durable snapshot facts")
            manifest = merge_candidate_manifest(result.staging_root)
            timestamp = utc_now()
            candidate = Snapshot(
                snapshot_id=result.merged_snapshot_id,
                workspace_id=snapshot.workspace_id,
                status=SnapshotStatus.READY,
                created_at=timestamp,
                completed_at=timestamp,
            )
            persisted_snapshot = await self._store.create_snapshot(
                candidate,
                manifest,
                owner_id=self._settings.service_principal,
            )
            result = result.model_copy(
                update={"merged_snapshot_id": persisted_snapshot.snapshot_id}
            )
            merged = running.model_copy(
                update={
                    "status": MergeLedgerStatus.MERGED,
                    "merged_snapshot_id": persisted_snapshot.snapshot_id,
                    "merged_content_hash": result.merged_content_hash,
                    "completed_at": timestamp,
                }
            )
            await self._store.update_merge_ledger(merged)
        else:
            terminal_status = (
                MergeLedgerStatus.CONFLICT
                if result.status is MergeStatus.CONFLICT
                else MergeLedgerStatus.UNKNOWN
            )
            unresolved = running.model_copy(
                update={
                    "status": terminal_status,
                    "completed_at": utc_now(),
                }
            )
            await self._store.update_merge_ledger(unresolved)
        return result

    def promote_merge(self, result: MergeResult, destination: Path) -> MergeResult:
        """Promote a verified merge through the atomic workspace boundary."""
        return promote_merge_result(result, destination)

    async def cancel(self, run_id: str) -> Run:
        """Request cancellation.

        A run that has not started is cancelled at once. A running run enters
        CANCELLING, which immediately blocks any further attempt. Cancelling an
        already terminal run changes nothing.
        """
        run = await self._store.get_run(run_id)
        if run.status.is_terminal:
            return run
        if run.status is RunStatus.CANCELLING:
            return run
        if run.status is RunStatus.PENDING:
            return await self._finish_run(run, RunStatus.CANCELLED)
        cancelling = Run.model_validate(
            run.model_dump()
            | {"status": transition_run(run.status, RunStatus.CANCELLING)}
        )
        async with self._store.transaction():
            await self._store.update_run(cancelling)
            await self._store.append_event(run_id, EventType.RUN_CANCELLING, {})
        return cancelling

    async def commit_plan(
        self,
        run_id: str,
        result: CompiledPlan | PlanRejection,
        *,
        target_graph_version: int,
        revision: PlanRevision | None = None,
    ) -> tuple[WorkUnit, ...] | PlanRejection:
        """Commit one compiled graph or record its structured rejection."""
        run = await self._store.get_run(run_id)
        rejection: PlanRejection | None = None
        if isinstance(result, PlanRejection):
            rejection = result
        elif result.run_id != run_id:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("compiled plan run_id does not match the target run",),
            )
        elif result.graph_version != target_graph_version:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("compiled plan graph_version does not match the target version",),
            )
        elif target_graph_version <= run.graph_version:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("target graph_version must be greater than the current version",),
            )
        elif run.status.is_terminal:
            rejection = PlanRejection(
                summary="The compiled plan was rejected",
                reasons=("a terminal run cannot accept a new graph",),
            )

        if rejection is not None:
            await self._record_plan_rejection(
                run, target_graph_version, rejection
            )
            return rejection

        assert isinstance(result, CompiledPlan)
        created_at = utc_now()
        work_units = tuple(
            WorkUnit(
                work_unit_id=draft.work_unit_id,
                run_id=draft.run_id,
                graph_version=draft.graph_version,
                lineage_key=draft.lineage_key,
                dependency_fingerprint=draft.dependency_fingerprint,
                content_fingerprint=draft.content_fingerprint,
                name=draft.name,
                instruction=draft.instruction,
                acceptance_criteria=draft.acceptance_criteria,
                output=draft.output,
                depends_on=draft.depends_on,
                resource_claims=draft.resource_claims,
                created_at=created_at,
            )
            for draft in result.nodes
        )
        reuse_sources: dict[str, WorkUnit] = {}
        reuse_facts: dict[
            str, tuple[tuple[Artifact, ...], tuple[Attempt, ...], tuple[Evidence, ...]]
        ] = {}
        reuse_decisions: dict[str, ReuseDecision] = {}
        invalidation_reasons: dict[str, str] = {}
        if revision is not None:
            (
                reuse_sources,
                reuse_facts,
                reuse_decisions,
                invalidation_reasons,
            ) = await self._prepare_revision_reuse(
                run,
                work_units,
                base_graph_version=revision.base_graph_version,
            )
            work_units = tuple(
                work_unit.model_copy(
                    update={"status": WorkUnitStatus.SUCCEEDED}
                )
                if work_unit.work_unit_id in reuse_sources
                else work_unit
                for work_unit in work_units
            )
        updated_run = Run.model_validate(
            run.model_dump()
            | {
                "graph_version": target_graph_version,
                "final_work_unit_id": result.final_work_unit_id,
            }
        )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.REVISE_PLAN
                if revision is not None
                else ControllerAction.COMMIT_PLAN
            ),
            rationale=(
                f"compiled graph version {target_graph_version} passed validation"
            ),
        )
        try:
            async with self._store.transaction():
                if revision is not None:
                    await self._store.append_event(
                        run.run_id,
                        EventType.PLAN_REVISED,
                        {
                            "graph_version": target_graph_version,
                            "base_graph_version": revision.base_graph_version,
                            "reason": revision.reason.value,
                            "summary": revision.summary,
                        },
                    )
                    for source_work_unit_id, reason in invalidation_reasons.items():
                        source = await self._store.get_work_unit(source_work_unit_id)
                        await self._store.append_event(
                            run.run_id,
                            EventType.WORK_UNIT_INVALIDATED,
                            {
                                "work_unit_id": source_work_unit_id,
                                "reason": reason,
                                "graph_version": source.graph_version,
                                "replacement_graph_version": target_graph_version,
                                "lineage_key": source.lineage_key,
                            },
                        )
                await self._store.append_event(
                    run.run_id,
                    EventType.PLAN_PROPOSED,
                    {
                        "graph_version": target_graph_version,
                        "node_count": len(work_units),
                    },
                )
                await self._store.create_graph(work_units)
                for work_unit in work_units:
                    await self._store.append_event(
                        run.run_id,
                        EventType.WORK_UNIT_CREATED,
                        {
                            "work_unit_id": work_unit.work_unit_id,
                            "name": work_unit.name,
                            "graph_version": target_graph_version,
                            "status": work_unit.status.value,
                        },
                    )
                for work_unit in work_units:
                    historical_source = reuse_sources.get(work_unit.work_unit_id)
                    if historical_source is None:
                        continue
                    facts = reuse_facts[work_unit.work_unit_id]
                    source_artifact_ids, artifact_ids, source_attempt_ids, attempt_ids = (
                        await self._copy_reused_facts(
                            run,
                            historical_source,
                            work_unit,
                            facts,
                        )
                    )
                    reuse_decision = reuse_decisions[work_unit.work_unit_id]
                    await self._store.append_event(
                        run.run_id,
                        EventType.WORK_UNIT_REUSED,
                        {
                            "work_unit_id": work_unit.work_unit_id,
                            "source_work_unit_id": historical_source.work_unit_id,
                            "source_attempt_id": source_attempt_ids[0],
                            "attempt_id": attempt_ids[0],
                            "source_attempt_ids": list(source_attempt_ids),
                            "attempt_ids": list(attempt_ids),
                            "lineage_key": work_unit.lineage_key,
                            "reason": reuse_decision.rationale,
                            "source_artifact_ids": list(source_artifact_ids),
                            "artifact_ids": list(artifact_ids),
                        },
                    )
                await self._store.update_run(updated_run)
                await self._store.append_event(
                    run.run_id,
                    EventType.CONTROLLER_DECISION,
                    payload_from_model("decision", decision),
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.PLAN_COMMITTED,
                    {
                        "graph_version": target_graph_version,
                        "final_work_unit_id": result.final_work_unit_id,
                        "work_unit_ids": [unit.work_unit_id for unit in work_units],
                    },
                )
        except StateError:
            rejection = PlanRejection(
                summary="The compiled plan could not be committed",
                reasons=("the graph write violated persisted state",),
            )
            await self._record_plan_rejection(run, target_graph_version, rejection)
            return rejection
        return work_units

    async def _prepare_revision_reuse(
        self,
        run: Run,
        candidates: tuple[WorkUnit, ...],
        *,
        base_graph_version: int,
    ) -> tuple[
        dict[str, WorkUnit],
        dict[str, tuple[tuple[Artifact, ...], tuple[Attempt, ...], tuple[Evidence, ...]]],
        dict[str, ReuseDecision],
        dict[str, str],
    ]:
        """Resolve revision reuse from the immutable base graph's public facts."""
        historical = await self._store.list_work_units(
            run.run_id, graph_version=base_graph_version
        )
        by_lineage = {
            unit.lineage_key: unit
            for unit in historical
            if unit.lineage_key is not None
        }
        candidate_artifact_hashes: dict[str, tuple[str | None, ...]] = {}
        reuse_sources: dict[str, WorkUnit] = {}
        reuse_facts: dict[
            str, tuple[tuple[Artifact, ...], tuple[Attempt, ...], tuple[Evidence, ...]]
        ] = {}
        decisions: dict[str, ReuseDecision] = {}
        invalidations: dict[str, str] = {}
        merge_ledgers = await self._store.list_merge_ledgers(run_id=run.run_id)
        latest_merge = merge_ledgers[-1] if merge_ledgers else None
        progressive_base_snapshot_id = (
            None if latest_merge is None else latest_merge.base_snapshot_id
        )
        progressive_merged_snapshot_id = (
            None if latest_merge is None else latest_merge.merged_snapshot_id
        )
        progressive_merge_input_digest = (
            None if latest_merge is None else latest_merge.input_digest
        )
        progressive_change_set_ids = (
            None if latest_merge is None else latest_merge.change_set_ids
        )

        for candidate in candidates:
            if candidate.lineage_key is None:
                continue
            source = by_lineage.get(candidate.lineage_key)
            if source is None:
                continue
            artifacts = await self._store.list_artifacts(source.work_unit_id)
            attempts = await self._store.list_attempts(source.work_unit_id)
            evidence = await self._store.list_evidence(source.work_unit_id)
            historical_dependency_hashes: list[str | None] = []
            for dependency_id in source.depends_on:
                dependency_artifacts = await self._store.list_artifacts(dependency_id)
                if not dependency_artifacts:
                    historical_dependency_hashes.append(None)
                else:
                    historical_dependency_hashes.extend(
                        hashlib.sha256(artifact.content.encode("utf-8")).hexdigest()
                        for artifact in dependency_artifacts
                    )
            candidate_dependency_hashes: list[str | None] = []
            for dependency_id in candidate.depends_on:
                candidate_dependency_hashes.extend(
                    candidate_artifact_hashes.get(dependency_id, (None,))
                )
            decision = decide_reuse(
                source,
                candidate,
                historical_dependency_artifact_hashes=tuple(
                    historical_dependency_hashes
                ),
                candidate_dependency_artifact_hashes=tuple(candidate_dependency_hashes),
                historical_base_snapshot_id=progressive_base_snapshot_id,
                candidate_base_snapshot_id=progressive_base_snapshot_id,
                historical_merged_snapshot_id=progressive_merged_snapshot_id,
                candidate_merged_snapshot_id=progressive_merged_snapshot_id,
                historical_merge_input_digest=progressive_merge_input_digest,
                candidate_merge_input_digest=progressive_merge_input_digest,
                historical_change_set_ids=progressive_change_set_ids,
                candidate_change_set_ids=progressive_change_set_ids,
                historical_evidence_ids=tuple(
                    evidence.evidence_id for evidence in evidence
                )
                if latest_merge is not None
                else None,
                candidate_evidence_ids=tuple(
                    evidence.evidence_id for evidence in evidence
                )
                if latest_merge is not None
                else None,
            )
            attempt_by_id = {attempt.attempt_id: attempt for attempt in attempts}
            artifact_attempts = {
                artifact.attempt_id for artifact in artifacts
            }
            facts_are_complete = bool(artifacts) and all(
                attempt_by_id.get(attempt_id) is not None
                and attempt_by_id[attempt_id].status is AttemptStatus.SUCCEEDED
                for attempt_id in artifact_attempts
            ) and bool(evidence) and all(
                attempt.status is AttemptStatus.SUCCEEDED for attempt in attempts
            ) and all(row.result is VerificationResult.PASS for row in evidence)
            if decision.disposition is ReuseDisposition.REUSE and not facts_are_complete:
                decision = ReuseDecision(
                    disposition=ReuseDisposition.RECOMPUTE,
                    reason=ReuseReason.ATTEMPT_HISTORY_NOT_PROVEN,
                    rationale=(
                        "historical success has incomplete, failed, unknown, or "
                        "non-passing attempt facts"
                    ),
                    lineage_key=candidate.lineage_key,
                )
            decisions[candidate.work_unit_id] = decision
            if decision.disposition is ReuseDisposition.REUSE:
                reuse_sources[candidate.work_unit_id] = source
                reuse_facts[candidate.work_unit_id] = (artifacts, attempts, evidence)
                candidate_artifact_hashes[candidate.work_unit_id] = tuple(
                    hashlib.sha256(artifact.content.encode("utf-8")).hexdigest()
                    for artifact in artifacts
                )
            else:
                invalidations[source.work_unit_id] = decision.reason.value
        return reuse_sources, reuse_facts, decisions, invalidations

    async def _copy_reused_facts(
        self,
        run: Run,
        source: WorkUnit,
        target: WorkUnit,
        facts: tuple[tuple[Artifact, ...], tuple[Attempt, ...], tuple[Evidence, ...]],
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        """Materialize a successful source result under the replacement unit."""
        source_artifacts, source_attempts, source_evidence = facts
        attempts_by_id = {attempt.attempt_id: attempt for attempt in source_attempts}
        new_attempt_by_source: dict[str, Attempt] = {}
        source_attempt_ids: list[str] = []
        attempt_ids: list[str] = []
        artifact_ids: list[str] = []
        source_artifact_ids: list[str] = []
        artifact_map: dict[str, str] = {}
        for artifact in source_artifacts:
            source_attempt = attempts_by_id[artifact.attempt_id]
            new_attempt = new_attempt_by_source.get(source_attempt.attempt_id)
            if new_attempt is None:
                timestamp = target.created_at
                new_attempt = Attempt(
                    attempt_id=new_attempt_id(),
                    run_id=run.run_id,
                    work_unit_id=target.work_unit_id,
                    attempt_index=len(new_attempt_by_source) + 1,
                    role=source_attempt.role,
                    model=source_attempt.model,
                    status=AttemptStatus.SUCCEEDED,
                    usage=source_attempt.usage,
                    created_at=timestamp,
                    started_at=timestamp,
                    completed_at=timestamp,
                )
                new_attempt_by_source[source_attempt.attempt_id] = new_attempt
                source_attempt_ids.append(source_attempt.attempt_id)
                attempt_ids.append(new_attempt.attempt_id)
                await self._store.create_attempt(new_attempt)
                await self._store.append_event(
                    run.run_id,
                    EventType.ATTEMPT_SUCCEEDED,
                    {
                        "work_unit_id": target.work_unit_id,
                        "attempt_id": new_attempt.attempt_id,
                        "source_attempt_id": source_attempt.attempt_id,
                        "reused": True,
                    },
                )
            copied = Artifact(
                artifact_id=new_artifact_id(),
                run_id=run.run_id,
                work_unit_id=target.work_unit_id,
                attempt_id=new_attempt.attempt_id,
                name=artifact.name,
                kind=artifact.kind,
                content=artifact.content,
                created_at=target.created_at,
            )
            await self._store.add_artifact(copied)
            artifact_map[artifact.artifact_id] = copied.artifact_id
            source_artifact_ids.append(artifact.artifact_id)
            artifact_ids.append(copied.artifact_id)
            await self._store.append_event(
                run.run_id,
                EventType.ARTIFACT_PRODUCED,
                {
                    "work_unit_id": target.work_unit_id,
                    "artifact_id": copied.artifact_id,
                    "name": copied.name,
                    "kind": copied.kind.value,
                    "source_artifact_id": artifact.artifact_id,
                    "reused": True,
                },
            )
        for evidence in source_evidence:
            copied_evidence = Evidence(
                evidence_id=new_evidence_id(),
                run_id=run.run_id,
                work_unit_id=target.work_unit_id,
                artifact_id=artifact_map[evidence.artifact_id],
                kind=evidence.kind,
                rule=evidence.rule,
                result=evidence.result,
                detail=evidence.detail,
                created_at=target.created_at,
            )
            await self._store.add_evidence(copied_evidence)
            await self._store.append_event(
                run.run_id,
                EventType.EVIDENCE_RECORDED,
                {
                    "work_unit_id": target.work_unit_id,
                    "evidence_id": copied_evidence.evidence_id,
                    "result": copied_evidence.result.value,
                    "source_evidence_id": evidence.evidence_id,
                    "reused": True,
                },
            )
        return (
            tuple(source_artifact_ids),
            tuple(artifact_ids),
            tuple(source_attempt_ids),
            tuple(attempt_ids),
        )

    async def commit_revision(
        self,
        run_id: str,
        revision: PlanRevision,
        *,
        target_graph_version: int | None = None,
    ) -> tuple[WorkUnit, ...] | PlanRejection:
        """Compile and atomically commit one strictly next graph version."""
        run = await self._store.get_run(run_id)
        target = (
            revision.base_graph_version + 1
            if target_graph_version is None
            else target_graph_version
        )
        rejection: PlanRejection | None = None
        if run.status.is_terminal:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("a terminal run cannot accept a new graph",),
            )
        elif revision.base_graph_version != run.graph_version:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("revision base_graph_version is stale",),
            )
        elif target != revision.base_graph_version + 1:
            rejection = PlanRejection(
                summary="The plan revision was rejected",
                reasons=("a revision must advance exactly one graph version",),
            )
        if rejection is not None:
            await self._record_plan_rejection(run, target, rejection)
            return rejection

        compiled = compile_plan(
            revision.proposal,
            run_id=run_id,
            graph_version=target,
        )
        return await self.commit_plan(
            run_id,
            compiled,
            target_graph_version=target,
            revision=revision,
        )

    async def dispatch_planned_wave(
        self,
        run_id: str,
        *,
        scheduler: Scheduler,
        execute: PlannedExecutor,
        slot_dispatcher: SlotDispatcher | None = None,
        execute_in_slot: SlotAwarePlannedExecutor | None = None,
        eligible_work_unit_ids: Sequence[str] | None = None,
        graph_version: int | None = None,
        max_concurrency: int | None = None,
        reservation_capacity_key: str | None = None,
        settle_run: bool = True,
        finalize_cancel: bool = True,
    ) -> WaveResult:
        """Dispatch one committed frontier wave through Controller callbacks."""
        run = await self._store.get_run(run_id)
        version = run.graph_version if graph_version is None else graph_version
        work_units = await self._store.list_work_units(
            run_id, graph_version=version
        )

        if run.status is RunStatus.CANCELLED:
            await self._cancel_planned_units(work_units)
            return WaveResult(graph_version=version, status=WaveStatus.CANCELLED)
        if run.status is RunStatus.SUCCEEDED:
            return WaveResult(graph_version=version, status=WaveStatus.COMPLETE)
        if run.status is RunStatus.FAILED:
            return WaveResult(graph_version=version, status=WaveStatus.BLOCKED)
        if run.status is RunStatus.PENDING:
            run = await self._start_run(run, ExecutionStrategy.PLANNED)
        if run.status is RunStatus.CANCELLING:
            await self._cancel_planned_units(work_units)
            current = await self._store.get_run(run_id)
            if current.status is RunStatus.CANCELLING and finalize_cancel:
                await self._finish_run(current, RunStatus.CANCELLED)
            return WaveResult(graph_version=version, status=WaveStatus.CANCELLED)

        async def on_started(unit: WorkUnit) -> WorkUnit | StartDecision | None:
            current = await self._store.get_run(run_id)
            if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                if not unit.status.is_terminal:
                    await self._cancel_planned_unit(unit)
                return None
            ready = unit
            if unit.status is WorkUnitStatus.PENDING:
                ready = await self._advance_work_unit(unit, WorkUnitStatus.READY)
            async with self._store.transaction():
                current = await self._store.get_run(run_id)
                if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                    if not ready.status.is_terminal:
                        await self._cancel_planned_unit(ready)
                    return StartDecision.cancel()

                admission_error = await self._preflight_budget_with_reservations(
                    current
                )
                if admission_error is not None:
                    blocked = WorkUnit.model_validate(
                        ready.model_dump()
                        | {
                            "status": transition_work_unit(
                                ready.status, WorkUnitStatus.BLOCKED
                            )
                        }
                    )
                    await self._store.update_work_unit(blocked)
                    await self._store.append_event(
                        run_id,
                        EventType.WORK_UNIT_BLOCKED,
                        {
                            "work_unit_id": ready.work_unit_id,
                            "reason": admission_error.message,
                            "error": admission_error.model_dump(mode="json"),
                        },
                    )
                    return StartDecision.reject(admission_error)
                try:
                    await self._store.reserve_reservation(
                        ReservationRequest(
                            run_id=run_id,
                            work_unit_id=ready.work_unit_id,
                            dispatch_key=(
                                f"{ready.work_unit_id}:1:{ModelRole.WORKER.value}"
                            ),
                            capacity_key=reservation_capacity_key,
                        ),
                        capacity_limit=self._profile_capacity(reservation_capacity_key),
                    )
                except StateError:
                    return StartDecision.defer()
                return await self._advance_work_unit(ready, WorkUnitStatus.RUNNING)

        async def on_succeeded(unit: WorkUnit) -> None:
            await self._advance_work_unit(unit, WorkUnitStatus.SUCCEEDED)

        async def on_failed(unit: WorkUnit, error: ErrorInfo) -> None:
            current = await self._store.get_run(run_id)
            if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED):
                await self._cancel_planned_unit(unit)
                return
            stored_unit = await self._store.get_work_unit(unit.work_unit_id)
            if stored_unit.status is WorkUnitStatus.RUNNING and error.message == (
                "tool call is awaiting approval"
            ):
                return
            await self._advance_work_unit(unit, WorkUnitStatus.FAILED, error=error)

        concurrency = max_concurrency
        if concurrency is None:
            concurrency = run.request.budget.max_concurrency or 1
        profile_capacity = self._profile_capacity(reservation_capacity_key)
        if profile_capacity is not None:
            concurrency = min(concurrency, profile_capacity)
        result = await scheduler.run_wave(
            work_units,
            graph_version=version,
            max_concurrency=concurrency,
            execute=execute,
            on_started=on_started,
            on_succeeded=on_succeeded,
            on_failed=on_failed,
            eligible_work_unit_ids=eligible_work_unit_ids,
            slot_dispatcher=slot_dispatcher,
            execute_in_slot=execute_in_slot,
            propagate_executor_exceptions=True,
        )
        for change_set in result.change_sets:
            await self._store.create_change_set(change_set)
        current = await self._store.get_run(run_id)
        if current.status in (RunStatus.CANCELLING, RunStatus.CANCELLED) or (
            result.status is WaveStatus.CANCELLED
        ):
            await self._cancel_planned_units(
                await self._store.list_work_units(run_id, graph_version=version)
            )
            current = await self._store.get_run(run_id)
            if current.status is RunStatus.CANCELLING and finalize_cancel:
                await self._finish_run(current, RunStatus.CANCELLED)
            return result.model_copy(update={"status": WaveStatus.CANCELLED})

        await self._propagate_planned_blocked(run_id, version)
        if settle_run:
            await self._settle_planned_run(run_id, version)
        return result

    def _profile_capacity(self, alias: str | None) -> int | None:
        if alias is None:
            return None
        return next(
            (
                profile.max_concurrency
                for profile in self._settings.profiles
                if profile.alias == alias
            ),
            None,
        )

    async def _cancel_planned_unit(self, unit: WorkUnit) -> None:
        """Cancel one non-terminal unit while preserving the event ledger."""
        if unit.status.is_terminal:
            return
        await self._advance_work_unit(unit, WorkUnitStatus.CANCELLED)

    async def _cancel_planned_units(self, units: tuple[WorkUnit, ...]) -> None:
        """Cancel every still-runnable unit in a committed graph version."""
        for unit in units:
            await self._cancel_planned_unit(unit)

    async def _propagate_planned_blocked(
        self, run_id: str, graph_version: int
    ) -> tuple[WorkUnit, ...]:
        """Materialize dependency failures as stable BLOCKED work-unit facts."""
        while True:
            units = await self._store.list_work_units(
                run_id, graph_version=graph_version
            )
            frontier = compute_frontier(units, graph_version=graph_version)
            by_id = {unit.work_unit_id: unit for unit in units}
            changed = False
            for detail in frontier.blocked_details:
                unit = by_id[detail.work_unit_id]
                if unit.status not in (WorkUnitStatus.PENDING, WorkUnitStatus.READY):
                    continue
                dependency_suffix = ", ".join(detail.dependency_ids)
                reason = detail.reason.value
                if dependency_suffix:
                    reason = f"{reason}: {dependency_suffix}"
                await self._advance_work_unit(
                    unit, WorkUnitStatus.BLOCKED, reason=reason
                )
                changed = True
            if not changed:
                return units

    async def _settle_planned_run(self, run_id: str, graph_version: int) -> Run:
        """Settle a planned run only after its current graph has no work left."""
        run = await self._store.get_run(run_id)
        if run.status.is_terminal:
            return run
        pending_approvals = await self._store.list_tool_calls(
            run_id,
            statuses=[ToolCallStatus.AWAITING_APPROVAL],
        )
        if pending_approvals:
            return run
        units = await self._store.list_work_units(run_id, graph_version=graph_version)
        frontier = compute_frontier(units, graph_version=graph_version)
        if frontier.ready or frontier.waiting:
            return run
        if not units:
            return run
        statuses = tuple(unit.status for unit in units)
        if run.status is RunStatus.CANCELLING:
            await self._cancel_planned_units(units)
            return await self._finish_run(
                await self._store.get_run(run_id), RunStatus.CANCELLED
            )
        if any(
            status in (WorkUnitStatus.FAILED, WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED)
            for status in statuses
        ):
            failed_ids = tuple(
                unit.work_unit_id
                for unit in units
                if unit.status
                in (WorkUnitStatus.FAILED, WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED)
            )
            error: ErrorInfo | None = None
            failed_id_set = set(failed_ids)
            for event in reversed(await self._store.list_events(run_id)):
                if event.event_type is not EventType.WORK_UNIT_FAILED:
                    continue
                if event.payload.get("work_unit_id") not in failed_id_set:
                    continue
                error = ErrorInfo.model_validate(event.payload["error"])
                break
            if error is None:
                for event in reversed(await self._store.list_events(run_id)):
                    if event.event_type is not EventType.WORK_UNIT_BLOCKED:
                        continue
                    if event.payload.get("work_unit_id") not in failed_id_set:
                        continue
                    blocked_error = event.payload.get("error")
                    if blocked_error is not None:
                        error = ErrorInfo.model_validate(blocked_error)
                        break
            if error is None:
                error = ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="planned graph failed: " + ", ".join(failed_ids),
                )
            return await self._finish_run(run, RunStatus.FAILED, error=error)
        if all(status is WorkUnitStatus.SUCCEEDED for status in statuses):
            return await self._finish_run(run, RunStatus.SUCCEEDED)
        if any(status is WorkUnitStatus.CANCELLED for status in statuses):
            return await self._finish_run(run, RunStatus.CANCELLED)
        return run

    async def _record_plan_rejection(
        self,
        run: Run,
        target_graph_version: int,
        rejection: PlanRejection,
    ) -> None:
        rationale = "; ".join(rejection.reasons)
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.REJECT_PLAN,
            rationale=rationale,
        )
        async with self._store.transaction():
            await self._store.append_event(
                run.run_id,
                EventType.PLAN_PROPOSED,
                {"graph_version": target_graph_version, "node_count": 0},
            )
            await self._store.append_event(
                run.run_id,
                EventType.CONTROLLER_DECISION,
                payload_from_model("decision", decision),
            )
            await self._store.append_event(
                run.run_id,
                EventType.PLAN_REJECTED,
                {
                    "graph_version": target_graph_version,
                    "reason": rejection.summary,
                    "reasons": list(rejection.reasons),
                },
            )

    async def execute(
        self,
        run_id: str,
        *,
        routing_facts: RoutingFacts | None = None,
        principal_id: str | None = None,
    ) -> Run:
        """Execute a pending run to a terminal status."""
        run = await self._store.get_run(run_id)
        execution_scope = (
            None
            if principal_id is None
            else await self._store.get_execution_scope(run_id, principal_id=principal_id)
        )
        if run.status.is_terminal or run.status is RunStatus.CANCELLING:
            return run
        if run.status is not RunStatus.PENDING:
            if run.status is RunStatus.RUNNING:
                return await self._resume_running_run(
                    run,
                    execution_scope=execution_scope,
                )
            raise StateError(
                f"run {run_id} is already {run.status.value}",
                code=ErrorCode.ILLEGAL_STATE_TRANSITION,
            )
        selection = self._select_strategy(run.request, facts=routing_facts)
        assert selection.strategy is not None
        strategy = selection.strategy
        cascade_chain: CascadeChain | None = None
        cascade_workers: tuple[Worker, ...] | None = None
        planner: Planner | None = None
        planned_worker: Worker | None = None
        if strategy is ExecutionStrategy.CASCADE:
            cascade_chain = build_cascade_chain(
                self._settings.require_profile(ModelRole.WORKER),
                self._settings.cascade_profiles,
            )
            # Validate the complete finite chain before starting the run. A
            # missing adapter is configuration failure, never a partial run.
            cascade_workers = tuple(
                self._worker_for_profile(profile, execution_scope=execution_scope)
                for profile in cascade_chain
            )
        elif strategy in (ExecutionStrategy.PLANNED, ExecutionStrategy.PROGRESSIVE):
            planner = self._planner()
            planned_worker = self._worker_for(
                ModelRole.WORKER, execution_scope=execution_scope
            )
        run = await self._start_run(run, strategy, selection=selection)
        if strategy is ExecutionStrategy.CASCADE:
            assert cascade_chain is not None
            assert cascade_workers is not None
            return await self._execute_cascade(run, cascade_chain, cascade_workers)
        if strategy is ExecutionStrategy.PLANNED:
            assert planner is not None
            assert planned_worker is not None
            return await self._execute_planned(run, planner, planned_worker)
        if strategy is ExecutionStrategy.PROGRESSIVE:
            assert planner is not None
            assert planned_worker is not None
            return await self._execute_progressive(run, planner, planned_worker)
        return await self._execute_direct(run, execution_scope=execution_scope)

    # --- strategy routing -------------------------------------------------------

    def _select_strategy(
        self,
        request: NativeRunRequest,
        *,
        facts: RoutingFacts | None = None,
    ) -> StrategyDecision:
        """Choose a strategy through the deterministic routing contract."""
        decision = route(
            request,
            facts=facts,
            available_strategies=SUPPORTED_STRATEGIES,
        )
        if decision.rejected:
            assert decision.rejection is not None
            raise DomainValidationError(
                f"{decision.reason} ({decision.rejection.code.value})",
                code=ErrorCode.INVALID_REQUEST,
                field=(
                    "strategy"
                    if request.routing_policy is RoutingPolicy.MANUAL
                    else "routing_policy"
                ),
            )
        assert decision.strategy is not None
        return decision

    # --- run lifecycle ----------------------------------------------------------

    async def _start_run(
        self,
        run: Run,
        strategy: ExecutionStrategy,
        *,
        selection: StrategyDecision | None = None,
    ) -> Run:
        rationale = (
            selection.reason
            if selection is not None
            else self._strategy_rationale(run.request, strategy)
        )
        started = Run.model_validate(
            run.model_dump()
            | {
                "status": transition_run(run.status, RunStatus.RUNNING),
                "strategy": strategy,
                "started_at": utc_now(),
            }
        )
        async with self._store.transaction():
            await self._store.append_event(
                run.run_id,
                EventType.STRATEGY_SELECTED,
                {
                    "strategy": strategy.value,
                    "routing_policy": run.request.routing_policy.value,
                    "rationale": rationale,
                },
            )
            await self._store.append_event(
                run.run_id,
                EventType.CONTROLLER_DECISION,
                payload_from_model(
                    "decision",
                    ControllerDecision(
                        run_id=run.run_id,
                        action=ControllerAction.SELECT_STRATEGY,
                        rationale=rationale,
                        to_strategy=strategy,
                    ),
                ),
            )
            await self._store.update_run(started)
            await self._store.append_event(run.run_id, EventType.RUN_STARTED, {})
        return started

    @staticmethod
    def _strategy_rationale(request: NativeRunRequest, strategy: ExecutionStrategy) -> str:
        if request.routing_policy is RoutingPolicy.MANUAL:
            return f"the caller pinned {strategy.value}"
        if strategy is ExecutionStrategy.PLANNED:
            return "a committed work graph requires PLANNED execution"
        if strategy is ExecutionStrategy.PROGRESSIVE:
            return "evidence-gated revision requires PROGRESSIVE execution"
        return (
            f"{strategy.value} is the weakest sufficient control level for a single "
            "text request"
        )

    async def _finish_run(
        self, run: Run, status: RunStatus, error: ErrorInfo | None = None
    ) -> Run:
        now = utc_now()
        stored = await self._store.get_run(run.run_id)
        metrics = stored.metrics
        if run.started_at is not None:
            metrics = metrics.model_copy(
                update={
                    "wall_clock_ms": max(
                        0, int((now - run.started_at).total_seconds() * 1000)
                    )
                }
            )
        changes: dict[str, object] = {
            "status": transition_run(run.status, status),
            "completed_at": now,
            "usage": stored.usage,
            "metrics": metrics,
        }
        if run.started_at is None:
            # A run cancelled before dispatch still needs a start marker: the
            # domain model requires one for every status past PENDING.
            changes["started_at"] = now
        if error is not None:
            changes["error"] = error
        finished = Run.model_validate(run.model_dump() | changes)
        payload: dict[str, JsonValue] = {}
        if error is not None:
            payload["error"] = error.model_dump(mode="json")
        async with self._store.transaction():
            await self._store.update_run(finished)
            await self._store.append_event(
                run.run_id, _TERMINAL_RUN_EVENTS[status], payload
            )
        return finished

    # --- DIRECT -----------------------------------------------------------------

    async def _resume_running_run(
        self,
        run: Run,
        *,
        execution_scope: ExecutionScope | None,
    ) -> Run:
        """Resume one direct work unit from Store-approved Agent facts."""
        if run.strategy is not ExecutionStrategy.DIRECT:
            return run
        units = await self._store.list_work_units(run.run_id, graph_version=run.graph_version)
        running_units = tuple(
            unit for unit in units if unit.status is WorkUnitStatus.RUNNING
        )
        if len(running_units) != 1:
            return run
        work_unit = running_units[0]
        attempts = await self._store.list_attempts(work_unit.work_unit_id)
        if not attempts:
            return run
        worker = self._worker_for(ModelRole.WORKER, execution_scope=execution_scope)
        state = await worker.load_resume_state(attempts[-1].attempt_id)
        if state.action in (ResumeAction.WAIT, ResumeAction.BLOCK):
            return run
        context = build_worker_context(work_unit, instructions=run.request.instructions)
        try:
            result = await worker.resume(
                run=run,
                work_unit=work_unit,
                context=context,
                state=state,
            )
        except AttemptNotAllowedError:
            return run
        return await self._settle_direct_worker(run, work_unit, worker, result)

    async def _execute_direct(
        self, run: Run, *, execution_scope: ExecutionScope | None = None
    ) -> Run:
        """One work unit, one attempt. Budget is checked before and after dispatch."""
        work_unit = await self._create_direct_work_unit(run)
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.READY)

        # preflight: deadline and attempt count before any provider call
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            # READY cannot transition to FAILED; cancel the unit instead
            await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
            return await self._settle_run(run, [WorkUnitStatus.CANCELLED], error=preflight)

        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.RUNNING)

        worker = self._worker_for(ModelRole.WORKER, execution_scope=execution_scope)
        context = build_worker_context(work_unit, instructions=run.request.instructions)
        try:
            result = await worker.execute(run=run, work_unit=work_unit, context=context)
        except AttemptNotAllowedError:
            # Cancellation landed between starting the unit and dispatching it.
            await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
            return await self._settle_run(run, [WorkUnitStatus.CANCELLED])

        return await self._settle_direct_worker(run, work_unit, worker, result)

    async def _settle_direct_worker(
        self,
        run: Run,
        work_unit: WorkUnit,
        worker: Worker,
        result: WorkerResult,
    ) -> Run:
        """Persist metrics and settle a direct worker result exactly once."""
        result = await self._persist_worker_metrics(run.run_id, result, worker.profile)

        if result.paused:
            return await self._store.get_run(run.run_id)

        if not result.succeeded or result.artifact is None:
            await self._advance_work_unit(
                work_unit, WorkUnitStatus.FAILED, error=result.error
            )
            return await self._settle_run(
                run, [WorkUnitStatus.FAILED], error=result.error
            )

        postflight = await self._postflight_budget(run)
        if postflight is not None:
            await self._advance_work_unit(
                work_unit, WorkUnitStatus.FAILED, error=postflight
            )
            return await self._settle_run(
                run, [WorkUnitStatus.FAILED], error=postflight
            )

        report = await RuleVerifier().verify_and_record(
            self._store,
            result.artifact,
            plan_for_output(work_unit.output),
        )
        evidence_ids = await self._record_evidence_events(
            run, work_unit, result.artifact.artifact_id
        )
        accepted = report.result is VerificationResult.PASS
        rationale = report.summary()
        verify_error = (
            None
            if accepted
            else ErrorInfo(
                category=ErrorCategory.VERIFICATION_FAILED,
                message=rationale,
            )
        )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.ACCEPT_ARTIFACT
                if accepted
                else ControllerAction.REJECT_ARTIFACT
            ),
            rationale=rationale,
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        outcome = WorkUnitStatus.SUCCEEDED if accepted else WorkUnitStatus.FAILED
        await self._advance_work_unit(work_unit, outcome, error=verify_error)
        return await self._settle_run(run, [outcome], error=verify_error)

    async def _execute_cascade(
        self,
        run: Run,
        chain: CascadeChain,
        workers: tuple[Worker, ...],
    ) -> Run:
        """Run the finite CASCADE chain until verification passes or it is exhausted."""
        work_unit = await self._create_work_unit(run, name="cascade")
        work_unit = await self._advance_work_unit(work_unit, WorkUnitStatus.READY)
        last_error: ErrorInfo | None = None
        pending_escalation: (
            tuple[ModelProfile, ModelProfile, str, tuple[str, ...]] | None
        ) = None

        for attempt_index, (profile, worker) in enumerate(
            zip(chain, workers, strict=True), start=1
        ):
            preflight = await self._preflight_budget(run)
            if preflight is not None:
                terminal = (
                    WorkUnitStatus.CANCELLED
                    if work_unit.status is WorkUnitStatus.READY
                    else WorkUnitStatus.FAILED
                )
                await self._advance_work_unit(work_unit, terminal, error=preflight)
                return await self._settle_run(run, [terminal], error=preflight)

            current_run = await self._store.get_run(run.run_id)
            if current_run.status is RunStatus.CANCELLING:
                await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
                return await self._settle_run(current_run, [WorkUnitStatus.CANCELLED])

            if pending_escalation is not None:
                from_profile, to_profile, rationale, evidence_ids = pending_escalation
                await self._record_cascade_escalation(
                    run,
                    work_unit,
                    from_profile,
                    to_profile,
                    rationale,
                    evidence_ids=evidence_ids,
                )
                pending_escalation = None

            if work_unit.status is WorkUnitStatus.READY:
                work_unit = await self._advance_work_unit(
                    work_unit, WorkUnitStatus.RUNNING
                )

            context = build_worker_context(
                work_unit, instructions=run.request.instructions
            )
            try:
                result = await worker.execute(
                    run=current_run,
                    work_unit=work_unit,
                    context=context,
                    attempt_index=attempt_index,
                )
            except AttemptNotAllowedError:
                await self._advance_work_unit(work_unit, WorkUnitStatus.CANCELLED)
                return await self._settle_run(
                    current_run, [WorkUnitStatus.CANCELLED]
                )

            result = await self._persist_worker_metrics(run.run_id, result, profile)

            if result.paused:
                return await self._store.get_run(run.run_id)

            if result.succeeded and result.artifact is not None:
                postflight = await self._postflight_budget(run)
                if postflight is not None:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.FAILED, error=postflight
                    )
                    return await self._settle_run(
                        run, [WorkUnitStatus.FAILED], error=postflight
                    )

                report = await RuleVerifier().verify_and_record(
                    self._store,
                    result.artifact,
                    plan_for_output(work_unit.output),
                )
                evidence_ids = await self._record_evidence_events(
                    run, work_unit, result.artifact.artifact_id
                )
                policy = decide_cascade(
                    verification_result=report.result,
                    has_next_profile=attempt_index < len(chain),
                )

                if policy.disposition is CascadeDisposition.ACCEPT:
                    action = ControllerAction.ACCEPT_ARTIFACT
                    verify_error = None
                else:
                    action = ControllerAction.REJECT_ARTIFACT
                    verify_error = ErrorInfo(
                        category=ErrorCategory.VERIFICATION_FAILED,
                        message=report.summary(),
                    )
                decision = ControllerDecision(
                    run_id=run.run_id,
                    action=action,
                    rationale=report.summary(),
                    work_unit_id=work_unit.work_unit_id,
                    evidence_ids=evidence_ids,
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.CONTROLLER_DECISION,
                    payload_from_model("decision", decision),
                )

                if policy.disposition is CascadeDisposition.ACCEPT:
                    work_unit = await self._advance_work_unit(
                        work_unit, WorkUnitStatus.SUCCEEDED
                    )
                    return await self._settle_run(run, [WorkUnitStatus.SUCCEEDED])
                last_error = verify_error
                current_run = await self._store.get_run(run.run_id)
                if current_run.status is RunStatus.CANCELLING:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.CANCELLED
                    )
                    return await self._settle_run(
                        current_run, [WorkUnitStatus.CANCELLED]
                    )
                if policy.disposition is CascadeDisposition.ESCALATE:
                    pending_escalation = (
                        profile,
                        chain[attempt_index],
                        policy.rationale,
                        evidence_ids,
                    )
                    continue
            else:
                last_error = result.error
                current_run = await self._store.get_run(run.run_id)
                if current_run.status is RunStatus.CANCELLING:
                    await self._advance_work_unit(
                        work_unit, WorkUnitStatus.CANCELLED
                    )
                    return await self._settle_run(
                        current_run, [WorkUnitStatus.CANCELLED]
                    )
                policy = decide_cascade(
                    provider_retryable=provider_failure_is_retryable(result.error),
                    has_next_profile=attempt_index < len(chain),
                )
                if policy.disposition is CascadeDisposition.ESCALATE:
                    pending_escalation = (
                        profile,
                        chain[attempt_index],
                        policy.rationale,
                        (),
                    )
                    continue
            break

        await self._advance_work_unit(
            work_unit, WorkUnitStatus.FAILED, error=last_error
        )
        return await self._settle_run(run, [WorkUnitStatus.FAILED], error=last_error)

    async def _execute_planned(
        self,
        run: Run,
        planner: Planner,
        worker: Worker,
    ) -> Run:
        """Plan, compile, commit, and execute one bounded graph to a terminal run."""
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return await self._finish_run(run, RunStatus.FAILED, error=preflight)

        planner_result, postflight = await self._execute_planner_propose(run, planner)
        if postflight is not None:
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=postflight,
            )
        current = await self._store.get_run(run.run_id)
        if current.status is RunStatus.CANCELLING:
            return await self._finish_run(current, RunStatus.CANCELLED)

        target_graph_version = run.graph_version + 1
        compiled = self._compile_planner_result(
            planner_result,
            run_id=run.run_id,
            graph_version=target_graph_version,
        )
        if self._can_repair_plan(planner_result, compiled):
            assert isinstance(compiled, PlanRejection)
            repair_budget_error = await self._preflight_budget(
                await self._store.get_run(run.run_id)
            )
            if repair_budget_error is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=repair_budget_error,
                )
            feedback = "; ".join(compiled.reasons)
            planner_result, postflight = await self._execute_planner_propose(
                run,
                planner,
                repair_feedback=feedback,
            )
            if postflight is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=postflight,
                )
            compiled = self._compile_planner_result(
                planner_result,
                run_id=run.run_id,
                graph_version=target_graph_version,
            )
        committed = await self.commit_plan(
            run.run_id,
            compiled,
            target_graph_version=target_graph_version,
        )
        if isinstance(committed, PlanRejection):
            error = planner_result.error or ErrorInfo(
                category=ErrorCategory.INVALID_REQUEST,
                message="plan_rejected: " + "; ".join(committed.reasons),
            )
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=error,
            )

        scheduler = Scheduler()

        async def execute_unit(unit: WorkUnit) -> WaveOutcome:
            return await self._execute_planned_work_unit(
                run.run_id,
                unit,
                worker,
            )

        # Every dispatched wave terminalizes at least one current-graph unit.
        # The extra iteration is a defensive bound for an immediately empty or
        # blocked frontier; no unbounded scheduling loop is permitted.
        for _ in range(len(committed) + 1):
            await self.dispatch_planned_wave(
                run.run_id,
                scheduler=scheduler,
                execute=execute_unit,
                graph_version=target_graph_version,
                reservation_capacity_key=worker.profile.alias,
            )
            current = await self._store.get_run(run.run_id)
            if current.status.is_terminal:
                return current

        error = ErrorInfo(
            category=ErrorCategory.UNKNOWN,
            message="planned graph did not converge within its bounded wave count",
        )
        return await self._finish_run(
            await self._store.get_run(run.run_id),
            RunStatus.FAILED,
            error=error,
        )

    @staticmethod
    def _compile_planner_result(
        planner_result: PlannerCallResult,
        *,
        run_id: str,
        graph_version: int,
    ) -> CompiledPlan | PlanRejection:
        if planner_result.rejection is not None:
            return planner_result.rejection
        assert isinstance(planner_result.proposal, PlanProposal)
        return compile_plan(
            planner_result.proposal,
            run_id=run_id,
            graph_version=graph_version,
        )

    @staticmethod
    def _can_repair_plan(
        planner_result: PlannerCallResult,
        compiled: CompiledPlan | PlanRejection,
    ) -> bool:
        if not isinstance(compiled, PlanRejection):
            return False
        if isinstance(planner_result.proposal, PlanProposal):
            return True
        return bool(
            planner_result.error is not None
            and planner_result.error.message.startswith("plan_schema_invalid:")
        )

    async def _execute_planned_work_unit(
        self,
        run_id: str,
        work_unit: WorkUnit,
        worker: Worker,
    ) -> WaveOutcome:
        """Execute and deterministically verify one Controller-started graph unit."""
        run = await self._store.get_run(run_id)
        if run.status is RunStatus.CANCELLING:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="run cancellation prevented planned dispatch",
                )
            )
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return WaveOutcome.failure(preflight)

        dependencies: list[DependencyArtifact] = []
        for dependency_id in work_unit.depends_on:
            dependency = await self._store.get_work_unit(dependency_id)
            artifacts = await self._store.list_artifacts(dependency_id)
            if not artifacts:
                return WaveOutcome.failure(
                    ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message=(
                            "succeeded dependency has no persisted artifact: "
                            f"{dependency_id}"
                        ),
                    )
                )
            dependencies.extend(
                DependencyArtifact(
                    work_unit_name=dependency.name,
                    artifact_name=artifact.name,
                    content=artifact.content,
                )
                for artifact in artifacts
            )

        context = build_worker_context(
            work_unit,
            instructions=run.request.instructions,
            dependencies=tuple(dependencies),
        )
        try:
            result = await worker.execute(
                run=run,
                work_unit=work_unit,
                context=context,
            )
        except AttemptNotAllowedError:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="run cancellation prevented planned attempt",
                )
            )
        result = await self._persist_worker_metrics(run.run_id, result, worker.profile)
        if result.paused:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="tool call is awaiting approval",
                )
            )
        if not result.succeeded or result.artifact is None:
            return WaveOutcome.failure(
                result.error
                or ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="planned worker failed without a classified error",
                )
            )

        postflight = await self._postflight_budget(run)
        if postflight is not None:
            return WaveOutcome.failure(postflight)

        report = await RuleVerifier().verify_and_record(
            self._store,
            result.artifact,
            plan_for_output(work_unit.output),
        )
        evidence_ids = await self._record_evidence_events(
            run,
            work_unit,
            result.artifact.artifact_id,
        )
        accepted = report.result is VerificationResult.PASS
        error = None
        if not accepted:
            error = ErrorInfo(
                category=ErrorCategory.VERIFICATION_FAILED,
                message=report.summary(),
            )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.ACCEPT_ARTIFACT
                if accepted
                else ControllerAction.REJECT_ARTIFACT
            ),
            rationale=report.summary(),
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        if error is not None:
            return WaveOutcome.failure(error)

        attempt_calls = await self._store.list_tool_calls_for_attempt(
            result.attempt.attempt_id
        )
        successful_write_calls = tuple(
            call
            for call in attempt_calls
            if call.status is ToolCallStatus.SUCCEEDED
            and call.effect is ToolEffect.WRITE
        )
        attempt_call_ids = {call.call_id for call in attempt_calls}
        persisted_change_sets = await self._store.list_change_sets(run_id=run_id)
        attempt_change_sets = tuple(
            change_set
            for change_set in persisted_change_sets
            if change_set.tool_call_id in attempt_call_ids
        )
        if len(attempt_change_sets) > 1:
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="one planned worker produced multiple ChangeSets",
                )
            )
        if len(attempt_change_sets) != len(successful_write_calls):
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="successful WRITE tool result has no unique ChangeSet",
                )
            )
        if not attempt_change_sets:
            return WaveOutcome.success()

        change_set = attempt_change_sets[0]
        write_call = successful_write_calls[0]
        if (
            change_set.run_id != run_id
            or change_set.tool_call_id != write_call.call_id
            or change_set.base_snapshot_id != write_call.snapshot_id
            or change_set.patch.base_snapshot_id != write_call.snapshot_id
        ):
            return WaveOutcome.failure(
                ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="ChangeSet does not match the successful WRITE call",
                )
            )
        return WaveOutcome.success(change_set)

    async def _execute_progressive(
        self,
        run: Run,
        planner: Planner,
        worker: Worker,
    ) -> Run:
        """Execute immutable Progressive rounds until evidence or budget settles."""
        preflight = await self._preflight_budget(run)
        if preflight is not None:
            return await self._finish_run(run, RunStatus.FAILED, error=preflight)

        planner_result, postflight = await self._execute_planner_propose(run, planner)
        if postflight is not None:
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=postflight,
            )
        current = await self._store.get_run(run.run_id)
        if current.status is RunStatus.CANCELLING:
            return await self._finish_run(current, RunStatus.CANCELLED)
        target_graph_version = run.graph_version + 1
        proposal: PlanProposal | PlanRejection
        if planner_result.rejection is not None:
            proposal = planner_result.rejection
        else:
            assert isinstance(planner_result.proposal, PlanProposal)
            proposal = planner_result.proposal
        compiled = (
            proposal
            if isinstance(proposal, PlanRejection)
            else compile_plan(
                proposal,
                run_id=run.run_id,
                graph_version=target_graph_version,
            )
        )
        committed = await self.commit_plan(
            run.run_id,
            compiled,
            target_graph_version=target_graph_version,
        )
        if isinstance(committed, PlanRejection):
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message=f"{committed.summary}: " + "; ".join(committed.reasons),
            )
            return await self._finish_run(
                await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=error,
            )

        workspace_id = await self._create_progressive_workspace(run)
        base_snapshot_id = await self._create_progressive_snapshot(
            run,
            workspace_id=workspace_id,
            round_index=0,
            graph_version=target_graph_version,
            kind="base",
        )
        scheduler = Scheduler()
        revision_count = 0
        round_index = 0
        round_change_set_ids: list[str] = []
        predecessor_round_id: str | None = None
        revision_reason: str | None = None
        previous_global_report: GlobalVerificationReport | None = None
        max_rounds = (
            (run.request.budget.max_plan_revisions + 1)
            if run.request.budget.max_plan_revisions is not None
            else 1
        )
        for _ in range(max_rounds):
            round_change_set_ids = []
            graph = await self._store.list_work_units(
                run.run_id,
                graph_version=(await self._store.get_run(run.run_id)).graph_version,
            )
            converged = False
            for _ in range(len(graph) + 1):
                current = await self._store.get_run(run.run_id)
                if current.status.is_terminal:
                    if current.status is RunStatus.CANCELLED:
                        await self._persist_progressive_round(
                            run,
                            round_index=round_index,
                            graph_version=current.graph_version,
                            base_snapshot_id=base_snapshot_id,
                            change_set_ids=tuple(round_change_set_ids),
                            status=RoundStatus.CANCELLED,
                            predecessor_round_id=predecessor_round_id,
                            revision_reason=revision_reason,
                            failure_reason="the Progressive run was cancelled",
                        )
                    return current
                graph = await self._store.list_work_units(
                    run.run_id, graph_version=current.graph_version
                )
                frontier = compute_frontier(
                    graph, graph_version=current.graph_version
                )
                if frontier.ready:
                    ready_units = tuple(
                        unit
                        for unit in graph
                        if unit.work_unit_id in set(frontier.ready)
                    )
                    capacity = run.request.budget.max_concurrency or 1
                    profile_capacity = self._profile_capacity(worker.profile.alias)
                    if profile_capacity is not None:
                        capacity = min(capacity, profile_capacity)
                    coordination = Coordinator().plan(
                        ready_units,
                        effects={
                            unit.work_unit_id: resolve_work_unit_effect(unit)
                            for unit in ready_units
                        },
                        capacity=max(1, capacity),
                    )
                    # A read-only Progressive batch is evaluated against one
                    # immutable graph snapshot. The coordinator's first batch
                    # determines this finite scheduler admission pass.
                    wave_capacity = max(
                        1,
                        len(coordination.batches[0].work_unit_ids),
                    )
                    wave_work_unit_ids = coordination.batches[0].work_unit_ids
                else:
                    wave_capacity = None
                    wave_work_unit_ids = ()
                wave_result = await self.dispatch_planned_wave(
                    run.run_id,
                    scheduler=scheduler,
                    execute=lambda unit: self._execute_planned_work_unit(
                        run.run_id, unit, worker
                    ),
                    graph_version=current.graph_version,
                    max_concurrency=wave_capacity,
                    eligible_work_unit_ids=wave_work_unit_ids,
                    reservation_capacity_key=worker.profile.alias,
                    settle_run=False,
                    finalize_cancel=False,
                )
                for change_set in wave_result.change_sets:
                    if change_set.change_set_id not in round_change_set_ids:
                        round_change_set_ids.append(change_set.change_set_id)
                current = await self._store.get_run(run.run_id)
                if current.status is RunStatus.CANCELLING:
                    await self._persist_progressive_round(
                        run,
                        round_index=round_index,
                        graph_version=current.graph_version,
                        base_snapshot_id=base_snapshot_id,
                        change_set_ids=(),
                        status=RoundStatus.CANCELLED,
                        predecessor_round_id=predecessor_round_id,
                        revision_reason=revision_reason,
                        failure_reason="the Progressive run was cancelled",
                    )
                    return await self._finish_run(current, RunStatus.CANCELLED)
                if current.status.is_terminal:
                    if current.status is RunStatus.CANCELLED:
                        await self._persist_progressive_round(
                            run,
                            round_index=round_index,
                            graph_version=current.graph_version,
                            base_snapshot_id=base_snapshot_id,
                            change_set_ids=tuple(round_change_set_ids),
                            status=RoundStatus.CANCELLED,
                            predecessor_round_id=predecessor_round_id,
                            revision_reason=revision_reason,
                            failure_reason="the Progressive run was cancelled",
                        )
                    return current
                if await self._store.list_tool_calls(
                    run.run_id,
                    statuses=[ToolCallStatus.AWAITING_APPROVAL],
                ):
                    return current
                graph = await self._store.list_work_units(
                    run.run_id, graph_version=current.graph_version
                )
                frontier = compute_frontier(
                    graph, graph_version=current.graph_version
                )
                if not frontier.ready and not frontier.waiting:
                    converged = True
                    break
            if not converged:
                failure_reason = "Progressive graph did not converge within its wave bound"
                await self._persist_progressive_round(
                    run,
                    round_index=round_index,
                    graph_version=current.graph_version,
                    base_snapshot_id=base_snapshot_id,
                    change_set_ids=tuple(round_change_set_ids),
                    status=RoundStatus.FAILED,
                    predecessor_round_id=predecessor_round_id,
                    revision_reason=revision_reason,
                    failure_reason=failure_reason,
                )
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message=failure_reason,
                    ),
                )

            graph = await self._store.list_work_units(
                run.run_id, graph_version=current.graph_version
            )
            failure_error: ErrorInfo | None
            planned_round: ProgressiveRound | None = None
            all_units_succeeded = all(
                unit.status is WorkUnitStatus.SUCCEEDED for unit in graph
            )
            read_only_round = (
                all_units_succeeded
                and not round_change_set_ids
                and not any(unit.writes() for unit in graph)
            )
            if all_units_succeeded and round_change_set_ids:
                round_id = new_round_id()
                planned_round = await self._persist_progressive_round(
                    run,
                    round_id=round_id,
                    round_index=round_index,
                    graph_version=current.graph_version,
                    base_snapshot_id=base_snapshot_id,
                    change_set_ids=tuple(round_change_set_ids),
                    status=RoundStatus.PLANNED,
                    predecessor_round_id=predecessor_round_id,
                    revision_reason=revision_reason,
                )
                if self._progressive_merge_roots is None:
                    failure_error = ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="Progressive merge roots are not available",
                    )
                    await self._store.update_round(
                        planned_round.model_copy(
                            update={
                                "status": RoundStatus.FAILED,
                                "failure_reason": failure_error.message,
                                "completed_at": utc_now(),
                            }
                        )
                    )
                    return await self._finish_run(
                        await self._store.get_run(run.run_id),
                        RunStatus.FAILED,
                        error=failure_error,
                    )
                try:
                    base_root, staged_roots, staging_root = (
                        await self._progressive_merge_roots(planned_round)
                    )
                    merge_result = await self.merge_progressive_round(
                        planned_round.round_id,
                        base_root=base_root,
                        staged_roots=staged_roots,
                        staging_root=staging_root,
                        verify=None,
                    )
                except Exception:
                    failure_error = ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="Progressive merge failed before producing a candidate",
                    )
                    await self._store.update_round(
                        planned_round.model_copy(
                            update={
                                "status": RoundStatus.FAILED,
                                "failure_reason": failure_error.message,
                                "completed_at": utc_now(),
                            }
                        )
                    )
                    return await self._finish_run(
                        await self._store.get_run(run.run_id),
                        RunStatus.FAILED,
                        error=failure_error,
                    )
                if (
                    merge_result.status is not MergeStatus.MERGED
                    or merge_result.merged_snapshot_id is None
                ):
                    failure_error = ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="Progressive merge did not produce a merged candidate",
                    )
                    await self._store.update_round(
                        planned_round.model_copy(
                            update={
                                "status": RoundStatus.FAILED,
                                "failure_reason": failure_error.message,
                                "completed_at": utc_now(),
                            }
                        )
                    )
                    return await self._finish_run(
                        await self._store.get_run(run.run_id),
                        RunStatus.FAILED,
                        error=failure_error,
                    )
                candidate_snapshot_id = merge_result.merged_snapshot_id
                report = await self._progressive_global_report(
                    run,
                    current,
                    graph,
                    round_id=round_id,
                    base_snapshot_id=base_snapshot_id,
                    candidate_snapshot_id=candidate_snapshot_id,
                    change_set_ids=tuple(round_change_set_ids),
                )
                comparison = compare_rounds(
                    previous_global_report,
                    report,
                    base_round_id=(
                        None
                        if previous_global_report is None
                        else previous_global_report.round_id
                    ),
                    candidate_round_id=round_id,
                )
                if report.result is VerificationResult.PASS:
                    try:
                        promoted = self.promote_merge(
                            merge_result.model_copy(update={"verified": True}),
                            merge_result.staging_root.with_name(
                                f"{merge_result.staging_root.name}.promoted-"
                                f"{candidate_snapshot_id}"
                            ),
                        )
                    except MergeError:
                        failure_error = ErrorInfo(
                            category=ErrorCategory.UNKNOWN,
                            message="Progressive candidate promotion failed",
                        )
                        await self._store.update_round(
                            planned_round.model_copy(
                                update={
                                    "status": RoundStatus.FAILED,
                                    "failure_reason": failure_error.message,
                                    "completed_at": utc_now(),
                                }
                            )
                        )
                        return await self._finish_run(
                            await self._store.get_run(run.run_id),
                            RunStatus.FAILED,
                            error=failure_error,
                        )
                    if not promoted.promoted:
                        failure_error = ErrorInfo(
                            category=ErrorCategory.UNKNOWN,
                            message="Progressive candidate promotion was not confirmed",
                        )
                        await self._store.update_round(
                            planned_round.model_copy(
                                update={
                                    "status": RoundStatus.FAILED,
                                    "failure_reason": failure_error.message,
                                    "completed_at": utc_now(),
                                }
                            )
                        )
                        return await self._finish_run(
                            await self._store.get_run(run.run_id),
                            RunStatus.FAILED,
                            error=failure_error,
                        )
                    await self._store.update_round(
                        planned_round.model_copy(
                            update={
                                "status": RoundStatus.VERIFIED,
                                "merged_snapshot_id": promoted.merged_snapshot_id,
                                "evidence_ids": report.evidence_ids,
                                "completed_at": utc_now(),
                            }
                        )
                    )
                    await self._record_progressive_verification(
                        run, report, comparison
                    )
                    return await self._settle_planned_run(
                        run.run_id, current.graph_version
                    )
                failure_error = ErrorInfo(
                    category=(
                        ErrorCategory.VERIFICATION_FAILED
                        if report.result is VerificationResult.FAIL
                        else ErrorCategory.UNKNOWN
                    ),
                    message=report.summary(),
                )
            else:
                _, failure_error = await self._progressive_failure_signal(
                    run.run_id, graph
                )
                round_id = new_round_id()
                report = await self._progressive_global_report(
                    run,
                    current,
                    graph,
                    round_id=round_id,
                    base_snapshot_id=base_snapshot_id,
                    candidate_snapshot_id=(
                        base_snapshot_id if read_only_round else None
                    ),
                    change_set_ids=tuple(round_change_set_ids),
                )
                comparison = compare_rounds(
                    previous_global_report,
                    report,
                    base_round_id=(
                        None
                        if previous_global_report is None
                        else previous_global_report.round_id
                    ),
                    candidate_round_id=round_id,
                )
                if all_units_succeeded and not round_change_set_ids and not read_only_round:
                    failure_error = ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message="Progressive write round produced no ChangeSets",
                )
                if read_only_round and report.result is VerificationResult.PASS:
                    await self._persist_progressive_round(
                        run,
                        round_id=round_id,
                        round_index=round_index,
                        graph_version=current.graph_version,
                        base_snapshot_id=base_snapshot_id,
                        status=RoundStatus.VERIFIED,
                        predecessor_round_id=predecessor_round_id,
                        revision_reason=revision_reason,
                        merged_snapshot_id=base_snapshot_id,
                        evidence_ids=report.evidence_ids,
                    )
                    await self._record_progressive_verification(
                        run, report, comparison
                    )
                    return await self._settle_planned_run(
                        run.run_id, current.graph_version
                    )
                if read_only_round:
                    failure_error = ErrorInfo(
                        category=(
                            ErrorCategory.VERIFICATION_FAILED
                            if report.result is VerificationResult.FAIL
                            else ErrorCategory.UNKNOWN
                        ),
                        message=report.summary(),
                    )
                if failure_error is None and report.result is not VerificationResult.PASS:
                    failure_error = ErrorInfo(
                        category=(
                            ErrorCategory.VERIFICATION_FAILED
                            if report.result is VerificationResult.FAIL
                            else ErrorCategory.UNKNOWN
                        ),
                        message=report.summary(),
                    )
            failed_round_reason = (
                failure_error.message
                if failure_error is not None
                else report.summary()
            )
            if planned_round is None:
                failed_round = await self._persist_progressive_round(
                    run,
                    round_id=round_id,
                    round_index=round_index,
                    graph_version=current.graph_version,
                    base_snapshot_id=base_snapshot_id,
                    change_set_ids=tuple(round_change_set_ids),
                    status=RoundStatus.FAILED,
                    predecessor_round_id=predecessor_round_id,
                    revision_reason=revision_reason,
                    failure_reason=failed_round_reason,
                )
            else:
                failed_round = await self._store.update_round(
                    planned_round.model_copy(
                        update={
                            "status": RoundStatus.FAILED,
                            "failure_reason": failed_round_reason,
                            "completed_at": utc_now(),
                        }
                    )
                )
            await self._record_progressive_verification(run, report, comparison)
            decision_result: VerificationResult | None = report.result
            decision_error: ErrorInfo | None = None
            decision_comparison: RoundComparison | None = comparison
            if (
                report.result is VerificationResult.INCONCLUSIVE
                and failure_error is not None
                and failure_error.category
                in {
                    ErrorCategory.BUDGET_EXCEEDED,
                    ErrorCategory.DEADLINE_EXCEEDED,
                    ErrorCategory.NETWORK,
                    ErrorCategory.RATE_LIMIT,
                    ErrorCategory.TIMEOUT,
                }
            ):
                # A classified provider/budget fact is stronger than the
                # absence of an artifact-level verdict.
                decision_result = None
                decision_error = failure_error
                decision_comparison = None
            decision = decide_revision(
                budget=run.request.budget,
                revision_count=revision_count,
                graph_version=current.graph_version,
                verification_result=decision_result,
                error=decision_error,
                run_status=current.status,
                usage=report.usage,
                comparison=decision_comparison,
            )
            if decision.disposition is RevisionDisposition.STOP:
                return await self._finish_progressive_stop(
                    current,
                    decision,
                    failure_error=failure_error,
                )

            assert decision.reason is not None
            previous_global_report = report
            latest = await self._store.get_run(run.run_id)
            if latest.status is RunStatus.CANCELLING:
                return await self._finish_run(latest, RunStatus.CANCELLED)
            preflight = await self._preflight_budget(latest)
            if preflight is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=preflight,
                )
            revision_result, postflight = await self._execute_planner_revise(
                latest,
                planner,
                base_graph_version=current.graph_version,
                reason=decision.reason,
                feedback=decision.rationale,
            )
            if postflight is not None:
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=postflight,
                )
            if revision_result.rejection is not None:
                rejection_error = revision_result.error or ErrorInfo(
                    category=ErrorCategory.UNKNOWN,
                    message="Planner revision failed without a classified error",
                )
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=rejection_error,
                )
            assert isinstance(revision_result.proposal, PlanRevision)
            committed_revision = await self.commit_revision(
                run.run_id,
                revision_result.proposal,
            )
            if isinstance(committed_revision, PlanRejection):
                return await self._finish_run(
                    await self._store.get_run(run.run_id),
                    RunStatus.FAILED,
                    error=ErrorInfo(
                        category=ErrorCategory.UNKNOWN,
                        message=(
                            f"{committed_revision.summary}: "
                            + "; ".join(committed_revision.reasons)
                        ),
                    ),
                )
            round_index += 1
            predecessor_round_id = failed_round.round_id
            revision_reason = decision.reason.value
            next_graph_version = (await self._store.get_run(run.run_id)).graph_version
            base_snapshot_id = await self._create_progressive_snapshot(
                run,
                workspace_id=workspace_id,
                round_index=round_index,
                graph_version=next_graph_version,
                kind="base",
            )
            revision_count += 1

        return await self._finish_run(
            await self._store.get_run(run.run_id),
                RunStatus.FAILED,
                error=ErrorInfo(
                    category=ErrorCategory.BUDGET_EXCEEDED,
                    message="the Progressive revision loop exhausted its hard bound",
                ),
            )

    async def _create_progressive_workspace(self, run: Run) -> str:
        """Create a server-owned ledger workspace for text-only rounds."""
        workspace = Workspace(
            workspace_id=new_workspace_id(),
            owner_id=self._settings.service_principal,
            alias=f"progressive-{run.run_id.removeprefix('run_')}",
            source=WorkspaceSource(
                source_type=WorkspaceSourceType.SERVER_ALIAS,
                server_alias="progressive-internal",
            ),
            created_at=utc_now(),
        )
        await self._store.create_workspace(workspace)
        return workspace.workspace_id

    async def _create_progressive_snapshot(
        self,
        run: Run,
        *,
        workspace_id: str,
        round_index: int,
        graph_version: int,
        kind: str,
        change_set_ids: Sequence[str] = (),
    ) -> str:
        """Persist an immutable manifest scanned from the actual workspace root."""
        if kind not in {"base", "candidate"}:
            raise StateError("Progressive snapshot kind is unsupported")
        if kind == "candidate" and change_set_ids:
            raise StateError(
                "Progressive candidate requires a real merged workspace root"
            )
        graph = await self._store.list_work_units(
            run.run_id,
            graph_version=graph_version,
        )
        if not graph:
            raise StateError("Progressive snapshot requires a persisted graph")
        with tempfile.TemporaryDirectory(prefix="prp-progressive-host-") as root:
            root_path = Path(root)

            workspace = await self._store.get_workspace(
                workspace_id,
                owner_id=self._settings.service_principal,
            )
            server_alias = workspace.source.server_alias
            if server_alias is None:
                raise StateError("Progressive workspace source is not a server alias")
            resolver = WorkspaceResolver(
                WorkspaceRootMapping.model_validate({server_alias: str(root_path)})
            )
            with resolver.resolve(
                workspace,
                owner_id=self._settings.service_principal,
            ) as resolved:
                manifest = resolved.snapshot_manifest()
        timestamp = utc_now()
        snapshot = Snapshot(
            snapshot_id=new_snapshot_id(),
            workspace_id=workspace_id,
            status=SnapshotStatus.READY,
            created_at=timestamp,
            completed_at=timestamp,
        )
        persisted = await self._store.create_snapshot(
            snapshot,
            manifest,
            owner_id=self._settings.service_principal,
        )
        return persisted.snapshot_id

    async def _persist_progressive_round(
        self,
        run: Run,
        *,
        round_id: str | None = None,
        round_index: int,
        graph_version: int,
        base_snapshot_id: str,
        change_set_ids: tuple[str, ...] = (),
        status: RoundStatus,
        predecessor_round_id: str | None,
        revision_reason: str | None,
        merged_snapshot_id: str | None = None,
        evidence_ids: tuple[str, ...] = (),
        failure_reason: str | None = None,
    ) -> ProgressiveRound:
        """Persist one closed immutable round fact after its work is settled."""
        timestamp = utc_now()
        progressive_round = ProgressiveRound(
            round_id=new_round_id() if round_id is None else round_id,
            run_id=run.run_id,
            round_index=round_index,
            graph_version=graph_version,
            base_snapshot_id=base_snapshot_id,
            merged_snapshot_id=merged_snapshot_id,
            change_set_ids=change_set_ids,
            evidence_ids=evidence_ids,
            status=status,
            revision_of_round_id=predecessor_round_id,
            revision_reason=revision_reason,
            failure_reason=(None if failure_reason is None else failure_reason[:512]),
            created_at=timestamp,
            completed_at=(None if status is RoundStatus.PLANNED else timestamp),
        )
        return await self._store.create_round(progressive_round)

    async def _progressive_evidence_ids(
        self, graph: tuple[WorkUnit, ...]
    ) -> tuple[str, ...]:
        """Load only persisted Evidence identities for the current graph."""
        evidence_ids: list[str] = []
        seen: set[str] = set()
        for unit in graph:
            for evidence in await self._store.list_evidence(unit.work_unit_id):
                if evidence.evidence_id in seen:
                    continue
                seen.add(evidence.evidence_id)
                evidence_ids.append(evidence.evidence_id)
        return tuple(evidence_ids)

    async def _progressive_global_report(
        self,
        run: Run,
        current: Run,
        graph: tuple[WorkUnit, ...],
        *,
        round_id: str,
        base_snapshot_id: str | None = None,
        candidate_snapshot_id: str | None = None,
        change_set_ids: Sequence[str] = (),
    ) -> GlobalVerificationReport:
        """Build a fresh current-round verdict from candidate-bound facts."""
        final_artifacts: tuple[Artifact, ...] = ()
        if current.final_work_unit_id is not None:
            final_artifacts = await self._store.list_artifacts(
                current.final_work_unit_id
            )

        evidence: list[Evidence] = []
        if current.final_work_unit_id is not None:
            evidence.extend(
                await self._store.list_evidence(current.final_work_unit_id)
            )

        change_sets: list[ChangeSet] = []
        for change_set_id in change_set_ids:
            change_sets.append(await self._store.get_change_set(change_set_id))
        required_checks = [
            GlobalCheckKind.FINAL_ARTIFACT,
            GlobalCheckKind.EVIDENCE,
        ]
        if candidate_snapshot_id is not None:
            required_checks.append(GlobalCheckKind.CANDIDATE)
            if change_set_ids:
                required_checks.append(GlobalCheckKind.CHANGE_SET)

        attempt_count = len(await self._store.list_run_attempts(run.run_id))
        metrics = current.metrics
        return verify_global_round(
            run_id=run.run_id,
            graph_version=current.graph_version,
            round_id=round_id,
            final_artifacts=final_artifacts,
            evidence=tuple(evidence),
            change_sets=tuple(change_sets),
            budget=current.request.budget,
            usage=metrics.usage,
            metrics=metrics,
            attempt_count=attempt_count,
            completed_at=utc_now(),
            base_snapshot_id=base_snapshot_id,
            candidate_snapshot_id=candidate_snapshot_id,
            required_checks=tuple(required_checks),
        )

    async def _record_progressive_verification(
        self,
        run: Run,
        report: GlobalVerificationReport,
        comparison: RoundComparison,
    ) -> None:
        """Record global quality and comparison facts in the event ledger."""
        decision = ControllerDecision(
            run_id=run.run_id,
            action=(
                ControllerAction.ACCEPT_ARTIFACT
                if report.result is VerificationResult.PASS
                else ControllerAction.REJECT_ARTIFACT
            ),
            rationale=report.summary(),
            evidence_ids=report.evidence_ids,
        )
        payload = payload_from_model("decision", decision)
        payload["global_report"] = report.model_dump(mode="json")
        payload["comparison"] = comparison.model_dump(mode="json")
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload,
        )

    @staticmethod
    def _progressive_failure_reason(
        graph: tuple[WorkUnit, ...], failure_error: ErrorInfo | None
    ) -> str:
        if failure_error is not None:
            return f"{failure_error.category.value}: {failure_error.message}"
        failed = tuple(
            unit.work_unit_id
            for unit in graph
            if unit.status
            in (WorkUnitStatus.FAILED, WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED)
        )
        if failed:
            return "Progressive graph failed or blocked: " + ", ".join(failed)
        return "Progressive graph produced no deterministic success evidence"

    async def _progressive_failure_signal(
        self,
        run_id: str,
        graph: tuple[WorkUnit, ...],
    ) -> tuple[VerificationResult | None, ErrorInfo | None]:
        """Recover one current-graph verdict/error without reading model text."""
        failed_ids = {
            unit.work_unit_id
            for unit in graph
            if unit.status is WorkUnitStatus.FAILED
        }
        evidence_results: list[VerificationResult] = []
        for work_unit_id in failed_ids:
            evidence_results.extend(
                row.result
                for row in await self._store.list_evidence(work_unit_id)
            )
        if evidence_results:
            if VerificationResult.FAIL in evidence_results:
                result = VerificationResult.FAIL
            elif VerificationResult.INCONCLUSIVE in evidence_results:
                result = VerificationResult.INCONCLUSIVE
            else:
                result = VerificationResult.PASS
            error = await self._failure_event_error(run_id, failed_ids)
            return result, error
        error = await self._failure_event_error(run_id, failed_ids)
        if error is None and any(unit.status is WorkUnitStatus.BLOCKED for unit in graph):
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message="the current Progressive graph contains blocked work",
            )
        return None, error

    async def _failure_event_error(
        self, run_id: str, work_unit_ids: set[str]
    ) -> ErrorInfo | None:
        for event in reversed(await self._store.list_events(run_id)):
            if event.event_type is not EventType.WORK_UNIT_FAILED:
                continue
            if event.payload.get("work_unit_id") not in work_unit_ids:
                continue
            return ErrorInfo.model_validate(event.payload["error"])
        return None

    async def _finish_progressive_stop(
        self,
        run: Run,
        decision: RevisionDecision,
        *,
        failure_error: ErrorInfo | None,
    ) -> Run:
        if decision.stop_reason is RevisionStopReason.CANCELLED:
            if run.status is RunStatus.CANCELLING:
                return await self._finish_run(run, RunStatus.CANCELLED)
            return run
        if decision.stop_reason is RevisionStopReason.BUDGET:
            error = ErrorInfo(
                category=ErrorCategory.BUDGET_EXCEEDED,
                message=decision.rationale,
            )
        else:
            error = failure_error or ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message=decision.rationale,
            )
        return await self._finish_run(run, RunStatus.FAILED, error=error)

    async def _record_evidence_events(
        self,
        run: Run,
        work_unit: WorkUnit,
        artifact_id: str,
    ) -> tuple[str, ...]:
        """Record ledger entries that reference the Evidence rows just persisted."""
        evidence = tuple(
            row
            for row in await self._store.list_evidence(work_unit.work_unit_id)
            if row.artifact_id == artifact_id
        )
        for row in evidence:
            await self._store.append_event(
                run.run_id,
                EventType.EVIDENCE_RECORDED,
                {
                    "work_unit_id": work_unit.work_unit_id,
                    "evidence_id": row.evidence_id,
                    "result": row.result.value,
                },
            )
        return tuple(row.evidence_id for row in evidence)

    async def _record_cascade_escalation(
        self,
        run: Run,
        work_unit: WorkUnit,
        from_profile: ModelProfile,
        to_profile: ModelProfile,
        rationale: str,
        *,
        evidence_ids: tuple[str, ...],
    ) -> None:
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.ESCALATE_MODEL,
            rationale=rationale,
            work_unit_id=work_unit.work_unit_id,
            evidence_ids=evidence_ids,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        await self._store.append_event(
            run.run_id,
            EventType.STRATEGY_ESCALATED,
            {
                "from_strategy": ExecutionStrategy.CASCADE.value,
                "to_strategy": ExecutionStrategy.CASCADE.value,
                "from_profile": from_profile.alias,
                "to_profile": to_profile.alias,
                "reason": rationale,
            },
        )

    async def _create_direct_work_unit(self, run: Run) -> WorkUnit:
        return await self._create_work_unit(run, name=DIRECT_WORK_UNIT_NAME)

    async def _create_work_unit(self, run: Run, *, name: str) -> WorkUnit:
        work_unit = WorkUnit(
            work_unit_id=new_work_unit_id(),
            run_id=run.run_id,
            graph_version=run.graph_version,
            name=name,
            instruction=run.request.input,
            output=run.request.output,
            created_at=utc_now(),
        )
        async with self._store.transaction():
            await self._store.create_work_unit(work_unit)
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_CREATED,
                {"work_unit_id": work_unit.work_unit_id, "name": work_unit.name},
            )
        return work_unit

    async def _advance_work_unit(
        self,
        work_unit: WorkUnit,
        status: WorkUnitStatus,
        error: ErrorInfo | None = None,
        *,
        reason: str | None = None,
    ) -> WorkUnit:
        updated = WorkUnit.model_validate(
            work_unit.model_dump()
            | {"status": transition_work_unit(work_unit.status, status)}
        )
        payload: dict[str, JsonValue] = {"work_unit_id": work_unit.work_unit_id}
        if status is WorkUnitStatus.FAILED:
            payload["error"] = (
                error.model_dump(mode="json")
                if error is not None
                else ErrorInfo(
                    category=ErrorCategory.UNKNOWN, message="the work unit failed"
                ).model_dump(mode="json")
            )
        if status in (WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED):
            payload["reason"] = reason or f"work unit became {status.value}"
        async with self._store.transaction():
            await self._store.update_work_unit(updated)
            await self._store.append_event(
                work_unit.run_id, _WORK_UNIT_EVENTS[status], payload
            )
        return updated

    async def _settle_run(
        self,
        run: Run,
        work_unit_statuses: list[WorkUnitStatus],
        error: ErrorInfo | None = None,
    ) -> Run:
        """Decide the run outcome from its work units and any cancel request."""
        current = await self._store.get_run(run.run_id)
        if current.status.is_terminal:
            return current
        cancel_requested = current.status is RunStatus.CANCELLING
        outcome = resolve_run_outcome(work_unit_statuses, cancel_requested=cancel_requested)
        return await self._finish_run(
            current, outcome, error if outcome is RunStatus.FAILED else None
        )

    # --- workers ----------------------------------------------------------------

    async def _execute_planner_propose(
        self,
        run: Run,
        planner: Planner,
        *,
        repair_feedback: str | None = None,
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Record one initial Planner call before any user graph is compiled."""
        return await self._execute_planner_call(
            run,
            planner,
            attempt_index=await self._next_planner_attempt_index(run.run_id),
            call=lambda: planner.propose_call(
                run.request,
                repair_feedback=repair_feedback,
            ),
        )

    async def _execute_planner_revise(
        self,
        run: Run,
        planner: Planner,
        *,
        base_graph_version: int,
        reason: PlanRevisionReason,
        feedback: str,
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Record one bounded revision call before committing its replacement graph."""
        return await self._execute_planner_call(
            run,
            planner,
            attempt_index=await self._next_planner_attempt_index(run.run_id),
            call=lambda: planner.revise_call(
                run.request,
                base_graph_version=base_graph_version,
                reason=reason,
                feedback=feedback,
            ),
        )

    async def _next_planner_attempt_index(self, run_id: str) -> int:
        attempts = await self._store.list_run_attempts(run_id)
        return sum(attempt.role is ModelRole.PLANNER for attempt in attempts) + 1

    async def _execute_planner_call(
        self,
        run: Run,
        planner: Planner,
        *,
        attempt_index: int,
        call: Callable[[], Awaitable[PlannerCallResult]],
    ) -> tuple[PlannerCallResult, ErrorInfo | None]:
        """Persist one Planner call and its measured provider facts."""
        work_unit = new_planning_work_unit(run.run_id)
        started_at = utc_now()
        attempt = Attempt(
            attempt_id=new_attempt_id(),
            run_id=run.run_id,
            work_unit_id=work_unit.work_unit_id,
            attempt_index=attempt_index,
            role=ModelRole.PLANNER,
            model=planner.profile.model_ref,
            status=transition_attempt(AttemptStatus.PENDING, AttemptStatus.RUNNING),
            created_at=started_at,
            started_at=started_at,
        )
        async with self._store.transaction():
            await self._store.create_work_unit(work_unit)
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_CREATED,
                {
                    "work_unit_id": work_unit.work_unit_id,
                    "name": work_unit.name,
                    "graph_version": work_unit.graph_version,
                },
            )
            reservation = await self._store.reserve_reservation(
                ReservationRequest(
                    run_id=run.run_id,
                    work_unit_id=work_unit.work_unit_id,
                    dispatch_key=f"{work_unit.work_unit_id}:{attempt_index}:PLANNER",
                    capacity_key=planner.profile.alias,
                ),
                capacity_limit=planner.profile.max_concurrency,
            )
            await self._store.append_event(
                run.run_id,
                EventType.WORK_UNIT_STARTED,
                {"work_unit_id": work_unit.work_unit_id},
            )
            await self._store.create_attempt(attempt)
            await self._store.append_event(
                run.run_id,
                EventType.ATTEMPT_STARTED,
                {
                    "work_unit_id": attempt.work_unit_id,
                    "attempt_id": attempt.attempt_id,
                    "model": attempt.model.identifier,
                    "role": attempt.role.value,
                    "attempt_index": attempt.attempt_index,
                },
            )

        try:
            result = await call()
        except BaseException:
            unknown = Attempt.model_validate(
                attempt.model_dump()
                | {
                    "status": transition_attempt(
                        attempt.status, AttemptStatus.UNKNOWN
                    ),
                    "completed_at": utc_now(),
                }
            )
            cancelled = WorkUnit.model_validate(
                work_unit.model_dump()
                | {
                    "status": transition_work_unit(
                        work_unit.status, WorkUnitStatus.CANCELLED
                    )
                }
            )
            async with self._store.transaction():
                await self._store.update_attempt(unknown)
                await self._store.update_work_unit(cancelled)
                await self._store.append_event(
                    run.run_id,
                    EventType.ATTEMPT_UNKNOWN,
                    {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                    },
                )
                await self._store.append_event(
                    run.run_id,
                    EventType.WORK_UNIT_CANCELLED,
                    {"work_unit_id": work_unit.work_unit_id},
                )
                await self._store.settle_reservation(
                    reservation.reservation_id,
                    measured_usage=None,
                )
            raise

        terminal_status = (
            AttemptStatus.SUCCEEDED if result.succeeded else AttemptStatus.FAILED
        )
        error = result.error
        if terminal_status is AttemptStatus.FAILED and error is None:
            error = ErrorInfo(
                category=ErrorCategory.UNKNOWN,
                message="Planner call failed without a classified error",
            )
        completed_at = utc_now()
        completed_attempt = Attempt.model_validate(
            attempt.model_dump()
            | {
                "status": transition_attempt(attempt.status, terminal_status),
                "provider_request_id": result.provider_request_id,
                "usage": result.usage,
                "cost": planner.profile.cost_for_usage(result.usage),
                "error": error,
                "completed_at": completed_at,
            }
        )
        completed_unit_status = (
            WorkUnitStatus.SUCCEEDED
            if terminal_status is AttemptStatus.SUCCEEDED
            else WorkUnitStatus.FAILED
        )
        completed_unit = WorkUnit.model_validate(
            work_unit.model_dump()
            | {"status": transition_work_unit(work_unit.status, completed_unit_status)}
        )
        async with self._store.transaction():
            await self._store.update_attempt(completed_attempt)
            await self._store.update_work_unit(completed_unit)
            await self._store.append_event(
                run.run_id,
                EventType.ATTEMPT_SUCCEEDED
                if terminal_status is AttemptStatus.SUCCEEDED
                else EventType.ATTEMPT_FAILED,
                (
                    {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                    }
                    if error is None
                    else {
                        "work_unit_id": attempt.work_unit_id,
                        "attempt_id": attempt.attempt_id,
                        "error": error.model_dump(mode="json"),
                    }
                ),
            )
            await self._store.append_event(
                run.run_id,
                _WORK_UNIT_EVENTS[completed_unit_status],
                (
                    {"work_unit_id": work_unit.work_unit_id}
                    if error is None
                    else {
                        "work_unit_id": work_unit.work_unit_id,
                        "error": error.model_dump(mode="json"),
                    }
                ),
            )
            if result.usage is not None:
                total = await self._store.add_run_usage(run.run_id, result.usage)
                await self._store.append_event(
                    run.run_id,
                    EventType.USAGE_UPDATED,
                    {"usage": total.model_dump(mode="json")},
                )
            await self._store.record_run_metrics(
                run.run_id,
                usage=result.usage,
                cost=planner.profile.cost_for_usage(result.usage),
                usage_already_accumulated=result.usage is not None,
            )
            await self._store.settle_reservation(
                reservation.reservation_id,
                measured_usage=result.usage,
            )

        postflight = await self._postflight_budget(run)
        return result, postflight

    async def _persist_worker_metrics(
        self, run_id: str, result: WorkerResult, profile: ModelProfile
    ) -> WorkerResult:
        """Persist exact profile-priced cost after Worker-owned attempt writes."""
        cost = profile.cost_for_usage(result.attempt.usage)
        attempt = result.attempt.model_copy(update={"cost": cost})
        async with self._store.transaction():
            await self._store.update_attempt(attempt)
            await self._store.record_run_metrics(
                run_id,
                usage=attempt.usage,
                cost=cost,
                usage_already_accumulated=attempt.usage is not None,
            )
        return result.model_copy(update={"attempt": attempt})

    def _planner(self) -> Planner:
        profile = self._settings.require_profile(ModelRole.PLANNER)
        adapter = self._adapters.get(profile.alias)
        if adapter is None:
            raise ProviderError(
                f"no adapter is registered for model alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return Planner(adapter, profile)

    def _worker_for(
        self, role: ModelRole, *, execution_scope: ExecutionScope | None = None
    ) -> Worker:
        profile = self._settings.require_profile(role)
        return self._worker_for_profile(profile, execution_scope=execution_scope)

    def _worker_for_profile(
        self,
        profile: ModelProfile,
        *,
        execution_scope: ExecutionScope | None = None,
    ) -> Worker:
        adapter = self._adapters.get(profile.alias)
        if adapter is None:
            raise ProviderError(
                f"no adapter is registered for model alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        tool_executor = self._tool_executor
        if execution_scope is not None and self._tool_executor_provider is not None:
            tool_executor = self._tool_executor_provider(execution_scope)
        return Worker(
            self._store,
            adapter,
            profile,
            tool_executor=tool_executor,
            max_tool_rounds=self._max_tool_rounds,
            execution_scope=execution_scope,
        )

    # --- budget helpers ---------------------------------------------------------

    async def _preflight_budget(self, run: Run) -> ErrorInfo | None:
        """Check deadline, attempts, and reached tokens before provider dispatch."""
        budget = run.request.budget
        now = utc_now()

        deadline_decision = check_deadline(budget, now)
        if not deadline_decision.allowed:
            assert deadline_decision.error is not None
            return await self._record_budget_stop(run, deadline_decision.error)

        attempts = await self._store.list_run_attempts(run.run_id)
        attempt_decision = check_attempt_budget(budget, attempt_count=len(attempts))
        if not attempt_decision.allowed:
            assert attempt_decision.error is not None
            return await self._record_budget_stop(run, attempt_decision.error)

        token_decision = check_token_budget_preflight(
            budget, await self._store.get_run_usage(run.run_id)
        )
        if not token_decision.allowed:
            assert token_decision.error is not None
            return await self._record_budget_stop(run, token_decision.error)

        return None

    async def _preflight_budget_with_reservations(
        self, run: Run
    ) -> ErrorInfo | None:
        """Check admission against attempts already held by this run."""
        held = await self._store.list_reservations(
            run.run_id, statuses=[ReservationStatus.HELD]
        )
        attempt_decision = check_attempt_budget(
            run.request.budget,
            attempt_count=len(await self._store.list_run_attempts(run.run_id))
            + sum(reservation.request.attempt_units for reservation in held),
        )
        if not attempt_decision.allowed:
            assert attempt_decision.error is not None
            return await self._record_budget_stop(run, attempt_decision.error)
        return await self._preflight_budget(run)

    async def _postflight_budget(self, run: Run) -> ErrorInfo | None:
        """Check token budget after usage has been recorded for the last attempt."""
        budget = run.request.budget
        usage = await self._store.get_run_usage(run.run_id)
        token_decision = check_token_budget_postflight(budget, usage)
        if not token_decision.allowed:
            assert token_decision.error is not None
            return await self._record_budget_stop(run, token_decision.error)
        return None

    async def _record_budget_stop(self, run: Run, error: Exception) -> ErrorInfo:
        """Persist the budget stop decision and BUDGET_EXHAUSTED event."""
        from prp_runtime.domain.errors import BudgetError

        assert isinstance(error, BudgetError)
        error_info = ErrorInfo(
            category=ErrorCategory.BUDGET_EXCEEDED, message=error.detail.message
        )
        decision = ControllerDecision(
            run_id=run.run_id,
            action=ControllerAction.STOP_ON_BUDGET,
            rationale=error.detail.message,
        )
        await self._store.append_event(
            run.run_id,
            EventType.CONTROLLER_DECISION,
            payload_from_model("decision", decision),
        )
        await self._store.append_event(
            run.run_id,
            EventType.BUDGET_EXHAUSTED,
            {"error": error_info.model_dump(mode="json")},
        )
        return error_info
