"""The deterministic rule verifier.

The verifier turns an artifact plus a plan into evidence. It writes evidence and
nothing else: it never changes run, work unit or attempt state, because deciding
what a verdict means belongs to the controller.
"""

from collections.abc import Sequence
from datetime import datetime
from enum import StrEnum, unique

from prp_runtime.analysis.syntax import SyntaxReport
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    Artifact,
    Budget,
    DomainModel,
    Evidence,
    EvidenceKind,
    GlobalCheck,
    GlobalVerificationReport,
    NonBlankText,
    RunMetrics,
    Usage,
    VerificationResult,
    new_evidence_id,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads
from prp_runtime.runtime.conflicts import classify_conflict
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.tools.command import CommandResult
from prp_runtime.verification.rules import (
    VerificationCheck,
    VerificationPlan,
    VerificationRule,
    check_json_schema,
)
from prp_runtime.workspace.changes import ChangeSet

__all__ = [
    "CheckOutcome",
    "GlobalCheckKind",
    "GlobalVerifier",
    "GlobalVerificationReport",
    "RuleVerifier",
    "VerificationReport",
    "aggregate",
    "verify_global",
    "verify_global_round",
]


def aggregate(results: tuple[VerificationResult, ...]) -> VerificationResult:
    """Combine check results into one verdict.

    A single failure fails the artifact. Otherwise a single undecidable check
    makes the whole verdict undecidable. An empty plan proves nothing, so it is
    ``INCONCLUSIVE`` rather than a pass.
    """
    if not results:
        return VerificationResult.INCONCLUSIVE
    if any(result is VerificationResult.FAIL for result in results):
        return VerificationResult.FAIL
    if any(result is VerificationResult.INCONCLUSIVE for result in results):
        return VerificationResult.INCONCLUSIVE
    return VerificationResult.PASS


class CheckOutcome(DomainModel):
    """The verdict of one rule."""

    rule: VerificationRule
    result: VerificationResult
    detail: NonBlankText

    @property
    def passed(self) -> bool:
        return self.result.is_pass


class VerificationReport(DomainModel):
    """Every verdict produced for one artifact."""

    run_id: str
    work_unit_id: str
    artifact_id: str
    outcomes: tuple[CheckOutcome, ...] = ()
    result: VerificationResult = VerificationResult.INCONCLUSIVE

    @property
    def passed(self) -> bool:
        return self.result.is_pass

    @property
    def failures(self) -> tuple[CheckOutcome, ...]:
        return tuple(
            outcome for outcome in self.outcomes if outcome.result is VerificationResult.FAIL
        )

    @property
    def undecided(self) -> tuple[CheckOutcome, ...]:
        return tuple(
            outcome
            for outcome in self.outcomes
            if outcome.result is VerificationResult.INCONCLUSIVE
        )

    def summary(self) -> str:
        """A one line, auditable summary of the verdict."""
        counts = {
            state: sum(1 for outcome in self.outcomes if outcome.result is state)
            for state in VerificationResult
        }
        return (
            f"{self.result.value}: {counts[VerificationResult.PASS]} passed, "
            f"{counts[VerificationResult.FAIL]} failed, "
            f"{counts[VerificationResult.INCONCLUSIVE]} undecided"
        )

    def to_evidence(self) -> tuple[Evidence, ...]:
        """Build one evidence row per check."""
        return tuple(
            Evidence(
                evidence_id=new_evidence_id(),
                run_id=self.run_id,
                work_unit_id=self.work_unit_id,
                artifact_id=self.artifact_id,
                kind=EvidenceKind.DETERMINISTIC_CHECK,
                rule=outcome.rule.value,
                result=outcome.result,
                detail=outcome.detail,
            )
            for outcome in self.outcomes
        )


@unique
class GlobalCheckKind(StrEnum):
    """The deterministic fact families accepted by global verification."""

    FINAL_ARTIFACT = "FINAL_ARTIFACT"
    EVIDENCE = "EVIDENCE"
    CANDIDATE = "CANDIDATE"
    CHANGE_SET = "CHANGE_SET"
    AST = "AST"
    COMMAND = "COMMAND"
    BUDGET = "BUDGET"
    METRICS = "METRICS"


def _unique_ids(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return tuple(result)


def _global_check(
    kind: GlobalCheckKind,
    result: VerificationResult,
    detail: str,
    *,
    evidence_ids: Sequence[str] = (),
    fact_ids: Sequence[str] = (),
) -> GlobalCheck:
    return GlobalCheck(
        kind=kind.value,
        result=result,
        detail=detail,
        evidence_ids=_unique_ids(evidence_ids),
        fact_ids=_unique_ids(fact_ids),
    )


def _global_artifact_check(
    artifacts: Sequence[Artifact],
) -> tuple[VerificationResult, str]:
    if not artifacts:
        return VerificationResult.INCONCLUSIVE, "no final artifact was persisted"
    ids = [artifact.artifact_id for artifact in artifacts]
    if len(ids) != len(set(ids)):
        return VerificationResult.FAIL, "final artifact identities are duplicated"
    if any(not artifact.content.strip() for artifact in artifacts):
        return VerificationResult.FAIL, "a final artifact is empty"
    return VerificationResult.PASS, "all final artifacts are persisted and non-empty"


def _global_evidence_check(
    artifacts: Sequence[Artifact], evidence: Sequence[Evidence]
) -> tuple[VerificationResult, str]:
    if not evidence:
        return VerificationResult.INCONCLUSIVE, "no verification Evidence was persisted"
    evidence_ids = [row.evidence_id for row in evidence]
    if len(evidence_ids) != len(set(evidence_ids)):
        return VerificationResult.FAIL, "verification Evidence identities are duplicated"
    artifacts_by_id = {artifact.artifact_id: artifact for artifact in artifacts}
    stale = [
        row.evidence_id
        for row in evidence
        if row.artifact_id not in artifacts_by_id
        or row.run_id != artifacts_by_id[row.artifact_id].run_id
        or row.work_unit_id != artifacts_by_id[row.artifact_id].work_unit_id
    ]
    if stale:
        return (
            VerificationResult.INCONCLUSIVE,
            "Evidence is stale or outside the merged candidate: "
            + ", ".join(stale[:5]),
        )
    artifact_ids = set(artifacts_by_id)
    covered = {row.artifact_id for row in evidence}
    missing = sorted(artifact_ids - covered)
    if missing:
        return (
            VerificationResult.INCONCLUSIVE,
            "final artifacts are missing Evidence: " + ", ".join(missing[:5]),
        )
    result = aggregate(tuple(row.result for row in evidence))
    if result is VerificationResult.FAIL:
        return VerificationResult.FAIL, "at least one persisted Evidence check failed"
    if result is VerificationResult.INCONCLUSIVE:
        return (
            VerificationResult.INCONCLUSIVE,
            "at least one persisted Evidence check is inconclusive",
        )
    return VerificationResult.PASS, "all persisted Evidence checks passed"


def _global_candidate_check(
    *,
    candidate_snapshot_id: str | None,
    round_id: str | None,
    artifact_ids: Sequence[str],
    change_set_ids: Sequence[str],
    evidence_ids: Sequence[str],
) -> tuple[VerificationResult, str]:
    """Require one candidate identity to close the round's evidence linkage."""
    missing: list[str] = []
    if not candidate_snapshot_id or not candidate_snapshot_id.strip():
        missing.append("candidate Snapshot")
    if not round_id or not round_id.strip():
        missing.append("round")
    if not artifact_ids:
        missing.append("final artifact")
    if not evidence_ids:
        missing.append("fresh Evidence")
    if missing:
        return (
            VerificationResult.INCONCLUSIVE,
            "candidate linkage is incomplete: " + ", ".join(missing),
        )
    return (
        VerificationResult.PASS,
        "candidate Snapshot is linked to this round, final artifact, ChangeSets, "
        "and Evidence",
    )


def _global_change_set_check(
    change_sets: Sequence[ChangeSet],
    *,
    run_id: str,
    base_snapshot_id: str | None,
) -> tuple[VerificationResult, str]:
    if not change_sets:
        return VerificationResult.INCONCLUSIVE, "no ChangeSet was persisted"
    ids = [change_set.change_set_id for change_set in change_sets]
    if len(ids) != len(set(ids)):
        return VerificationResult.FAIL, "ChangeSet identities are duplicated"
    if any(change_set.run_id != run_id for change_set in change_sets):
        return VerificationResult.FAIL, "a ChangeSet belongs to another run"
    if base_snapshot_id is not None and any(
        change_set.base_snapshot_id != base_snapshot_id for change_set in change_sets
    ):
        return VerificationResult.FAIL, "ChangeSets do not share the round base Snapshot"
    for index, left in enumerate(change_sets):
        for right in change_sets[index + 1 :]:
            conflict = classify_conflict(left, right)
            if conflict.conflict:
                return VerificationResult.FAIL, conflict.reason
    return VerificationResult.PASS, "ChangeSets share the declared base and have no conflict"


def _global_ast_check(
    reports: Sequence[SyntaxReport],
) -> tuple[VerificationResult, str]:
    if not reports:
        return VerificationResult.INCONCLUSIVE, "no AST report was persisted"
    if any(not report.parse_ok for report in reports):
        return VerificationResult.FAIL, "at least one Python AST report failed to parse"
    if any(report.unknown for report in reports):
        return VerificationResult.INCONCLUSIVE, "at least one AST report is unknown"
    return VerificationResult.PASS, "all AST reports parsed without unknown facts"


def _global_command_check(
    results: Sequence[CommandResult],
) -> tuple[VerificationResult, str]:
    if not results:
        return VerificationResult.INCONCLUSIVE, "no targeted command result was persisted"
    if any(result.timed_out for result in results):
        return VerificationResult.FAIL, "a targeted command timed out"
    if any(result.cancelled for result in results):
        return VerificationResult.FAIL, "a targeted command was cancelled"
    if any(result.exit_code is None for result in results):
        return VerificationResult.INCONCLUSIVE, "a targeted command has no exit code"
    failed = [result.exit_code for result in results if result.exit_code != 0]
    if failed:
        return VerificationResult.FAIL, "a targeted command exited non-zero"
    return VerificationResult.PASS, "all targeted commands exited with code zero"


def _global_budget_check(
    budget: Budget | None,
    *,
    usage: Usage | None,
    attempt_count: int | None,
    completed_at: datetime | None,
) -> tuple[VerificationResult, str]:
    if budget is None:
        return VerificationResult.INCONCLUSIVE, "no budget fact was supplied"
    unknown: list[str] = []
    failures: list[str] = []
    checked = False
    if budget.max_total_tokens is not None:
        checked = True
        if usage is None:
            unknown.append("total tokens")
        elif usage.total_tokens > budget.max_total_tokens:
            failures.append("total token ceiling exceeded")
    if budget.max_strong_model_tokens is not None:
        checked = True
        if usage is None:
            unknown.append("strong model tokens")
        elif usage.strong_model_tokens > budget.max_strong_model_tokens:
            failures.append("strong model token ceiling exceeded")
    if budget.max_attempts is not None:
        checked = True
        if attempt_count is None:
            unknown.append("attempt count")
        elif attempt_count > budget.max_attempts:
            failures.append("attempt ceiling exceeded")
    if budget.deadline is not None:
        checked = True
        if completed_at is None:
            unknown.append("completion time")
        elif completed_at >= budget.deadline:
            failures.append("deadline exceeded")
    if failures:
        return VerificationResult.FAIL, "; ".join(failures)
    if unknown:
        return VerificationResult.INCONCLUSIVE, "missing budget facts: " + ", ".join(unknown)
    if not checked:
        return VerificationResult.PASS, "no finite budget ceiling was declared"
    return VerificationResult.PASS, "all declared budget ceilings were respected"


def _global_metrics_check(
    metrics: RunMetrics | None,
) -> tuple[VerificationResult, str]:
    if metrics is None:
        return VerificationResult.INCONCLUSIVE, "no run metrics were supplied"
    if (
        metrics.usage is None
        and metrics.provider_elapsed_ms is None
        and metrics.wall_clock_ms is None
        and metrics.cost is None
    ):
        return VerificationResult.INCONCLUSIVE, "run metrics contain no measured facts"
    return VerificationResult.PASS, "run metrics contain measured usage, time, or cost facts"


def verify_global_round(
    *,
    run_id: str,
    graph_version: int,
    round_id: str | None = None,
    final_artifacts: Sequence[Artifact] = (),
    evidence: Sequence[Evidence] = (),
    change_sets: Sequence[ChangeSet] = (),
    syntax_reports: Sequence[SyntaxReport] = (),
    command_results: Sequence[CommandResult] = (),
    budget: Budget | None = None,
    usage: Usage | None = None,
    metrics: RunMetrics | None = None,
    attempt_count: int | None = None,
    completed_at: datetime | None = None,
    base_snapshot_id: str | None = None,
    candidate_snapshot_id: str | None = None,
    required_checks: Sequence[GlobalCheckKind] = (
        GlobalCheckKind.FINAL_ARTIFACT,
        GlobalCheckKind.EVIDENCE,
    ),
) -> GlobalVerificationReport:
    """Build one whole-round verdict from persisted public facts only.

    Optional fact families are checked when supplied. A caller can require an
    otherwise absent family through ``required_checks``; absence then remains
    ``INCONCLUSIVE`` instead of becoming an accidental pass.
    """
    required = {GlobalCheckKind(kind) for kind in required_checks}
    checks: list[GlobalCheck] = []
    artifact_ids = _unique_ids([artifact.artifact_id for artifact in final_artifacts])
    evidence_ids = _unique_ids([row.evidence_id for row in evidence])
    change_set_ids = _unique_ids([change_set.change_set_id for change_set in change_sets])
    candidate_required = GlobalCheckKind.CANDIDATE in required

    if final_artifacts or GlobalCheckKind.FINAL_ARTIFACT in required:
        result, detail = _global_artifact_check(final_artifacts)
        checks.append(
            _global_check(
                GlobalCheckKind.FINAL_ARTIFACT,
                result,
                detail,
                fact_ids=artifact_ids,
            )
        )
    if evidence or GlobalCheckKind.EVIDENCE in required:
        result, detail = _global_evidence_check(final_artifacts, evidence)
        checks.append(
            _global_check(
                GlobalCheckKind.EVIDENCE,
                result,
                detail,
                evidence_ids=evidence_ids,
                fact_ids=evidence_ids,
            )
        )
    if candidate_snapshot_id is not None or candidate_required:
        result, detail = _global_candidate_check(
            candidate_snapshot_id=candidate_snapshot_id,
            round_id=round_id,
            artifact_ids=artifact_ids,
            change_set_ids=change_set_ids,
            evidence_ids=evidence_ids,
        )
        checks.append(
            _global_check(
                GlobalCheckKind.CANDIDATE,
                result,
                detail,
                evidence_ids=evidence_ids,
                fact_ids=(
                    (candidate_snapshot_id,) if candidate_snapshot_id else ()
                )
                + ((round_id,) if round_id else ())
                + tuple(change_set_ids)
                + tuple(evidence_ids)
                + tuple(artifact_ids),
            )
        )
    if change_sets or GlobalCheckKind.CHANGE_SET in required:
        result, detail = _global_change_set_check(
            change_sets, run_id=run_id, base_snapshot_id=base_snapshot_id
        )
        checks.append(
            _global_check(
                GlobalCheckKind.CHANGE_SET,
                result,
                detail,
                fact_ids=change_set_ids,
            )
        )
    if syntax_reports or GlobalCheckKind.AST in required:
        result, detail = _global_ast_check(syntax_reports)
        checks.append(
            _global_check(
                GlobalCheckKind.AST,
                result,
                detail,
                fact_ids=tuple(f"syntax:{index}" for index in range(len(syntax_reports))),
            )
        )
    if command_results or GlobalCheckKind.COMMAND in required:
        result, detail = _global_command_check(command_results)
        checks.append(
            _global_check(
                GlobalCheckKind.COMMAND,
                result,
                detail,
                fact_ids=tuple(f"command:{index}" for index in range(len(command_results))),
            )
        )
    if budget is not None or GlobalCheckKind.BUDGET in required:
        result, detail = _global_budget_check(
            budget,
            usage=usage,
            attempt_count=attempt_count,
            completed_at=completed_at,
        )
        checks.append(_global_check(GlobalCheckKind.BUDGET, result, detail))
    if metrics is not None or GlobalCheckKind.METRICS in required:
        result, detail = _global_metrics_check(metrics)
        checks.append(_global_check(GlobalCheckKind.METRICS, result, detail))

    report_result = aggregate(tuple(check.result for check in checks))
    return GlobalVerificationReport(
        run_id=run_id,
        round_id=round_id,
        graph_version=graph_version,
        result=report_result,
        checks=tuple(checks),
        final_artifact_ids=artifact_ids,
        change_set_ids=change_set_ids,
        evidence_ids=evidence_ids,
        syntax_report_count=len(syntax_reports),
        command_result_count=len(command_results),
        usage=usage,
        metrics=metrics,
    )


verify_global = verify_global_round


class GlobalVerifier:
    """Object boundary for callers that prefer an injectable verifier."""

    def verify(self, **facts: object) -> GlobalVerificationReport:
        return verify_global_round(**facts)  # type: ignore[arg-type]


class RuleVerifier:
    """Applies a deterministic plan to an artifact."""

    def verify(self, artifact: Artifact, plan: VerificationPlan) -> VerificationReport:
        """Evaluate every check in the plan against the artifact."""
        outcomes = tuple(self._evaluate(artifact, check) for check in plan)
        return VerificationReport(
            run_id=artifact.run_id,
            work_unit_id=artifact.work_unit_id,
            artifact_id=artifact.artifact_id,
            outcomes=outcomes,
            result=aggregate(tuple(outcome.result for outcome in outcomes)),
        )

    async def verify_and_record(
        self, store: SqliteStore, artifact: Artifact, plan: VerificationPlan
    ) -> VerificationReport:
        """Evaluate the plan and persist the evidence in one transaction.

        Run and work unit state are deliberately untouched.
        """
        report = self.verify(artifact, plan)
        evidence = report.to_evidence()
        if evidence:
            async with store.transaction():
                for row in evidence:
                    await store.add_evidence(row)
        return report

    def _evaluate(self, artifact: Artifact, check: VerificationCheck) -> CheckOutcome:
        match check.rule:
            case VerificationRule.NON_EMPTY_OUTPUT:
                result, detail = self._non_empty(artifact)
            case VerificationRule.OUTPUT_KIND_MATCHES:
                result, detail = self._kind_matches(artifact, check)
            case VerificationRule.VALID_JSON:
                result, detail = self._valid_json(artifact)
            case VerificationRule.MATCHES_JSON_SCHEMA:
                result, detail = self._matches_schema(artifact, check)
            case VerificationRule.REQUIRED_REFERENCES:
                result, detail = self._required_references(artifact, check)
            case VerificationRule.WITHIN_LENGTH_LIMIT:
                result, detail = self._within_length(artifact, check)
            case _:
                # A rule was added to the enum without an implementation here.
                # Refusing is the only honest answer: guessing would fabricate a
                # verdict for a check that never ran.
                raise DomainValidationError(
                    f"verification rule {check.rule!r} has no deterministic implementation",
                    code=ErrorCode.INVALID_REQUEST,
                    field="rule",
                )
        return CheckOutcome(rule=check.rule, result=result, detail=detail)

    @staticmethod
    def _non_empty(artifact: Artifact) -> tuple[VerificationResult, str]:
        if artifact.content.strip():
            return VerificationResult.PASS, "the output contains text"
        return VerificationResult.FAIL, "the output is empty"

    @staticmethod
    def _kind_matches(
        artifact: Artifact, check: VerificationCheck
    ) -> tuple[VerificationResult, str]:
        expected = check.expected_kind
        assert expected is not None  # guaranteed by VerificationCheck
        if artifact.kind is expected:
            return VerificationResult.PASS, f"the output is {expected.value} as required"
        return (
            VerificationResult.FAIL,
            f"the output is {artifact.kind.value} but {expected.value} was required",
        )

    @staticmethod
    def _valid_json(artifact: Artifact) -> tuple[VerificationResult, str]:
        # Standard JSON only: NaN and Infinity cannot be written back out as JSON,
        # so accepting them here would pass an output no JSON reader can consume.
        try:
            strict_json_loads(artifact.content)
        except StrictJsonError as error:
            return VerificationResult.FAIL, f"the output is not valid JSON: {error.reason}"
        return VerificationResult.PASS, "the output is valid JSON"

    @staticmethod
    def _matches_schema(
        artifact: Artifact, check: VerificationCheck
    ) -> tuple[VerificationResult, str]:
        schema = check.json_schema
        assert schema is not None  # guaranteed by VerificationCheck
        return check_json_schema(artifact.content, schema)

    @staticmethod
    def _required_references(
        artifact: Artifact, check: VerificationCheck
    ) -> tuple[VerificationResult, str]:
        missing = [
            reference
            for reference in check.required_references
            if reference not in artifact.content
        ]
        if missing:
            return (
                VerificationResult.FAIL,
                "the output does not reference: " + ", ".join(missing[:5]),
            )
        return (
            VerificationResult.PASS,
            f"the output references all {len(check.required_references)} required inputs",
        )

    @staticmethod
    def _within_length(
        artifact: Artifact, check: VerificationCheck
    ) -> tuple[VerificationResult, str]:
        limit = check.max_characters
        assert limit is not None  # guaranteed by VerificationCheck
        length = len(artifact.content)
        if length <= limit:
            return VerificationResult.PASS, f"the output is {length} of {limit} characters"
        return (
            VerificationResult.FAIL,
            f"the output is {length} characters, over the {limit} character limit",
        )
