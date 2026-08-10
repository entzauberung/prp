"""The deterministic rule verifier.

The verifier turns an artifact plus a plan into evidence. It writes evidence and
nothing else: it never changes run, work unit or attempt state, because deciding
what a verdict means belongs to the controller.
"""

from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    Artifact,
    DomainModel,
    Evidence,
    EvidenceKind,
    NonBlankText,
    VerificationResult,
    new_evidence_id,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.rules import (
    VerificationCheck,
    VerificationPlan,
    VerificationRule,
    check_json_schema,
)

__all__ = ["CheckOutcome", "RuleVerifier", "VerificationReport", "aggregate"]


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
