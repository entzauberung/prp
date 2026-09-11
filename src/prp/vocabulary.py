"""Closed protocol vocabulary. Unknown values are not representable."""

from enum import StrEnum, unique

__all__ = [
    "AttemptStatus",
    "ExecutionStrategy",
    "ReuseDisposition",
    "ReuseReason",
    "RevisionDisposition",
    "RevisionStopReason",
    "RunStatus",
    "VerificationResult",
    "WorkUnitStatus",
]


@unique
class ExecutionStrategy(StrEnum):
    DIRECT = "DIRECT"
    CASCADE = "CASCADE"
    PLANNED = "PLANNED"
    PROGRESSIVE = "PROGRESSIVE"


@unique
class RunStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    CANCELLING = "CANCELLING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def is_terminal(self) -> bool:
        return self in {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}


@unique
class WorkUnitStatus(StrEnum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    BLOCKED = "BLOCKED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INVALIDATED = "INVALIDATED"

    @property
    def is_terminal(self) -> bool:
        return self in {
            WorkUnitStatus.SUCCEEDED,
            WorkUnitStatus.FAILED,
            WorkUnitStatus.CANCELLED,
            WorkUnitStatus.INVALIDATED,
        }


@unique
class AttemptStatus(StrEnum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            AttemptStatus.SUCCEEDED,
            AttemptStatus.FAILED,
            AttemptStatus.CANCELLED,
            AttemptStatus.INTERRUPTED,
            AttemptStatus.UNKNOWN,
        }


@unique
class VerificationResult(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    INCONCLUSIVE = "INCONCLUSIVE"

    @property
    def is_pass(self) -> bool:
        return self is VerificationResult.PASS


@unique
class RevisionDisposition(StrEnum):
    REVISE = "REVISE"
    STOP = "STOP"


@unique
class RevisionStopReason(StrEnum):
    PASS = "PASS"
    TERMINAL = "TERMINAL"
    CANCELLED = "CANCELLED"
    BUDGET = "BUDGET"
    REVISION_LIMIT = "REVISION_LIMIT"
    NO_REVISION_BUDGET = "NO_REVISION_BUDGET"
    NO_TRIGGER = "NO_TRIGGER"
    NO_GAIN = "NO_GAIN"
    REGRESSION = "REGRESSION"
    INCONCLUSIVE = "INCONCLUSIVE"


@unique
class ReuseDisposition(StrEnum):
    REUSE = "REUSE"
    RECOMPUTE = "RECOMPUTE"


@unique
class ReuseReason(StrEnum):
    ALL_FACTS_MATCH = "ALL_FACTS_MATCH"
    HISTORICAL_UNIT_NOT_SUCCEEDED = "HISTORICAL_UNIT_NOT_SUCCEEDED"
    MISSING_LINEAGE_OR_FINGERPRINT = "MISSING_LINEAGE_OR_FINGERPRINT"
    LINEAGE_CHANGED = "LINEAGE_CHANGED"
    CONTENT_FINGERPRINT_CHANGED = "CONTENT_FINGERPRINT_CHANGED"
    DEPENDENCY_FINGERPRINT_CHANGED = "DEPENDENCY_FINGERPRINT_CHANGED"
