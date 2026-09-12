"""Progressive Reasoning Protocol kernel."""

from prp.machines import (
    ATTEMPT_TRANSITIONS,
    RUN_TRANSITIONS,
    STRATEGY_CONTROL_STRENGTH,
    WORK_UNIT_TRANSITIONS,
    can_escalate_strategy,
    can_transition_attempt,
    can_transition_run,
    can_transition_work_unit,
)
from prp.revision import RevisionDecision, decide_reuse, decide_revision
from prp.progressive import ComparisonOutcome, ProgressiveDecision, decide_progressive
from prp.vocabulary import (
    AttemptStatus,
    ExecutionStrategy,
    ReuseDisposition,
    ReuseReason,
    RevisionDisposition,
    RevisionStopReason,
    RunStatus,
    VerificationResult,
    WorkUnitStatus,
)

__all__ = [
    "ATTEMPT_TRANSITIONS",
    "RUN_TRANSITIONS",
    "STRATEGY_CONTROL_STRENGTH",
    "WORK_UNIT_TRANSITIONS",
    "AttemptStatus",
    "ExecutionStrategy",
    "ReuseDisposition",
    "ReuseReason",
    "RevisionDecision",
    "ProgressiveDecision",
    "RevisionDisposition",
    "RevisionStopReason",
    "RunStatus",
    "VerificationResult",
    "WorkUnitStatus",
    "can_escalate_strategy",
    "can_transition_attempt",
    "can_transition_run",
    "can_transition_work_unit",
    "decide_reuse",
    "decide_revision",
    "decide_progressive",
    "ComparisonOutcome",
]

__version__ = "0.0.2"
