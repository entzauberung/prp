"""Deterministic verification.

Verification decides by inspecting artifacts. A check that cannot decide reports
``INCONCLUSIVE``; there is no general purpose model-as-judge here.
"""

from prp_runtime.verification.rules import (
    SUPPORTED_SCHEMA_KEYWORDS,
    VerificationCheck,
    VerificationPlan,
    VerificationRule,
    check_json_schema,
    plan_for_output,
)
from prp_runtime.verification.verifier import (
    CheckOutcome,
    RuleVerifier,
    VerificationReport,
    aggregate,
)

__all__ = [
    "SUPPORTED_SCHEMA_KEYWORDS",
    "CheckOutcome",
    "RuleVerifier",
    "VerificationCheck",
    "VerificationPlan",
    "VerificationReport",
    "VerificationRule",
    "aggregate",
    "check_json_schema",
    "plan_for_output",
]
