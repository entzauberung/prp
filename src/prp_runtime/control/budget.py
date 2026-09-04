"""Deterministic budget enforcement.

All functions here are pure: they receive a snapshot of current consumption and
the declared ceilings, and return a decision. No IO, no side effects, no clock
access except through an injected ``now`` parameter.

Boundary semantics depend on the call phase:
- A ceiling of ``None`` means no limit for that dimension.
- Before dispatch, consumption *equal to* the ceiling means the limit is
  reached, so the next attempt is stopped.
- After dispatch, consumption *equal to* the ceiling permits the current
  result; only consumption above the ceiling is stopped.

``Usage.total_tokens`` and ``Usage.strong_model_tokens`` are measured values.
An ``attempt_count`` of zero means no attempt has run yet. A ``None`` ceiling
for a dimension is never reached, so the dimension is always ALLOW.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum, unique

from prp_runtime.domain.errors import BudgetError, ErrorCode
from prp_runtime.domain.models import Budget, Usage

__all__ = [
    "BudgetDecision",
    "BudgetOutcome",
    "check_attempt_budget",
    "check_deadline",
    "check_role_dispatch",
    "check_token_budget",
    "check_token_budget_postflight",
    "check_token_budget_preflight",
]


@unique
class BudgetOutcome(StrEnum):
    ALLOW = "ALLOW"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class BudgetDecision:
    """The result of one budget check."""

    outcome: BudgetOutcome
    dimension: str
    error: BudgetError | None = None

    @property
    def allowed(self) -> bool:
        return self.outcome is BudgetOutcome.ALLOW


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_token_budget(budget: Budget, usage: Usage) -> BudgetDecision:
    """Check whether another provider dispatch is within the token ceiling.

    This compatibility entry point has preflight semantics: consumption equal
    to a ceiling stops the next provider call.
    """
    return check_token_budget_preflight(budget, usage)


def check_token_budget_preflight(budget: Budget, usage: Usage) -> BudgetDecision:
    """Stop a new provider dispatch when measured usage reached a ceiling."""
    return _check_token_budget(budget, usage, stop_at_ceiling=True)


def check_token_budget_postflight(budget: Budget, usage: Usage) -> BudgetDecision:
    """Accept the current result at a ceiling and stop only strict overage."""
    return _check_token_budget(budget, usage, stop_at_ceiling=False)


def _check_token_budget(
    budget: Budget,
    usage: Usage,
    *,
    stop_at_ceiling: bool,
) -> BudgetDecision:
    """Apply the shared total/strong token checks for one call phase.

    Both total and strong-model tokens are checked; the tighter constraint wins.
    A dimension with ``None`` ceiling is always ALLOW for that dimension.
    """
    total_stopped = budget.max_total_tokens is not None and (
        usage.total_tokens >= budget.max_total_tokens
        if stop_at_ceiling
        else usage.total_tokens > budget.max_total_tokens
    )
    if total_stopped:
        assert budget.max_total_tokens is not None
        relation = "reached" if stop_at_ceiling else "exceeded"
        return _stop(
            "total_tokens",
            ErrorCode.TOKEN_BUDGET_EXCEEDED,
            f"total tokens {usage.total_tokens} {relation} the ceiling of "
            f"{budget.max_total_tokens}",
        )
    strong_stopped = budget.max_strong_model_tokens is not None and (
        usage.strong_model_tokens >= budget.max_strong_model_tokens
        if stop_at_ceiling
        else usage.strong_model_tokens > budget.max_strong_model_tokens
    )
    if strong_stopped:
        assert budget.max_strong_model_tokens is not None
        relation = "reached" if stop_at_ceiling else "exceeded"
        return _stop(
            "strong_model_tokens",
            ErrorCode.TOKEN_BUDGET_EXCEEDED,
            f"strong-model tokens {usage.strong_model_tokens} {relation} the ceiling of "
            f"{budget.max_strong_model_tokens}",
        )
    return _allow("tokens")


def check_attempt_budget(budget: Budget, attempt_count: int) -> BudgetDecision:
    """Check whether the number of completed attempts is within the declared ceiling.

    ``attempt_count`` is the number of attempts already completed (including the
    one whose result is being evaluated). A ceiling of 1 allows exactly one
    attempt.
    """
    if budget.max_attempts is not None and attempt_count >= budget.max_attempts:
        return _stop(
            "max_attempts",
            ErrorCode.ATTEMPT_BUDGET_EXCEEDED,
            f"attempt count {attempt_count} reached the ceiling of {budget.max_attempts}",
        )
    return _allow("attempts")


def check_role_dispatch(
    budget: Budget,
    usage: Usage,
    *,
    attempt_count: int,
    now: datetime,
    deterministic: bool = False,
    context_window_tokens: int | None = None,
    max_output_tokens: int | None = None,
    declared_input_tokens: int | None = None,
    declared_output_tokens: int | None = None,
    timeout_seconds: float | None = None,
) -> BudgetDecision:
    """Stop an over-budget role call before provider dispatch.

    Deterministic Analyzer/Verifier completion does not consume a provider
    budget and must not fabricate usage. Declared input/output bounds are
    ceilings, never estimates recorded as measured tokens.
    """
    if deterministic:
        return _allow("deterministic")
    deadline = check_deadline(budget, now)
    if not deadline.allowed:
        return deadline
    attempts = check_attempt_budget(budget, attempt_count)
    if not attempts.allowed:
        return attempts
    tokens = check_token_budget_preflight(budget, usage)
    if not tokens.allowed:
        return tokens
    if (
        declared_output_tokens is not None
        and max_output_tokens is not None
        and declared_output_tokens > max_output_tokens
    ):
        return _stop(
            "max_output_tokens",
            ErrorCode.TOKEN_BUDGET_EXCEEDED,
            f"declared output tokens {declared_output_tokens} exceed the role "
            f"ceiling of {max_output_tokens}",
        )
    planned_tokens = usage.input_tokens
    if declared_input_tokens is not None:
        planned_tokens = max(planned_tokens, declared_input_tokens)
    if declared_output_tokens is not None:
        planned_tokens += declared_output_tokens
    if (
        context_window_tokens is not None
        and planned_tokens > context_window_tokens
    ):
        return _stop(
            "context_window_tokens",
            ErrorCode.TOKEN_BUDGET_EXCEEDED,
            f"declared role context {planned_tokens} exceeds the window of "
            f"{context_window_tokens}",
        )
    if timeout_seconds is not None and usage.elapsed_ms >= int(timeout_seconds * 1000):
        return _stop(
            "timeout",
            ErrorCode.DEADLINE_EXCEEDED,
            f"measured provider elapsed {usage.elapsed_ms}ms reached the role "
            f"timeout of {timeout_seconds}s",
        )
    return _allow("role_dispatch")


def check_deadline(budget: Budget, now: datetime) -> BudgetDecision:
    """Check whether ``now`` is before the declared deadline.

    The deadline is inclusive: a ``now`` equal to the deadline means time is up.
    ``now`` must be timezone-aware; the deadline stored in ``Budget`` is always
    UTC-aware by the domain model validator.
    """
    if budget.deadline is not None and now >= budget.deadline:
        return _stop(
            "deadline",
            ErrorCode.DEADLINE_EXCEEDED,
            f"current time {now.isoformat()} reached the deadline "
            f"{budget.deadline.isoformat()}",
        )
    return _allow("deadline")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _allow(dimension: str) -> BudgetDecision:
    return BudgetDecision(outcome=BudgetOutcome.ALLOW, dimension=dimension)


def _stop(dimension: str, code: ErrorCode, message: str) -> BudgetDecision:
    return BudgetDecision(
        outcome=BudgetOutcome.STOP,
        dimension=dimension,
        error=BudgetError(message, code=code),
    )
