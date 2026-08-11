"""Targeted tests for the budget enforcement pure functions."""

from datetime import UTC, datetime, timedelta

import pytest

from prp_runtime.control.budget import (
    BudgetOutcome,
    check_attempt_budget,
    check_deadline,
    check_token_budget,
    check_token_budget_postflight,
    check_token_budget_preflight,
)
from prp_runtime.domain.errors import ErrorCode
from prp_runtime.domain.models import Budget, Usage

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

T0 = datetime(2026, 8, 11, 12, 0, tzinfo=UTC)
NO_BUDGET = Budget()


def _usage(total: int = 0, strong: int = 0) -> Usage:
    return Usage(input_tokens=total, output_tokens=0, strong_model_tokens=strong)


# ---------------------------------------------------------------------------
# check_token_budget: total_tokens
# ---------------------------------------------------------------------------


def test_token_budget_allows_when_no_ceiling() -> None:
    decision = check_token_budget(NO_BUDGET, _usage(total=999_999))
    assert decision.allowed is True
    assert decision.outcome is BudgetOutcome.ALLOW


def test_token_budget_allows_below_ceiling() -> None:
    budget = Budget(max_total_tokens=100)
    decision = check_token_budget(budget, _usage(total=99))
    assert decision.allowed is True


def test_token_budget_stops_at_ceiling() -> None:
    budget = Budget(max_total_tokens=100)
    decision = check_token_budget(budget, _usage(total=100))
    assert decision.allowed is False
    assert decision.outcome is BudgetOutcome.STOP
    assert decision.error is not None
    assert decision.error.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
    assert "total_tokens" in decision.dimension


def test_token_budget_stops_above_ceiling() -> None:
    budget = Budget(max_total_tokens=100)
    decision = check_token_budget(budget, _usage(total=150))
    assert decision.allowed is False


def test_token_budget_zero_ceiling_stops_immediately() -> None:
    budget = Budget(max_total_tokens=0)
    decision = check_token_budget(budget, _usage(total=0))
    assert decision.allowed is False
    assert decision.error is not None


# ---------------------------------------------------------------------------
# check_token_budget: strong_model_tokens
# ---------------------------------------------------------------------------


def test_strong_token_budget_allows_when_no_ceiling() -> None:
    budget = Budget(max_total_tokens=1000)
    decision = check_token_budget(budget, _usage(total=500, strong=499))
    assert decision.allowed is True


def test_strong_token_budget_allows_below_ceiling() -> None:
    budget = Budget(max_total_tokens=1000, max_strong_model_tokens=50)
    decision = check_token_budget(budget, _usage(total=100, strong=49))
    assert decision.allowed is True


def test_strong_token_budget_stops_at_ceiling() -> None:
    budget = Budget(max_total_tokens=1000, max_strong_model_tokens=50)
    decision = check_token_budget(budget, _usage(total=100, strong=50))
    assert decision.allowed is False
    assert decision.error is not None
    assert decision.error.code is ErrorCode.TOKEN_BUDGET_EXCEEDED
    assert "strong_model_tokens" in decision.dimension


def test_total_budget_checked_before_strong_budget() -> None:
    budget = Budget(max_total_tokens=10, max_strong_model_tokens=10)
    decision = check_token_budget(budget, _usage(total=10, strong=5))
    assert decision.allowed is False
    assert "total_tokens" in decision.dimension


def test_strong_token_ceiling_none_never_fires() -> None:
    budget = Budget(max_total_tokens=1000, max_strong_model_tokens=None)
    decision = check_token_budget(budget, _usage(total=500, strong=500))
    assert decision.allowed is True


# ---------------------------------------------------------------------------
# token phase boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("budget", "usage"),
    [
        (Budget(max_total_tokens=10), _usage(total=10)),
        (
            Budget(max_total_tokens=100, max_strong_model_tokens=5),
            _usage(total=10, strong=5),
        ),
    ],
)
def test_token_budget_preflight_stops_at_ceiling(
    budget: Budget, usage: Usage
) -> None:
    assert check_token_budget_preflight(budget, usage).allowed is False


@pytest.mark.parametrize(
    ("budget", "usage"),
    [
        (Budget(max_total_tokens=10), _usage(total=10)),
        (
            Budget(max_total_tokens=100, max_strong_model_tokens=5),
            _usage(total=10, strong=5),
        ),
    ],
)
def test_token_budget_postflight_allows_exact_ceiling(
    budget: Budget, usage: Usage
) -> None:
    assert check_token_budget_postflight(budget, usage).allowed is True


@pytest.mark.parametrize(
    ("budget", "usage"),
    [
        (Budget(max_total_tokens=10), _usage(total=11)),
        (
            Budget(max_total_tokens=100, max_strong_model_tokens=5),
            _usage(total=10, strong=6),
        ),
    ],
)
def test_token_budget_postflight_stops_above_ceiling(
    budget: Budget, usage: Usage
) -> None:
    decision = check_token_budget_postflight(budget, usage)
    assert decision.allowed is False
    assert decision.error is not None
    assert decision.error.code is ErrorCode.TOKEN_BUDGET_EXCEEDED


# ---------------------------------------------------------------------------
# check_attempt_budget
# ---------------------------------------------------------------------------


def test_attempt_budget_allows_when_no_ceiling() -> None:
    decision = check_attempt_budget(NO_BUDGET, attempt_count=9999)
    assert decision.allowed is True


def test_attempt_budget_allows_below_ceiling() -> None:
    budget = Budget(max_attempts=3)
    assert check_attempt_budget(budget, attempt_count=2).allowed is True


def test_attempt_budget_stops_at_ceiling() -> None:
    budget = Budget(max_attempts=3)
    decision = check_attempt_budget(budget, attempt_count=3)
    assert decision.allowed is False
    assert decision.error is not None
    assert decision.error.code is ErrorCode.ATTEMPT_BUDGET_EXCEEDED
    assert "max_attempts" in decision.dimension


def test_attempt_budget_stops_above_ceiling() -> None:
    budget = Budget(max_attempts=1)
    assert check_attempt_budget(budget, attempt_count=2).allowed is False


def test_attempt_budget_ceiling_one_allows_first_attempt() -> None:
    budget = Budget(max_attempts=1)
    assert check_attempt_budget(budget, attempt_count=0).allowed is True


def test_attempt_budget_ceiling_one_stops_after_first_attempt() -> None:
    budget = Budget(max_attempts=1)
    assert check_attempt_budget(budget, attempt_count=1).allowed is False


# ---------------------------------------------------------------------------
# check_deadline
# ---------------------------------------------------------------------------


def test_deadline_allows_when_no_deadline() -> None:
    decision = check_deadline(NO_BUDGET, now=T0)
    assert decision.allowed is True


def test_deadline_allows_before_deadline() -> None:
    budget = Budget(deadline=T0 + timedelta(seconds=1))
    decision = check_deadline(budget, now=T0)
    assert decision.allowed is True


def test_deadline_stops_at_deadline() -> None:
    deadline = T0 + timedelta(seconds=60)
    budget = Budget(deadline=deadline)
    decision = check_deadline(budget, now=deadline)
    assert decision.allowed is False
    assert decision.error is not None
    assert decision.error.code is ErrorCode.DEADLINE_EXCEEDED
    assert "deadline" in decision.dimension


def test_deadline_stops_after_deadline() -> None:
    deadline = T0
    budget = Budget(deadline=deadline)
    decision = check_deadline(budget, now=T0 + timedelta(seconds=1))
    assert decision.allowed is False


def test_deadline_none_never_fires() -> None:
    budget = Budget(deadline=None)
    far_future = T0 + timedelta(days=3650)
    assert check_deadline(budget, now=far_future).allowed is True


# ---------------------------------------------------------------------------
# BudgetDecision contract
# ---------------------------------------------------------------------------


def test_allow_decision_has_no_error() -> None:
    decision = check_token_budget(NO_BUDGET, _usage())
    assert decision.error is None
    assert decision.allowed is True


def test_stop_decision_carries_budget_error() -> None:
    budget = Budget(max_attempts=1)
    decision = check_attempt_budget(budget, attempt_count=1)
    assert decision.error is not None
    from prp_runtime.domain.errors import BudgetError
    assert isinstance(decision.error, BudgetError)


def test_decision_is_immutable() -> None:
    decision = check_token_budget(NO_BUDGET, _usage())
    with pytest.raises(Exception):
        decision.outcome = BudgetOutcome.STOP  # type: ignore[misc]


def test_all_dimensions_allow_on_zero_consumption_no_budget() -> None:
    assert check_token_budget(NO_BUDGET, Usage()).allowed is True
    assert check_attempt_budget(NO_BUDGET, 0).allowed is True
    assert check_deadline(NO_BUDGET, T0).allowed is True
