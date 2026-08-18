"""AUTO routing decision table and monotonic escalation tests."""

import pytest

from prp_runtime.control.routing import (
    RouteRejectionCode,
    RoutingFacts,
    StrategyDecision,
    escalate,
    facts_from_request,
    route,
)
from prp_runtime.domain.enums import ExecutionStrategy, RoutingPolicy
from prp_runtime.domain.models import Budget, NativeRunRequest, RoutingIntent


def request(
    *,
    routing_policy: RoutingPolicy = RoutingPolicy.AUTO,
    strategy: ExecutionStrategy | None = None,
    routing: RoutingIntent | None = None,
    budget: Budget | None = None,
) -> NativeRunRequest:
    return NativeRunRequest(
        input="perform the requested operation",
        routing_policy=routing_policy,
        strategy=strategy,
        routing=routing,
        budget=budget or Budget(),
    )


@pytest.mark.parametrize(
    "intent",
    [
        None,
        RoutingIntent(),
        RoutingIntent(requires_cascade=True),
        RoutingIntent(requires_plan=True),
        RoutingIntent(requires_revision=True),
        RoutingIntent(desired_parallelism=3),
    ],
)
def test_public_routing_intent_maps_one_to_one_without_retryable_failure(
    intent: RoutingIntent | None,
) -> None:
    facts = facts_from_request(request(routing=intent))

    assert facts == RoutingFacts(
        requires_cascade=intent.requires_cascade if intent else False,
        requires_plan=intent.requires_plan if intent else False,
        requires_revision=intent.requires_revision if intent else False,
        desired_parallelism=intent.desired_parallelism if intent else None,
        retryable_failure=False,
    )


@pytest.mark.parametrize(
    ("facts", "expected"),
    [
        (RoutingFacts(), ExecutionStrategy.DIRECT),
        (RoutingFacts(requires_cascade=True), ExecutionStrategy.CASCADE),
        (RoutingFacts(requires_plan=True), ExecutionStrategy.PLANNED),
        (RoutingFacts(desired_parallelism=2), ExecutionStrategy.PLANNED),
        (RoutingFacts(requires_revision=True), ExecutionStrategy.PROGRESSIVE),
    ],
)
def test_auto_selects_the_weakest_sufficient_strategy(
    facts: RoutingFacts,
    expected: ExecutionStrategy,
) -> None:
    budget = Budget(max_plan_revisions=1) if expected is ExecutionStrategy.PROGRESSIVE else Budget()
    decision = route(request(budget=budget), facts=facts)

    assert decision.strategy is expected
    assert decision.accepted is True
    assert decision.rejection is None
    assert "perform" not in decision.reason


@pytest.mark.parametrize("strategy", list(ExecutionStrategy))
def test_manual_returns_the_pinned_strategy_without_rewriting_it(
    strategy: ExecutionStrategy,
) -> None:
    decision = route(
        request(routing_policy=RoutingPolicy.MANUAL, strategy=strategy),
        facts=RoutingFacts(requires_revision=True, desired_parallelism=4),
    )

    assert decision.strategy is strategy
    assert decision.reason == f"MANUAL pins {strategy.value}"


def test_unavailable_target_is_a_structured_rejection_not_direct_fallback() -> None:
    decision = route(
        request(),
        facts=RoutingFacts(requires_plan=True),
        available_strategies=frozenset({ExecutionStrategy.DIRECT}),
    )

    assert decision.strategy is None
    assert decision.rejection is not None
    assert decision.rejection.code is RouteRejectionCode.STRATEGY_UNAVAILABLE


def test_strategy_decision_rejects_an_empty_outcome() -> None:
    with pytest.raises(ValueError, match="accepted or rejected"):
        StrategyDecision(reason="no decision")


def test_budget_rejects_progressive_and_parallelism_before_selection() -> None:
    progressive = route(
        request(budget=Budget(max_plan_revisions=0)),
        facts=RoutingFacts(requires_revision=True),
    )
    parallel = route(
        request(budget=Budget(max_concurrency=1)),
        facts=RoutingFacts(desired_parallelism=2),
    )

    assert progressive.rejection is not None
    assert progressive.rejection.code is RouteRejectionCode.BUDGET_INSUFFICIENT
    assert parallel.rejection is not None
    assert parallel.rejection.code is RouteRejectionCode.BUDGET_INSUFFICIENT


def test_same_facts_produce_identical_serializable_decisions() -> None:
    facts = RoutingFacts(requires_cascade=True, retryable_failure=True)
    first = route(request(), facts=facts)
    second = route(request(), facts=facts)

    assert first == second
    assert first.model_dump_json() == second.model_dump_json()


@pytest.mark.parametrize(
    ("current", "facts", "expected"),
    [
        (ExecutionStrategy.DIRECT, RoutingFacts(requires_cascade=True), ExecutionStrategy.CASCADE),
        (ExecutionStrategy.CASCADE, RoutingFacts(requires_plan=True), ExecutionStrategy.PLANNED),
        (
            ExecutionStrategy.PLANNED,
            RoutingFacts(requires_revision=True),
            ExecutionStrategy.PROGRESSIVE,
        ),
        (ExecutionStrategy.PROGRESSIVE, RoutingFacts(), ExecutionStrategy.PROGRESSIVE),
    ],
)
def test_auto_escalation_is_strictly_one_way(
    current: ExecutionStrategy,
    facts: RoutingFacts,
    expected: ExecutionStrategy,
) -> None:
    budget = Budget(max_plan_revisions=1) if facts.requires_revision else Budget()
    decision = escalate(current, request(budget=budget), facts=facts)

    assert decision.strategy is expected
    assert decision.rejected is False


def test_manual_escalation_is_rejected_without_changing_current_strategy() -> None:
    decision = escalate(
        ExecutionStrategy.DIRECT,
        request(routing_policy=RoutingPolicy.MANUAL, strategy=ExecutionStrategy.DIRECT),
        facts=RoutingFacts(requires_revision=True),
    )

    assert decision.strategy is ExecutionStrategy.DIRECT
    assert decision.rejection is not None
    assert decision.rejection.code is RouteRejectionCode.MANUAL_ESCALATION_FORBIDDEN


def test_auto_never_downgrades_a_stronger_current_strategy() -> None:
    decision = escalate(
        ExecutionStrategy.PROGRESSIVE,
        request(),
        facts=RoutingFacts(),
    )

    assert decision == StrategyDecision(
        strategy=ExecutionStrategy.PROGRESSIVE,
        reason="AUTO retains PROGRESSIVE; no stronger control is required",
    )
