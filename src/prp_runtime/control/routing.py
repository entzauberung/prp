"""Deterministic AUTO strategy routing and one-way escalation."""

from enum import StrEnum, unique

from pydantic import Field, model_validator

from prp_runtime.domain.enums import ExecutionStrategy, ModelRole, RoutingPolicy
from prp_runtime.domain.models import DomainModel, NativeRunRequest
from prp_runtime.domain.transitions import control_strength

__all__ = [
    "RoleDecision",
    "RoleFacts",
    "RouteFacts",
    "RouteRejection",
    "RouteRejectionCode",
    "RoutingFacts",
    "StrategyDecision",
    "escalate",
    "facts_from_request",
    "route",
    "route_role",
]


@unique
class RouteRejectionCode(StrEnum):
    """Stable reasons an otherwise deterministic target cannot be selected."""

    STRATEGY_UNAVAILABLE = "STRATEGY_UNAVAILABLE"
    BUDGET_INSUFFICIENT = "BUDGET_INSUFFICIENT"
    MANUAL_ESCALATION_FORBIDDEN = "MANUAL_ESCALATION_FORBIDDEN"
    DOWNGRADE_FORBIDDEN = "DOWNGRADE_FORBIDDEN"


class RouteRejection(DomainModel):
    """A public routing rejection with no request text or provider details."""

    code: RouteRejectionCode
    reason: str = Field(min_length=1)


class RoutingFacts(DomainModel):
    """Explicit facts supplied by persisted control state or a binding."""

    requires_cascade: bool = False
    requires_plan: bool = False
    requires_revision: bool = False
    desired_parallelism: int | None = Field(default=None, ge=1)
    retryable_failure: bool = False


RouteFacts = RoutingFacts


class RoleFacts(DomainModel):
    """Server-owned facts that select a model role.

    These facts are never taken from a client payload, profile, URL or
    credential. Exactly one purpose must be named.
    """

    planning: bool = False
    work: bool = False
    analysis: bool = False
    verification: bool = False
    deterministic_analysis: bool = False
    deterministic_verification: bool = False

    @model_validator(mode="after")
    def _one_server_purpose(self) -> "RoleFacts":
        purposes = (self.planning, self.work, self.analysis, self.verification)
        if sum(purposes) != 1:
            raise ValueError("role facts must name exactly one server purpose")
        if self.deterministic_analysis and not self.analysis:
            raise ValueError("deterministic analysis requires the analysis purpose")
        if self.deterministic_verification and not self.verification:
            raise ValueError("deterministic verification requires the verification purpose")
        return self


class RoleDecision(DomainModel):
    """One server role choice or a deterministic provider bypass."""

    role: ModelRole | None = None
    deterministic: bool = False
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _outcome_is_closed(self) -> "RoleDecision":
        if self.role is None and not self.deterministic:
            raise ValueError("role decision must select a role or a deterministic bypass")
        if self.deterministic and self.role is not None:
            raise ValueError("deterministic bypass cannot select a provider role")
        if self.role is ModelRole.WORKER and self.deterministic:
            raise ValueError("WORKER cannot be a deterministic alias")
        return self


def route_role(facts: RoleFacts) -> RoleDecision:
    """Select Planner, Worker, Analyzer or Verifier from server facts only."""
    if facts.planning:
        return RoleDecision(
            role=ModelRole.PLANNER,
            reason="server selected PLANNER for plan proposal",
        )
    if facts.work:
        return RoleDecision(
            role=ModelRole.WORKER,
            reason="server selected WORKER for work-unit execution",
        )
    if facts.analysis:
        if facts.deterministic_analysis:
            return RoleDecision(
                deterministic=True,
                reason="deterministic analysis facts suffice without a provider",
            )
        return RoleDecision(
            role=ModelRole.ANALYZER,
            reason="server selected ANALYZER; it is not a Worker alias",
        )
    if facts.deterministic_verification:
        return RoleDecision(
            deterministic=True,
            reason="deterministic verification facts suffice without a provider",
        )
    return RoleDecision(
        role=ModelRole.VERIFIER,
        reason="server selected VERIFIER; it is not a Worker alias",
    )


def facts_from_request(request: NativeRunRequest) -> RoutingFacts:
    """Map public AUTO intent to control facts without consulting models or state."""
    intent = request.routing
    if intent is None:
        return RoutingFacts()
    return RoutingFacts(
        requires_cascade=intent.requires_cascade,
        requires_plan=intent.requires_plan,
        requires_revision=intent.requires_revision,
        desired_parallelism=intent.desired_parallelism,
        retryable_failure=False,
    )


class StrategyDecision(DomainModel):
    """One stable strategy choice or a structured rejection."""

    strategy: ExecutionStrategy | None = None
    reason: str = Field(min_length=1)
    rejection: RouteRejection | None = None

    @model_validator(mode="after")
    def _outcome_is_closed(self) -> "StrategyDecision":
        if self.strategy is None and self.rejection is None:
            raise ValueError("strategy decision must be accepted or rejected")
        return self

    @property
    def accepted(self) -> bool:
        return self.strategy is not None and self.rejection is None

    @property
    def rejected(self) -> bool:
        return self.rejection is not None


_STRENGTH_ORDER: tuple[ExecutionStrategy, ...] = (
    ExecutionStrategy.DIRECT,
    ExecutionStrategy.CASCADE,
    ExecutionStrategy.PLANNED,
    ExecutionStrategy.PROGRESSIVE,
)


def _target_for(facts: RoutingFacts) -> ExecutionStrategy:
    if facts.requires_revision:
        return ExecutionStrategy.PROGRESSIVE
    if facts.requires_plan or (facts.desired_parallelism or 1) > 1:
        return ExecutionStrategy.PLANNED
    if facts.requires_cascade or facts.retryable_failure:
        return ExecutionStrategy.CASCADE
    return ExecutionStrategy.DIRECT


def _unavailable(target: ExecutionStrategy) -> StrategyDecision:
    return StrategyDecision(
        reason=f"AUTO requires {target.value}, but that strategy is unavailable",
        rejection=RouteRejection(
            code=RouteRejectionCode.STRATEGY_UNAVAILABLE,
            reason=f"required strategy {target.value} is not configured",
        ),
    )


def route(
    request: NativeRunRequest,
    *,
    facts: RoutingFacts | None = None,
    available_strategies: frozenset[ExecutionStrategy] = frozenset(
        _STRENGTH_ORDER
    ),
) -> StrategyDecision:
    """Choose the weakest sufficient strategy without model or network calls."""
    if request.routing_policy is RoutingPolicy.MANUAL:
        assert request.strategy is not None
        if request.strategy not in available_strategies:
            return _unavailable(request.strategy)
        return StrategyDecision(
            strategy=request.strategy,
            reason=f"MANUAL pins {request.strategy.value}",
        )

    current_facts = facts or RoutingFacts()
    target = _target_for(current_facts)
    if target is ExecutionStrategy.PROGRESSIVE and (
        request.budget.max_plan_revisions is None
        or request.budget.max_plan_revisions < 1
    ):
        return StrategyDecision(
            reason="AUTO requires a positive max_plan_revisions budget",
            rejection=RouteRejection(
                code=RouteRejectionCode.BUDGET_INSUFFICIENT,
                reason="PROGRESSIVE requires an explicit positive revision budget",
            ),
        )
    if current_facts.desired_parallelism is not None and (
        request.budget.max_concurrency is not None
        and request.budget.max_concurrency < current_facts.desired_parallelism
    ):
        return StrategyDecision(
            reason="AUTO parallelism exceeds the declared concurrency budget",
            rejection=RouteRejection(
                code=RouteRejectionCode.BUDGET_INSUFFICIENT,
                reason="max_concurrency is below the requested parallelism",
            ),
        )
    if target not in available_strategies:
        return _unavailable(target)
    return StrategyDecision(
        strategy=target,
        reason=f"AUTO selects the weakest sufficient strategy: {target.value}",
    )


def escalate(
    current: ExecutionStrategy,
    request: NativeRunRequest,
    *,
    facts: RoutingFacts | None = None,
    available_strategies: frozenset[ExecutionStrategy] = frozenset(
        _STRENGTH_ORDER
    ),
) -> StrategyDecision:
    """Apply AUTO's one-way upgrade rule to a persisted current strategy."""
    if request.routing_policy is RoutingPolicy.MANUAL:
        return StrategyDecision(
            strategy=current,
            reason="MANUAL forbids automatic strategy escalation",
            rejection=RouteRejection(
                code=RouteRejectionCode.MANUAL_ESCALATION_FORBIDDEN,
                reason="a manually pinned strategy cannot be changed",
            ),
        )
    desired = route(
        request,
        facts=facts,
        available_strategies=available_strategies,
    )
    if desired.rejected:
        return desired
    assert desired.strategy is not None
    if control_strength(desired.strategy) <= control_strength(current):
        return StrategyDecision(
            strategy=current,
            reason=f"AUTO retains {current.value}; no stronger control is required",
        )
    return StrategyDecision(
        strategy=desired.strategy,
        reason=f"AUTO escalates one way from {current.value} to {desired.strategy.value}",
    )
