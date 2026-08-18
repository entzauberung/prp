"""Immutable reservation contracts for bounded provider dispatch.

This module contains no Store, clock or Provider access. Estimates describe
admission facts only; measured Usage is kept separate and is supplied when a
held reservation is settled.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum, unique
from typing import Annotated, Self

from pydantic import Field, StringConstraints, model_validator

from prp_runtime.domain.enums import ReservationStatus
from prp_runtime.domain.models import Budget, DomainModel, Usage
from prp_runtime.domain.transitions import transition_reservation
from prp_runtime.domain.values import (
    ReservationId,
    RunId,
    UtcTimestamp,
    WorkUnitId,
    utc_now,
)

__all__ = [
    "ReservationDecision",
    "ReservationDelta",
    "ReservationOutcome",
    "ReservationReason",
    "Reservation",
    "ReservationRequest",
    "ReservationStatus",
    "decide_hold",
    "hold_reservation",
    "release_reservation",
    "reservation_delta",
    "settle_reservation",
]

DispatchKey = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
]


class ReservationRequest(DomainModel):
    """The public facts required to admit one provider dispatch."""

    run_id: RunId
    work_unit_id: WorkUnitId
    dispatch_key: DispatchKey
    attempt_units: int = Field(default=1, ge=1)
    estimated_input_tokens: int | None = Field(default=None, ge=0)
    estimated_output_tokens: int | None = Field(default=None, ge=0)
    token_upper_bound: int | None = Field(default=None, ge=0)
    strong_token_upper_bound: int | None = Field(default=None, ge=0)
    capacity_key: DispatchKey | None = None

    @model_validator(mode="after")
    def _token_estimates_are_bounded(self) -> ReservationRequest:
        estimate: int | None = None
        if (
            self.estimated_input_tokens is not None
            and self.estimated_output_tokens is not None
        ):
            estimate = self.estimated_input_tokens + self.estimated_output_tokens
        if estimate is not None and self.token_upper_bound is not None:
            if estimate > self.token_upper_bound:
                raise ValueError("token estimate cannot exceed token_upper_bound")
        if (
            self.strong_token_upper_bound is not None
            and self.token_upper_bound is not None
            and self.strong_token_upper_bound > self.token_upper_bound
        ):
            raise ValueError(
                "strong_token_upper_bound cannot exceed token_upper_bound"
            )
        return self

    @property
    def total_token_upper_bound(self) -> int | None:
        """Return a proven total-token upper bound, if one was declared."""
        return self.token_upper_bound


@unique
class ReservationOutcome(StrEnum):
    """Admission result for one proposed hold."""

    ALLOW = "ALLOW"
    REJECT = "REJECT"


@unique
class ReservationReason(StrEnum):
    """Stable reasons for an admission result."""

    ALLOWED = "ALLOWED"
    ATTEMPT_CEILING = "ATTEMPT_CEILING"
    TOKEN_CEILING = "TOKEN_CEILING"
    STRONG_TOKEN_CEILING = "STRONG_TOKEN_CEILING"
    UNKNOWN_TOKEN_UPPER_BOUND = "UNKNOWN_TOKEN_UPPER_BOUND"
    UNKNOWN_STRONG_TOKEN_UPPER_BOUND = "UNKNOWN_STRONG_TOKEN_UPPER_BOUND"
    UNKNOWN_MEASURED_USAGE = "UNKNOWN_MEASURED_USAGE"


class ReservationDecision(DomainModel):
    """Pure admission result with projected held resource facts."""

    outcome: ReservationOutcome
    reason: ReservationReason
    rationale: str = Field(min_length=1)
    projected_attempts: int = Field(ge=0)
    projected_token_upper_bound: int | None = Field(default=None, ge=0)
    projected_strong_token_upper_bound: int | None = Field(default=None, ge=0)

    @property
    def allowed(self) -> bool:
        return self.outcome is ReservationOutcome.ALLOW

    @model_validator(mode="after")
    def _reason_matches_outcome(self) -> Self:
        if (
            self.outcome is ReservationOutcome.ALLOW
            and self.reason is not ReservationReason.ALLOWED
        ):
            raise ValueError("ALLOW requires the ALLOWED reason")
        if self.outcome is ReservationOutcome.REJECT and self.reason is ReservationReason.ALLOWED:
            raise ValueError("REJECT cannot use the ALLOWED reason")
        return self


class ReservationDelta(DomainModel):
    """Facts released from a hold and optional measured settlement facts."""

    released_attempt_units: int = Field(ge=0)
    released_token_upper_bound: int | None = Field(default=None, ge=0)
    released_strong_token_upper_bound: int | None = Field(default=None, ge=0)
    measured_usage: Usage | None = None
    total_token_overage: int | None = Field(default=None, ge=0)
    strong_token_overage: int | None = Field(default=None, ge=0)


def decide_hold(
    request: ReservationRequest,
    budget: Budget,
    *,
    completed_attempt_units: int = 0,
    held_attempt_units: int = 0,
    measured_usage: Usage | None = None,
    held_token_upper_bound: int = 0,
    held_strong_token_upper_bound: int = 0,
) -> ReservationDecision:
    """Decide whether one request may be held against a resource snapshot.

    Attempt units are hard reservations. Token limits use declared upper bounds;
    an unknown bound is rejected only when a corresponding budget ceiling exists.
    Equality is admitted for the current dispatch and therefore blocks only a
    later hold.
    """
    if completed_attempt_units < 0 or held_attempt_units < 0:
        raise ValueError("attempt counts must be non-negative")
    if held_token_upper_bound < 0 or held_strong_token_upper_bound < 0:
        raise ValueError("held token bounds must be non-negative")

    projected_attempts = (
        completed_attempt_units + held_attempt_units + request.attempt_units
    )
    if budget.max_attempts is not None and projected_attempts > budget.max_attempts:
        return _reject_decision(
            ReservationReason.ATTEMPT_CEILING,
            f"projected attempts {projected_attempts} exceed the ceiling of {budget.max_attempts}",
            projected_attempts,
        )

    request_upper = request.total_token_upper_bound
    if (
        measured_usage is None
        and (
            budget.max_total_tokens is not None
            or budget.max_strong_model_tokens is not None
        )
    ):
        return _reject_decision(
            ReservationReason.UNKNOWN_MEASURED_USAGE,
            "measured usage is required before admitting against a token ceiling",
            projected_attempts,
        )
    known_usage = Usage() if measured_usage is None else measured_usage
    if budget.max_total_tokens is not None and request_upper is None:
        return _reject_decision(
            ReservationReason.UNKNOWN_TOKEN_UPPER_BOUND,
            "a total-token upper bound is required for a bounded token hold",
            projected_attempts,
        )
    projected_total = None
    if request_upper is not None:
        projected_total = known_usage.total_tokens + held_token_upper_bound + request_upper
        if budget.max_total_tokens is not None and projected_total > budget.max_total_tokens:
            return _reject_decision(
                ReservationReason.TOKEN_CEILING,
                f"projected token upper bound {projected_total} exceeds the "
                f"ceiling of {budget.max_total_tokens}",
                projected_attempts,
                projected_token_upper_bound=projected_total,
            )

    request_strong_upper = request.strong_token_upper_bound
    if budget.max_strong_model_tokens is not None and request_strong_upper is None:
        return _reject_decision(
            ReservationReason.UNKNOWN_STRONG_TOKEN_UPPER_BOUND,
            "a strong-token upper bound is required for a bounded strong-token hold",
            projected_attempts,
            projected_token_upper_bound=projected_total,
        )
    projected_strong = None
    if request_strong_upper is not None:
        projected_strong = (
            known_usage.strong_model_tokens
            + held_strong_token_upper_bound
            + request_strong_upper
        )
        if (
            budget.max_strong_model_tokens is not None
            and projected_strong > budget.max_strong_model_tokens
        ):
            return _reject_decision(
                ReservationReason.STRONG_TOKEN_CEILING,
                "projected strong-token upper bound exceeds the declared ceiling",
                projected_attempts,
                projected_token_upper_bound=projected_total,
                projected_strong_token_upper_bound=projected_strong,
            )

    return ReservationDecision(
        outcome=ReservationOutcome.ALLOW,
        reason=ReservationReason.ALLOWED,
        rationale="attempt and declared token bounds fit the current budget snapshot",
        projected_attempts=projected_attempts,
        projected_token_upper_bound=projected_total,
        projected_strong_token_upper_bound=projected_strong,
    )


def hold_reservation(reservation: Reservation, *, held_at: datetime) -> Reservation:
    """Move one pending reservation to HELD without side effects."""
    if reservation.status is not ReservationStatus.PENDING:
        raise ValueError("only a PENDING reservation can be held")
    transition_reservation(reservation.status, ReservationStatus.HELD)
    return Reservation.model_validate(
        reservation.model_dump()
        | {"status": ReservationStatus.HELD, "held_at": held_at}
    )


def settle_reservation(
    reservation: Reservation,
    *,
    completed_at: datetime,
    measured_usage: Usage | None,
) -> Reservation:
    """Settle a held reservation with measured or explicitly unknown usage."""
    if reservation.status is not ReservationStatus.HELD:
        raise ValueError("only a HELD reservation can be settled")
    transition_reservation(reservation.status, ReservationStatus.SETTLED)
    return Reservation.model_validate(
        reservation.model_dump()
        | {
            "status": ReservationStatus.SETTLED,
            "completed_at": completed_at,
            "measured_usage": measured_usage,
        }
    )


def release_reservation(
    reservation: Reservation,
    *,
    completed_at: datetime,
    expired: bool = False,
) -> Reservation:
    """Release or expire a held reservation without inventing Usage."""
    if reservation.status is not ReservationStatus.HELD:
        raise ValueError("only a HELD reservation can be released")
    target = ReservationStatus.EXPIRED if expired else ReservationStatus.RELEASED
    transition_reservation(reservation.status, target)
    return Reservation.model_validate(
        reservation.model_dump()
        | {
            "status": (
                target
            ),
            "completed_at": completed_at,
            "measured_usage": None,
        }
    )


def reservation_delta(reservation: Reservation) -> ReservationDelta:
    """Return resource quantities released by a terminal reservation."""
    if not reservation.status.is_terminal:
        raise ValueError("reservation delta requires a terminal reservation")
    request = reservation.request
    actual = reservation.measured_usage
    total_overage = None
    if actual is not None and request.total_token_upper_bound is not None:
        total_overage = max(
            0, actual.total_tokens - request.total_token_upper_bound
        )
    strong_overage = None
    if actual is not None and request.strong_token_upper_bound is not None:
        strong_overage = max(
            0, actual.strong_model_tokens - request.strong_token_upper_bound
        )
    return ReservationDelta(
        released_attempt_units=request.attempt_units,
        released_token_upper_bound=request.total_token_upper_bound,
        released_strong_token_upper_bound=request.strong_token_upper_bound,
        measured_usage=actual,
        total_token_overage=total_overage,
        strong_token_overage=strong_overage,
    )


def _reject_decision(
    reason: ReservationReason,
    rationale: str,
    projected_attempts: int,
    *,
    projected_token_upper_bound: int | None = None,
    projected_strong_token_upper_bound: int | None = None,
) -> ReservationDecision:
    return ReservationDecision(
        outcome=ReservationOutcome.REJECT,
        reason=reason,
        rationale=rationale,
        projected_attempts=projected_attempts,
        projected_token_upper_bound=projected_token_upper_bound,
        projected_strong_token_upper_bound=projected_strong_token_upper_bound,
    )


class Reservation(DomainModel):
    """One immutable reservation snapshot and its terminal settlement facts."""

    reservation_id: ReservationId
    request: ReservationRequest
    status: ReservationStatus = ReservationStatus.PENDING
    created_at: UtcTimestamp = Field(default_factory=utc_now)
    held_at: UtcTimestamp | None = None
    completed_at: UtcTimestamp | None = None
    measured_usage: Usage | None = None

    @model_validator(mode="after")
    def _lifecycle_is_consistent(self) -> Reservation:
        if self.status is ReservationStatus.PENDING:
            if (
                self.held_at is not None
                or self.completed_at is not None
                or self.measured_usage is not None
            ):
                raise ValueError("PENDING reservation has no hold or settlement facts")
        elif self.held_at is None:
            raise ValueError("a reservation that left PENDING must have held_at")

        if self.status is ReservationStatus.HELD:
            if self.completed_at is not None or self.measured_usage is not None:
                raise ValueError("HELD reservation must not have settlement facts")
        elif self.status.is_terminal:
            if self.completed_at is None:
                raise ValueError("a terminal reservation must have completed_at")
            held_at = self.held_at
            assert held_at is not None
            if self.completed_at < held_at:
                raise ValueError("completed_at cannot precede held_at")

        if self.status in (ReservationStatus.RELEASED, ReservationStatus.EXPIRED):
            if self.measured_usage is not None:
                raise ValueError("released or expired reservation has no measured usage")

        if self.held_at is not None and self.held_at < self.created_at:
            raise ValueError("held_at cannot precede created_at")
        return self
