"""Control layer: one deterministic controller for every strategy."""

from typing import Any

from prp_runtime.control.reservations import (
    Reservation,
    ReservationDecision,
    ReservationDelta,
    ReservationOutcome,
    ReservationReason,
    ReservationRequest,
    decide_hold,
    hold_reservation,
    release_reservation,
    reservation_delta,
    settle_reservation,
)

__all__ = [
    "DIRECT_WORK_UNIT_NAME",
    "SUPPORTED_STRATEGIES",
    "RunController",
    "Reservation",
    "ReservationDecision",
    "ReservationDelta",
    "ReservationOutcome",
    "ReservationReason",
    "ReservationRequest",
    "decide_hold",
    "hold_reservation",
    "release_reservation",
    "reservation_delta",
    "settle_reservation",
]

_CONTROLLER_EXPORTS = frozenset(
    {"DIRECT_WORK_UNIT_NAME", "SUPPORTED_STRATEGIES", "RunController"}
)


def __getattr__(name: str) -> Any:
    if name not in _CONTROLLER_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from prp_runtime.control.controller import (
        DIRECT_WORK_UNIT_NAME,
        SUPPORTED_STRATEGIES,
        RunController,
    )

    globals()["DIRECT_WORK_UNIT_NAME"] = DIRECT_WORK_UNIT_NAME
    globals()["SUPPORTED_STRATEGIES"] = SUPPORTED_STRATEGIES
    globals()["RunController"] = RunController
    return globals()[name]


def __dir__() -> list[str]:
    return sorted(__all__)
