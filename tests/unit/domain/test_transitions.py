"""Full transition table tests for the shared state machine."""

import itertools

import pytest

from prp_runtime.domain.enums import (
    AgentMode,
    AttemptStatus,
    BridgeClaimStatus,
    ExecutionStrategy,
    ReservationStatus,
    RoutingPolicy,
    RunStatus,
    ToolCallStatus,
    WorkUnitStatus,
)
from prp_runtime.domain.transitions import (
    ATTEMPT_TRANSITIONS,
    BRIDGE_CLAIM_TRANSITIONS,
    RESERVATION_TRANSITIONS,
    RUN_TRANSITIONS,
    TOOL_CALL_TRANSITIONS,
    WORK_UNIT_TRANSITIONS,
    AttemptNotAllowedError,
    DomainTransitionError,
    IllegalStatusTransitionError,
    RecoveryAction,
    RunCompletionNotAllowedError,
    StrategyEscalationNotAllowedError,
    assert_can_start_attempt,
    can_escalate_strategy,
    can_start_attempt,
    can_transition_attempt,
    can_transition_bridge_claim,
    can_transition_reservation,
    can_transition_run,
    can_transition_work_unit,
    control_strength,
    decide_attempt_recovery,
    decide_reservation_recovery,
    decide_run_recovery,
    decide_tool_call_recovery,
    decide_work_unit_recovery,
    escalate_strategy,
    mark_attempt_unconfirmed,
    recover_attempt_on_restart,
    resolve_run_outcome,
    transition_attempt,
    transition_bridge_claim,
    transition_reservation,
    transition_run,
    transition_work_unit,
)

# --- table completeness ---------------------------------------------------------


def test_every_status_has_a_transition_entry() -> None:
    assert set(RUN_TRANSITIONS) == set(RunStatus)
    assert set(WORK_UNIT_TRANSITIONS) == set(WorkUnitStatus)
    assert set(ATTEMPT_TRANSITIONS) == set(AttemptStatus)
    assert set(RESERVATION_TRANSITIONS) == set(ReservationStatus)
    assert set(TOOL_CALL_TRANSITIONS) == set(ToolCallStatus)
    assert set(BRIDGE_CLAIM_TRANSITIONS) == set(BridgeClaimStatus)


def test_terminal_statuses_have_no_outgoing_transition() -> None:
    for status in RunStatus:
        assert bool(RUN_TRANSITIONS[status]) is not status.is_terminal
    for status in WorkUnitStatus:
        if status.is_terminal:
            assert WORK_UNIT_TRANSITIONS[status] == frozenset()
    for status in AttemptStatus:
        if status.is_terminal:
            assert ATTEMPT_TRANSITIONS[status] == frozenset()
    for status in ToolCallStatus:
        if status.is_terminal:
            assert TOOL_CALL_TRANSITIONS[status] == frozenset()


def test_no_status_transitions_to_itself() -> None:
    for status in RunStatus:
        assert status not in RUN_TRANSITIONS[status]
    for status in WorkUnitStatus:
        assert status not in WORK_UNIT_TRANSITIONS[status]
    for status in AttemptStatus:
        assert status not in ATTEMPT_TRANSITIONS[status]
    for status in ReservationStatus:
        assert status not in RESERVATION_TRANSITIONS[status]


RESERVATION_ALLOWED: frozenset[tuple[ReservationStatus, ReservationStatus]] = frozenset(
    {
        (ReservationStatus.PENDING, ReservationStatus.HELD),
        (ReservationStatus.HELD, ReservationStatus.SETTLED),
        (ReservationStatus.HELD, ReservationStatus.RELEASED),
        (ReservationStatus.HELD, ReservationStatus.EXPIRED),
    }
)


@pytest.mark.parametrize(
    ("current", "target"), itertools.product(ReservationStatus, ReservationStatus)
)
def test_reservation_transition_table(
    current: ReservationStatus, target: ReservationStatus
) -> None:
    expected = (current, target) in RESERVATION_ALLOWED
    assert can_transition_reservation(current, target) is expected
    if expected:
        assert transition_reservation(current, target) is target
    else:
        with pytest.raises(IllegalStatusTransitionError) as excinfo:
            transition_reservation(current, target)
        assert excinfo.value.entity == "reservation"
        assert excinfo.value.current == current.value
        assert excinfo.value.target == target.value


def test_reservation_terminal_replay_is_explicitly_idempotent() -> None:
    assert (
        transition_reservation(
            ReservationStatus.SETTLED,
            ReservationStatus.SETTLED,
            idempotent_terminal=True,
        )
        is ReservationStatus.SETTLED
    )
    with pytest.raises(IllegalStatusTransitionError):
        transition_reservation(ReservationStatus.SETTLED, ReservationStatus.RELEASED)


# --- run transitions ------------------------------------------------------------

RUN_ALLOWED: frozenset[tuple[RunStatus, RunStatus]] = frozenset(
    {
        (RunStatus.PENDING, RunStatus.RUNNING),
        (RunStatus.PENDING, RunStatus.CANCELLED),
        (RunStatus.PENDING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLING),
        (RunStatus.RUNNING, RunStatus.SUCCEEDED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLED),
        (RunStatus.CANCELLING, RunStatus.CANCELLED),
    }
)


@pytest.mark.parametrize(("current", "target"), itertools.product(RunStatus, RunStatus))
def test_run_transition_table(current: RunStatus, target: RunStatus) -> None:
    expected = (current, target) in RUN_ALLOWED
    assert can_transition_run(current, target) is expected
    if expected:
        assert transition_run(current, target) is target
    else:
        with pytest.raises(IllegalStatusTransitionError) as excinfo:
            transition_run(current, target)
        assert excinfo.value.entity == "run"
        assert excinfo.value.current == current.value
        assert excinfo.value.target == target.value


def test_cancelling_run_cannot_succeed() -> None:
    assert not can_transition_run(RunStatus.CANCELLING, RunStatus.SUCCEEDED)


# --- work unit transitions ------------------------------------------------------

WORK_UNIT_ALLOWED: frozenset[tuple[WorkUnitStatus, WorkUnitStatus]] = frozenset(
    {
        (WorkUnitStatus.PENDING, WorkUnitStatus.READY),
        (WorkUnitStatus.PENDING, WorkUnitStatus.BLOCKED),
        (WorkUnitStatus.PENDING, WorkUnitStatus.CANCELLED),
        (WorkUnitStatus.PENDING, WorkUnitStatus.INVALIDATED),
        (WorkUnitStatus.READY, WorkUnitStatus.RUNNING),
        (WorkUnitStatus.READY, WorkUnitStatus.BLOCKED),
        (WorkUnitStatus.READY, WorkUnitStatus.CANCELLED),
        (WorkUnitStatus.READY, WorkUnitStatus.INVALIDATED),
        (WorkUnitStatus.RUNNING, WorkUnitStatus.SUCCEEDED),
        (WorkUnitStatus.RUNNING, WorkUnitStatus.FAILED),
        (WorkUnitStatus.RUNNING, WorkUnitStatus.CANCELLED),
        (WorkUnitStatus.BLOCKED, WorkUnitStatus.READY),
        (WorkUnitStatus.BLOCKED, WorkUnitStatus.CANCELLED),
        (WorkUnitStatus.BLOCKED, WorkUnitStatus.INVALIDATED),
    }
)


@pytest.mark.parametrize(
    ("current", "target"), itertools.product(WorkUnitStatus, WorkUnitStatus)
)
def test_work_unit_transition_table(current: WorkUnitStatus, target: WorkUnitStatus) -> None:
    expected = (current, target) in WORK_UNIT_ALLOWED
    assert can_transition_work_unit(current, target) is expected
    if expected:
        assert transition_work_unit(current, target) is target
    else:
        with pytest.raises(IllegalStatusTransitionError):
            transition_work_unit(current, target)


def test_failed_work_unit_cannot_be_retried_in_place() -> None:
    for target in (WorkUnitStatus.READY, WorkUnitStatus.RUNNING, WorkUnitStatus.PENDING):
        assert not can_transition_work_unit(WorkUnitStatus.FAILED, target)


def test_running_work_unit_cannot_be_invalidated_before_cancellation() -> None:
    assert not can_transition_work_unit(WorkUnitStatus.RUNNING, WorkUnitStatus.INVALIDATED)
    assert can_transition_work_unit(WorkUnitStatus.RUNNING, WorkUnitStatus.CANCELLED)


# --- attempt transitions --------------------------------------------------------

ATTEMPT_ALLOWED: frozenset[tuple[AttemptStatus, AttemptStatus]] = frozenset(
    {
        (AttemptStatus.PENDING, AttemptStatus.RUNNING),
        (AttemptStatus.PENDING, AttemptStatus.CANCELLED),
        (AttemptStatus.RUNNING, AttemptStatus.SUCCEEDED),
        (AttemptStatus.RUNNING, AttemptStatus.FAILED),
        (AttemptStatus.RUNNING, AttemptStatus.CANCELLED),
        (AttemptStatus.RUNNING, AttemptStatus.INTERRUPTED),
        (AttemptStatus.RUNNING, AttemptStatus.UNKNOWN),
    }
)


@pytest.mark.parametrize(("current", "target"), itertools.product(AttemptStatus, AttemptStatus))
def test_attempt_transition_table(current: AttemptStatus, target: AttemptStatus) -> None:
    expected = (current, target) in ATTEMPT_ALLOWED
    assert can_transition_attempt(current, target) is expected
    if expected:
        assert transition_attempt(current, target) is target
    else:
        with pytest.raises(IllegalStatusTransitionError):
            transition_attempt(current, target)


@pytest.mark.parametrize(
    "terminal",
    [
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.INTERRUPTED,
        AttemptStatus.UNKNOWN,
    ],
)
def test_terminal_attempt_never_runs_again(terminal: AttemptStatus) -> None:
    assert not can_transition_attempt(terminal, AttemptStatus.RUNNING)


# --- recovery boundaries --------------------------------------------------------


def test_restart_marks_running_attempt_interrupted_and_keeps_history() -> None:
    assert recover_attempt_on_restart(AttemptStatus.RUNNING) is AttemptStatus.INTERRUPTED
    for status in AttemptStatus:
        if status is not AttemptStatus.RUNNING:
            assert recover_attempt_on_restart(status) is status


def test_unconfirmed_cancellation_maps_running_to_unknown() -> None:
    assert mark_attempt_unconfirmed(AttemptStatus.RUNNING) is AttemptStatus.UNKNOWN
    assert mark_attempt_unconfirmed(AttemptStatus.SUCCEEDED) is AttemptStatus.SUCCEEDED
    with pytest.raises(IllegalStatusTransitionError):
        mark_attempt_unconfirmed(AttemptStatus.PENDING)


def test_tool_recovery_never_guesses_an_unconfirmed_side_effect() -> None:
    running = decide_tool_call_recovery(ToolCallStatus.RUNNING)
    assert running.action is RecoveryAction.INTERRUPT
    assert running.target_status is ToolCallStatus.UNKNOWN

    awaiting = decide_tool_call_recovery(ToolCallStatus.AWAITING_APPROVAL)
    assert awaiting.action is RecoveryAction.CONTINUE
    assert awaiting.target_status is ToolCallStatus.AWAITING_APPROVAL
    assert awaiting.retry_allowed is True

    missing_approval = decide_tool_call_recovery(
        ToolCallStatus.AWAITING_APPROVAL,
        has_approval=False,
    )
    assert missing_approval.action is RecoveryAction.BLOCK

    unknown = decide_tool_call_recovery(ToolCallStatus.UNKNOWN)
    assert unknown.action is RecoveryAction.BLOCK
    assert unknown.target_status is ToolCallStatus.UNKNOWN

    terminal = decide_tool_call_recovery(
        ToolCallStatus.SUCCEEDED,
        has_result=True,
        has_history_result=False,
    )
    assert terminal.action is RecoveryAction.PRESERVE
    assert terminal.target_status is ToolCallStatus.SUCCEEDED

    conflict = decide_tool_call_recovery(
        ToolCallStatus.REQUESTED,
        has_conflict=True,
    )
    assert conflict.action is RecoveryAction.BLOCK


def test_recovery_decision_covers_attempt_outcome_boundaries() -> None:
    pending = decide_attempt_recovery(AttemptStatus.PENDING)
    assert pending.action is RecoveryAction.CONTINUE
    assert pending.target_status is AttemptStatus.PENDING
    assert pending.can_continue is True

    running = decide_attempt_recovery(AttemptStatus.RUNNING)
    assert running.action is RecoveryAction.INTERRUPT
    assert running.target_status is AttemptStatus.INTERRUPTED
    assert running.retry_allowed is False

    unknown = decide_attempt_recovery(AttemptStatus.UNKNOWN)
    assert unknown.action is RecoveryAction.PRESERVE
    assert unknown.target_status is AttemptStatus.UNKNOWN
    assert unknown.can_continue is False

    for terminal in (
        AttemptStatus.SUCCEEDED,
        AttemptStatus.FAILED,
        AttemptStatus.CANCELLED,
        AttemptStatus.INTERRUPTED,
    ):
        decision = decide_attempt_recovery(terminal)
        assert decision.action is RecoveryAction.PRESERVE
        assert decision.target_status is terminal


def test_recovery_decision_classifies_run_conflicts_and_terminal_states() -> None:
    assert (
        decide_run_recovery(RunStatus.PENDING).action is RecoveryAction.CONTINUE
    )
    conflict = decide_run_recovery(
        RunStatus.PENDING,
        has_dispatched_attempt=True,
    )
    assert conflict.action is RecoveryAction.BLOCK
    assert conflict.target_status is RunStatus.PENDING

    cancelling = decide_run_recovery(RunStatus.CANCELLING)
    assert cancelling.action is RecoveryAction.PRESERVE
    assert cancelling.retry_allowed is False

    terminal = decide_run_recovery(RunStatus.SUCCEEDED)
    assert terminal.action is RecoveryAction.PRESERVE
    assert terminal.target_status is RunStatus.SUCCEEDED

    terminal_conflict = decide_run_recovery(
        RunStatus.SUCCEEDED,
        has_dispatched_attempt=True,
    )
    assert terminal_conflict.action is RecoveryAction.BLOCK


def test_recovery_decision_blocks_work_unit_fact_conflicts() -> None:
    pending = decide_work_unit_recovery(WorkUnitStatus.PENDING)
    assert pending.action is RecoveryAction.CONTINUE
    assert pending.can_continue is True

    dispatched = decide_work_unit_recovery(
        WorkUnitStatus.READY,
        has_dispatched_attempt=True,
    )
    assert dispatched.action is RecoveryAction.BLOCK
    assert dispatched.target_status is WorkUnitStatus.BLOCKED

    interrupted = decide_work_unit_recovery(
        WorkUnitStatus.RUNNING,
        has_running_attempt=True,
    )
    assert interrupted.action is RecoveryAction.TERMINATE
    assert interrupted.target_status is WorkUnitStatus.FAILED

    unknown = decide_work_unit_recovery(
        WorkUnitStatus.RUNNING,
        has_unconfirmed_attempt=True,
    )
    assert unknown.action is RecoveryAction.BLOCK
    assert unknown.target_status is WorkUnitStatus.RUNNING

    succeeded = decide_work_unit_recovery(WorkUnitStatus.SUCCEEDED)
    assert succeeded.action is RecoveryAction.PRESERVE
    assert succeeded.target_status is WorkUnitStatus.SUCCEEDED


def test_recovery_decision_never_releases_an_uncertain_reservation() -> None:
    pending = decide_reservation_recovery(ReservationStatus.PENDING)
    assert pending.action is RecoveryAction.CONTINUE

    orphan = decide_reservation_recovery(ReservationStatus.HELD)
    assert orphan.action is RecoveryAction.RELEASE
    assert orphan.target_status is ReservationStatus.RELEASED

    for kwargs in (
        {"has_inflight_attempt": True},
        {"has_unconfirmed_attempt": True},
    ):
        held = decide_reservation_recovery(ReservationStatus.HELD, **kwargs)
        assert held.action is RecoveryAction.PRESERVE
        assert held.target_status is ReservationStatus.HELD

    settled = decide_reservation_recovery(ReservationStatus.SETTLED)
    assert settled.action is RecoveryAction.PRESERVE
    assert settled.target_status is ReservationStatus.SETTLED


# --- new attempt admission ------------------------------------------------------


@pytest.mark.parametrize(
    ("run_status", "work_unit_status", "expected"),
    [
        (RunStatus.RUNNING, WorkUnitStatus.READY, True),
        (RunStatus.RUNNING, WorkUnitStatus.RUNNING, True),
        (RunStatus.PENDING, WorkUnitStatus.READY, True),
        (RunStatus.RUNNING, WorkUnitStatus.PENDING, False),
        (RunStatus.RUNNING, WorkUnitStatus.BLOCKED, False),
        (RunStatus.RUNNING, WorkUnitStatus.SUCCEEDED, False),
        (RunStatus.CANCELLING, WorkUnitStatus.READY, False),
        (RunStatus.CANCELLED, WorkUnitStatus.READY, False),
        (RunStatus.SUCCEEDED, WorkUnitStatus.READY, False),
        (RunStatus.FAILED, WorkUnitStatus.RUNNING, False),
    ],
)
def test_attempt_admission(
    run_status: RunStatus, work_unit_status: WorkUnitStatus, expected: bool
) -> None:
    assert can_start_attempt(run_status, work_unit_status) is expected
    if expected:
        assert assert_can_start_attempt(run_status, work_unit_status) is None
    else:
        with pytest.raises(AttemptNotAllowedError):
            assert_can_start_attempt(run_status, work_unit_status)


@pytest.mark.parametrize("run_status", [RunStatus.CANCELLING, RunStatus.CANCELLED])
def test_cancelled_run_reports_cancellation_as_the_reason(run_status: RunStatus) -> None:
    with pytest.raises(AttemptNotAllowedError) as excinfo:
        assert_can_start_attempt(run_status, WorkUnitStatus.READY)
    assert excinfo.value.reason == "run is cancelled"
    assert excinfo.value.run_status is run_status


# --- run completion -------------------------------------------------------------


def test_run_cannot_complete_while_work_is_outstanding() -> None:
    for pending in (
        WorkUnitStatus.PENDING,
        WorkUnitStatus.READY,
        WorkUnitStatus.RUNNING,
        WorkUnitStatus.BLOCKED,
    ):
        with pytest.raises(RunCompletionNotAllowedError):
            resolve_run_outcome([WorkUnitStatus.SUCCEEDED, pending])


def test_run_outcome_precedence() -> None:
    assert resolve_run_outcome([WorkUnitStatus.SUCCEEDED]) is RunStatus.SUCCEEDED
    assert (
        resolve_run_outcome([WorkUnitStatus.SUCCEEDED, WorkUnitStatus.FAILED])
        is RunStatus.FAILED
    )
    assert (
        resolve_run_outcome([WorkUnitStatus.SUCCEEDED, WorkUnitStatus.CANCELLED])
        is RunStatus.CANCELLED
    )
    assert (
        resolve_run_outcome([WorkUnitStatus.FAILED, WorkUnitStatus.CANCELLED])
        is RunStatus.FAILED
    )


def test_invalidated_units_do_not_decide_the_outcome() -> None:
    assert (
        resolve_run_outcome([WorkUnitStatus.INVALIDATED, WorkUnitStatus.SUCCEEDED])
        is RunStatus.SUCCEEDED
    )
    with pytest.raises(RunCompletionNotAllowedError):
        resolve_run_outcome([WorkUnitStatus.INVALIDATED])


def test_cancel_request_wins_over_successful_units() -> None:
    assert (
        resolve_run_outcome([WorkUnitStatus.SUCCEEDED], cancel_requested=True)
        is RunStatus.CANCELLED
    )


def test_empty_graph_cannot_complete() -> None:
    with pytest.raises(RunCompletionNotAllowedError):
        resolve_run_outcome([])


# --- strategy escalation --------------------------------------------------------


def test_control_strength_is_strictly_increasing() -> None:
    strengths = [
        control_strength(strategy)
        for strategy in (
            ExecutionStrategy.DIRECT,
            ExecutionStrategy.CASCADE,
            ExecutionStrategy.PLANNED,
            ExecutionStrategy.PROGRESSIVE,
        )
    ]
    assert strengths == sorted(set(strengths))


def test_agent_modes_do_not_expand_strategy_control_ordering() -> None:
    strategies = tuple(ExecutionStrategy)
    assert strategies == (
        ExecutionStrategy.DIRECT,
        ExecutionStrategy.CASCADE,
        ExecutionStrategy.PLANNED,
        ExecutionStrategy.PROGRESSIVE,
    )
    assert {mode.value for mode in AgentMode}.isdisjoint(
        {strategy.value for strategy in strategies}
    )
    assert [control_strength(strategy) for strategy in strategies] == [0, 1, 2, 3]


def test_escalation_is_one_directional() -> None:
    assert can_escalate_strategy(ExecutionStrategy.DIRECT, ExecutionStrategy.CASCADE)
    assert not can_escalate_strategy(ExecutionStrategy.PLANNED, ExecutionStrategy.CASCADE)
    assert not can_escalate_strategy(ExecutionStrategy.CASCADE, ExecutionStrategy.CASCADE)


def test_auto_routing_may_escalate_within_the_ordering() -> None:
    assert (
        escalate_strategy(
            ExecutionStrategy.CASCADE,
            ExecutionStrategy.PROGRESSIVE,
            routing_policy=RoutingPolicy.AUTO,
        )
        is ExecutionStrategy.PROGRESSIVE
    )
    with pytest.raises(StrategyEscalationNotAllowedError):
        escalate_strategy(
            ExecutionStrategy.PROGRESSIVE,
            ExecutionStrategy.DIRECT,
            routing_policy=RoutingPolicy.AUTO,
        )


def test_manual_routing_never_escalates() -> None:
    with pytest.raises(StrategyEscalationNotAllowedError) as excinfo:
        escalate_strategy(
            ExecutionStrategy.DIRECT,
            ExecutionStrategy.PLANNED,
            routing_policy=RoutingPolicy.MANUAL,
        )
    assert excinfo.value.reason == "manual routing pins the strategy"


# --- error hierarchy ------------------------------------------------------------


def test_all_state_machine_errors_share_one_base() -> None:
    for error in (
        IllegalStatusTransitionError("run", "PENDING", "SUCCEEDED"),
        AttemptNotAllowedError("run is cancelled", RunStatus.CANCELLED, WorkUnitStatus.READY),
        RunCompletionNotAllowedError("no work unit"),
        StrategyEscalationNotAllowedError("manual routing pins the strategy"),
    ):
        assert isinstance(error, DomainTransitionError)
        assert isinstance(error, ValueError)
        assert str(error)



def test_bridge_claim_transition_table() -> None:
    assert can_transition_bridge_claim(BridgeClaimStatus.ACTIVE, BridgeClaimStatus.EXPIRED)
    assert can_transition_bridge_claim(BridgeClaimStatus.ACTIVE, BridgeClaimStatus.SETTLED)
    assert can_transition_bridge_claim(BridgeClaimStatus.ACTIVE, BridgeClaimStatus.RELEASED)
    assert transition_bridge_claim(BridgeClaimStatus.ACTIVE, BridgeClaimStatus.RELEASED) is BridgeClaimStatus.RELEASED
    with pytest.raises(IllegalStatusTransitionError):
        transition_bridge_claim(BridgeClaimStatus.EXPIRED, BridgeClaimStatus.SETTLED)
    assert (
        transition_bridge_claim(
            BridgeClaimStatus.RELEASED,
            BridgeClaimStatus.RELEASED,
            idempotent_terminal=True,
        )
        is BridgeClaimStatus.RELEASED
    )
