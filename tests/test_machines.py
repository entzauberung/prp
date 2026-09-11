from prp.machines import (
    can_escalate_strategy,
    can_transition_attempt,
    can_transition_run,
    can_transition_work_unit,
)
from prp.vocabulary import AttemptStatus, ExecutionStrategy, RunStatus, WorkUnitStatus


def test_terminal_run_cannot_return() -> None:
    for status in (RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED):
        assert not can_transition_run(status, RunStatus.RUNNING)
        assert not can_transition_run(status, RunStatus.PENDING)


def test_run_legal_path() -> None:
    assert can_transition_run(RunStatus.PENDING, RunStatus.RUNNING)
    assert can_transition_run(RunStatus.RUNNING, RunStatus.SUCCEEDED)
    assert can_transition_run(RunStatus.RUNNING, RunStatus.CANCELLING)
    assert can_transition_run(RunStatus.CANCELLING, RunStatus.CANCELLED)
    assert not can_transition_run(RunStatus.CANCELLING, RunStatus.SUCCEEDED)


def test_invalidated_work_unit_is_terminal() -> None:
    assert WorkUnitStatus.INVALIDATED.is_terminal
    assert not can_transition_work_unit(
        WorkUnitStatus.INVALIDATED, WorkUnitStatus.READY
    )
    assert can_transition_work_unit(WorkUnitStatus.PENDING, WorkUnitStatus.INVALIDATED)


def test_interrupted_attempt_is_not_success() -> None:
    assert AttemptStatus.INTERRUPTED.is_terminal
    assert AttemptStatus.UNKNOWN.is_terminal
    assert can_transition_attempt(AttemptStatus.RUNNING, AttemptStatus.INTERRUPTED)
    assert not can_transition_attempt(AttemptStatus.INTERRUPTED, AttemptStatus.SUCCEEDED)


def test_strategy_escalation_is_one_way() -> None:
    assert can_escalate_strategy(ExecutionStrategy.DIRECT, ExecutionStrategy.PROGRESSIVE)
    assert not can_escalate_strategy(
        ExecutionStrategy.PROGRESSIVE, ExecutionStrategy.DIRECT
    )
    assert not can_escalate_strategy(ExecutionStrategy.PLANNED, ExecutionStrategy.PLANNED)
