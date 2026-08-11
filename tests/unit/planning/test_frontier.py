"""Ready, waiting, blocked, and complete frontier matrix."""

from datetime import UTC, datetime

import pytest

from prp_runtime.domain.enums import WorkUnitStatus
from prp_runtime.domain.models import WorkUnit
from prp_runtime.planning.frontier import (
    BlockedReason,
    FrontierResult,
    compute_frontier,
)

T0 = datetime(2026, 8, 11, tzinfo=UTC)


def unit(
    key: str,
    *,
    version: int = 1,
    status: WorkUnitStatus = WorkUnitStatus.PENDING,
    depends_on: tuple[str, ...] = (),
) -> WorkUnit:
    return WorkUnit(
        work_unit_id=f"wu_{key}",
        run_id="run_frontier",
        graph_version=version,
        name=key,
        instruction=f"do {key}",
        status=status,
        depends_on=depends_on,
        created_at=T0,
    )


def test_root_and_chain_frontier_advances_only_after_success() -> None:
    root = unit("root", status=WorkUnitStatus.SUCCEEDED)
    child = unit("child", depends_on=(root.work_unit_id,))
    result = compute_frontier((child, root), graph_version=1)
    assert result == FrontierResult(
        graph_version=1,
        ready=(child.work_unit_id,),
        waiting=(),
        blocked=(),
        blocked_details=(),
        complete=(root.work_unit_id,),
    )


def test_root_is_ready_and_running_unit_is_waiting() -> None:
    result = compute_frontier(
        (
            unit("running", status=WorkUnitStatus.RUNNING),
            unit("root"),
        ),
        graph_version=1,
    )
    assert result.ready == ("wu_root",)
    assert result.waiting == ("wu_running",)


def test_diamond_returns_stable_ready_order() -> None:
    root = unit("root", status=WorkUnitStatus.SUCCEEDED)
    result = compute_frontier(
        (
            unit("join", depends_on=("wu_left", "wu_right")),
            unit("right", depends_on=(root.work_unit_id,)),
            unit("left", depends_on=(root.work_unit_id,)),
            root,
        ),
        graph_version=1,
    )
    assert result.ready == ("wu_left", "wu_right")
    assert result.waiting == ("wu_join",)


@pytest.mark.parametrize(
    ("dependency_status", "reason"),
    [
        (WorkUnitStatus.FAILED, BlockedReason.DEPENDENCY_FAILED),
        (WorkUnitStatus.CANCELLED, BlockedReason.DEPENDENCY_CANCELLED),
        (WorkUnitStatus.INVALIDATED, BlockedReason.DEPENDENCY_INVALIDATED),
        (WorkUnitStatus.BLOCKED, BlockedReason.DEPENDENCY_BLOCKED),
    ],
)
def test_terminal_unsuccessful_dependency_blocks_dependents(
    dependency_status: WorkUnitStatus,
    reason: BlockedReason,
) -> None:
    dependency = unit("dependency", status=dependency_status)
    dependent = unit("dependent", depends_on=(dependency.work_unit_id,))
    result = compute_frontier((dependent, dependency), graph_version=1)
    dependent_detail = next(
        detail
        for detail in result.blocked_details
        if detail.work_unit_id == dependent.work_unit_id
    )
    assert dependent.work_unit_id in result.blocked
    assert dependent_detail.reason is reason
    if dependency_status is WorkUnitStatus.BLOCKED:
        assert dependency.work_unit_id in result.blocked
        assert result.complete == ()
    else:
        assert result.complete == (dependency.work_unit_id,)


def test_own_blocked_status_is_reported_without_reclassification() -> None:
    result = compute_frontier(
        (unit("blocked", status=WorkUnitStatus.BLOCKED),), graph_version=1
    )
    assert result.blocked == ("wu_blocked",)
    assert result.blocked_details[0].reason is BlockedReason.OWN_BLOCKED


def test_dangling_dependency_is_blocked_not_ready() -> None:
    result = compute_frontier(
        (unit("dangling", depends_on=("wu_missing",)),), graph_version=1
    )
    assert result.ready == ()
    assert result.blocked == ("wu_dangling",)
    assert result.blocked_details[0].reason is BlockedReason.DANGLING_DEPENDENCY
    assert result.blocked_details[0].dependency_ids == ("wu_missing",)


def test_old_graph_version_is_ignored_completely() -> None:
    old = unit("old", version=1)
    current = unit("current", version=2)
    result = compute_frontier((old, current), graph_version=2)
    assert result.ready == (current.work_unit_id,)
    assert old.work_unit_id not in result.model_dump_json()


def test_complete_contains_all_terminal_success_and_failure_facts() -> None:
    result = compute_frontier(
        (
            unit("success", status=WorkUnitStatus.SUCCEEDED),
            unit("failed", status=WorkUnitStatus.FAILED),
            unit("cancelled", status=WorkUnitStatus.CANCELLED),
            unit("invalid", status=WorkUnitStatus.INVALIDATED),
        ),
        graph_version=1,
    )
    assert result.complete == (
        "wu_cancelled",
        "wu_failed",
        "wu_invalid",
        "wu_success",
    )


def test_frontier_is_pure_and_rejects_duplicate_ids() -> None:
    values = (unit("same"), unit("same"))
    with pytest.raises(ValueError, match="duplicate"):
        compute_frontier(values, graph_version=1)
    before = values[0].status
    compute_frontier((unit("root"),), graph_version=1)
    assert values[0].status is before
