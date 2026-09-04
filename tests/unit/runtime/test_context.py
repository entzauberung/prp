"""R0 contract tests for DEV runtime context boundaries."""

import pytest

import prp_runtime.tools.command  # noqa: F401

from prp_runtime.analysis.syntax import SyntaxReport
from prp_runtime.domain.enums import ExecutionLocation, IsolationMode
from prp_runtime.domain.values import (
    new_principal_id,
    new_run_id,
    new_work_unit_id,
    new_workspace_id,
)
from prp_runtime.policy.models import DevExecutionMode, guard_dev_scope
from prp_runtime.runtime.context import (
    DevExecutionContext,
    StaticFact,
    WorkerContext,
    select_relevant_facts,
    static_facts_from_syntax_reports,
)


def make_scope():
    return guard_dev_scope(
        principal_id=new_principal_id(),
        workspace_id=new_workspace_id(),
        mode=DevExecutionMode.HOST,
        isolation_mode=IsolationMode.HOST,
        execution_location=ExecutionLocation.CLOUD,
    )


def test_temporary_host_root_is_internal_and_not_public_evidence() -> None:
    context = DevExecutionContext.from_temporary_root(make_scope(), "/tmp/prp-dev-root")
    assert context.internal_temporary_root.as_posix() == "/tmp/prp-dev-root"
    assert context.resolve_internal_path("src/main.py").as_posix() == (
        "/tmp/prp-dev-root/src/main.py"
    )
    assert "/tmp/prp-dev-root" not in context.model_dump_json()
    assert "temporary_root" not in context.model_dump()


def test_dev_context_rejects_unsafe_or_missing_internal_paths() -> None:
    context = DevExecutionContext.from_temporary_root(make_scope(), "/tmp/prp-dev-root")
    for path in ("../secret", "/etc/passwd", "C:\\secret", "src//main.py"):
        with pytest.raises(ValueError):
            context.resolve_internal_path(path)
    with pytest.raises(RuntimeError, match="temporary root"):
        DevExecutionContext(scope=make_scope()).internal_temporary_root


def test_dev_context_requires_an_absolute_temporary_root() -> None:
    with pytest.raises(ValueError, match="absolute"):
        DevExecutionContext.from_temporary_root(make_scope(), "relative-root")


def test_select_relevant_facts_excludes_unrelated_private_and_oversized_items() -> None:
    keep = StaticFact(
        key="function:run",
        kind="ast",
        summary="FUNCTION run L1",
        work_unit_id="wu_keep",
        round_id="round_a",
    )
    other_unit = keep.model_copy(update={"key": "function:other", "work_unit_id": "wu_other"})
    other_round = keep.model_copy(update={"key": "function:old", "round_id": "round_b"})
    secret = keep.model_copy(update={"key": "function:secret", "summary": "api_key=sk-live"})
    rooted = keep.model_copy(update={"key": "function:root", "summary": "/tmp/project/main.py"})
    duplicate = keep.model_copy(update={"summary": "FUNCTION run duplicate"})
    extras = [
        StaticFact(
            key=f"function:extra{index}",
            kind="ast",
            summary="FUNCTION extra",
            work_unit_id="wu_keep",
            round_id="round_a",
        )
        for index in range(8)
    ]
    selected = select_relevant_facts(
        (keep, other_unit, other_round, secret, rooted, duplicate, *extras),
        work_unit_id="wu_keep",
        round_id="round_a",
        max_items=3,
        max_bytes=200,
    )
    assert [fact.key for fact in selected] == ["function:run", "function:extra0", "function:extra1"]
    bounded = select_relevant_facts(
        (keep, *extras),
        work_unit_id="wu_keep",
        max_items=16,
        max_bytes=40,
    )
    assert [fact.key for fact in bounded] == ["function:run"]
    assert all(fact.work_unit_id == "wu_keep" for fact in selected)


def test_worker_context_renders_only_selected_static_facts() -> None:
    context = WorkerContext(
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        instruction="continue the current unit",
        static_facts=(
            StaticFact(key="function:run", kind="ast", summary="FUNCTION run L1"),
        ),
    )
    rendered = context.render_input()
    assert rendered.startswith("continue the current unit")
    assert "### Static facts" in rendered
    assert "function:run" in rendered
    assert "/home/" not in rendered
    omitted = WorkerContext(
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        instruction="continue the current unit",
    )
    assert "### Static facts" not in omitted.render_input()


def test_absent_syntax_reports_do_not_become_worker_facts() -> None:
    facts = static_facts_from_syntax_reports(
        (
            SyntaxReport(
                parse_ok=False,
                unknown=True,
                before_parse_error="source is absent",
            ),
        )
    )
    assert facts == ()
