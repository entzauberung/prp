"""Deterministic DAG compilation and rejection matrix."""

from collections.abc import Iterable

import pytest

from prp_runtime.domain.enums import ResourceAccess
from prp_runtime.domain.models import ArtifactKind
from prp_runtime.domain.values import ResourceClaim, new_run_id
from prp_runtime.planning.compiler import (
    CompiledPlan,
    compile_plan,
    compile_proposal,
)
from prp_runtime.planning.models import PlanNode, PlanProposal, PlanRejection

RUN_ID = new_run_id()


def _node(key: str, **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "key": key,
        "name": key.title(),
        "instruction": f"produce {key}",
    }
    values.update(overrides)
    return values


def proposal(*nodes: dict[str, object]) -> PlanProposal:
    return PlanProposal(summary="compile this graph", nodes=nodes or (_node("only"),))


def compile_(value: PlanProposal, *, version: int = 3):
    return compile_plan(value, run_id=RUN_ID, graph_version=version)


def keys(plan: CompiledPlan) -> tuple[str, ...]:
    return tuple(node.node_key for node in plan.nodes)


def test_single_node_compiles_to_one_stable_draft() -> None:
    result = compile_(proposal(_node("only")))
    assert isinstance(result, CompiledPlan)
    assert keys(result) == ("only",)
    assert result.nodes[0].run_id == RUN_ID
    assert result.nodes[0].graph_version == 3
    assert result.nodes[0].depends_on == ()
    assert result.nodes[0].work_unit_id.startswith("wu_")
    assert result.node_map[0].work_unit_id == result.nodes[0].work_unit_id


def test_chain_compiles_in_stable_topological_order_and_maps_dependencies() -> None:
    result = compile_(
        proposal(
            _node("finish", depends_on=("middle",)),
            _node("middle", depends_on=("start",)),
            _node("start"),
        )
    )
    assert isinstance(result, CompiledPlan)
    assert keys(result) == ("start", "middle", "finish")
    by_key = {node.node_key: node for node in result.nodes}
    assert by_key["middle"].depends_on == (by_key["start"].work_unit_id,)
    assert by_key["finish"].depends_on == (by_key["middle"].work_unit_id,)


def test_diamond_has_deterministic_sibling_order() -> None:
    value = proposal(
        _node("join", depends_on=("left", "right")),
        _node("right", depends_on=("root",)),
        _node("root"),
        _node("left", depends_on=("root",)),
    )
    first = compile_(value)
    second = compile_(value)
    assert isinstance(first, CompiledPlan)
    assert isinstance(second, CompiledPlan)
    assert keys(first) == keys(second) == ("root", "left", "right", "join")
    assert first.model_dump() == second.model_dump()


def test_generated_ids_are_stable_for_same_run_and_version() -> None:
    value = proposal(_node("only"))
    first = compile_(value, version=4)
    second = compile_proposal(value, run_id=RUN_ID, graph_version=4)
    other_version = compile_(value, version=5)
    assert isinstance(first, CompiledPlan)
    assert isinstance(second, CompiledPlan)
    assert isinstance(other_version, CompiledPlan)
    assert first.work_unit_ids == second.work_unit_ids
    assert first.work_unit_ids != other_version.work_unit_ids


@pytest.mark.parametrize(
    "nodes",
    [
        (
            _node("a", depends_on=("b",)),
            _node("b", depends_on=("a",)),
        ),
        (
            _node("a", depends_on=("missing",)),
            _node("b"),
        ),
        (
            _node("same"),
            _node("same"),
        ),
        (
            _node("self", depends_on=("self",)),
        ),
    ],
)
def test_invalid_graphs_return_structured_rejection(
    nodes: Iterable[dict[str, object]],
) -> None:
    # model_construct lets the compiler's defensive checks be exercised even
    # when an earlier Planner contract would normally reject the same shape.
    raw_nodes = tuple(
        PlanNode.model_construct(
            key=node["key"],
            name=node["name"],
            instruction=node["instruction"],
            acceptance_criteria=node.get("acceptance_criteria"),
            output={"kind": ArtifactKind.TEXT},
            depends_on=tuple(node.get("depends_on", ())),
            resource_claims=(),
        )
        for node in nodes
    )
    raw = PlanProposal.model_construct(summary="unsafe fixture", nodes=raw_nodes)
    result = compile_(raw)
    assert isinstance(result, PlanRejection)
    assert result.summary == "The plan proposal was rejected"
    assert result.reasons


def test_conflicting_resource_claims_are_rejected_before_drafts() -> None:
    claims = (
        ResourceClaim(resource="file", access=ResourceAccess.READ),
        ResourceClaim(resource="file", access=ResourceAccess.WRITE),
    )
    node = PlanNode.model_construct(
        key="conflict",
        name="Conflict",
        instruction="read and write",
        acceptance_criteria=None,
        output={"kind": ArtifactKind.TEXT},
        depends_on=(),
        resource_claims=claims,
    )
    result = compile_(PlanProposal.model_construct(summary="unsafe", nodes=(node,)))
    assert isinstance(result, PlanRejection)
    assert result.reasons == ("plan node conflict declares conflicting resource claims",)


def test_invalid_run_id_and_graph_version_are_rejections() -> None:
    invalid_run = compile_plan(proposal(_node("only")), run_id="not-a-run", graph_version=1)
    invalid_version = compile_plan(proposal(_node("only")), run_id=RUN_ID, graph_version=0)
    assert isinstance(invalid_run, PlanRejection)
    assert isinstance(invalid_version, PlanRejection)


def test_compiler_has_no_store_dependency() -> None:
    import inspect

    source = inspect.getsource(compile_plan)
    assert "SqliteStore" not in source
    assert "async" not in source
