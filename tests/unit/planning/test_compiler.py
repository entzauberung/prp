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
    values = nodes or (_node("only"),)
    return PlanProposal(
        summary="compile this graph",
        final_node=values[0]["key"],
        nodes=values,
    )


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
    assert result.final_node == "only"
    assert result.final_work_unit_id == result.nodes[0].work_unit_id
    assert result.nodes[0].lineage_key == "only"
    assert len(result.nodes[0].dependency_fingerprint) == 64
    assert len(result.nodes[0].content_fingerprint) == 64


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
    assert result.final_node == "finish"
    assert result.final_work_unit_id == by_key["finish"].work_unit_id


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


def test_lineage_is_stable_across_graph_versions_while_ids_change() -> None:
    value = PlanProposal(
        summary="revise one node",
        final_node="finish",
        nodes=(
            PlanNode(
                key="finish",
                lineage_key="shared-finish",
                name="Finish",
                instruction="produce finish",
            ),
        ),
    )
    first = compile_(value, version=2)
    second = compile_(value, version=3)
    assert isinstance(first, CompiledPlan)
    assert isinstance(second, CompiledPlan)
    assert first.nodes[0].lineage_key == second.nodes[0].lineage_key == "shared-finish"
    assert first.nodes[0].work_unit_id != second.nodes[0].work_unit_id


def test_fingerprints_are_stable_for_public_contract_and_ignore_persisted_facts() -> None:
    value = proposal(
        _node(
            "finish",
            instruction="same instruction",
            acceptance_criteria="same criteria",
            resource_claims=(
                {"resource": "result", "access": ResourceAccess.WRITE},
            ),
        )
    )
    first = compile_plan(value, run_id=RUN_ID, graph_version=2)
    second = compile_plan(value, run_id=new_run_id(), graph_version=9)
    assert isinstance(first, CompiledPlan)
    assert isinstance(second, CompiledPlan)
    assert first.nodes[0].dependency_fingerprint == second.nodes[0].dependency_fingerprint
    assert first.nodes[0].content_fingerprint == second.nodes[0].content_fingerprint


@pytest.mark.parametrize(
    "change",
    [
        {"instruction": "changed"},
        {"acceptance_criteria": "changed"},
        {"output": {"kind": ArtifactKind.JSON, "json_schema": '{"type":"object"}'}},
        {"resource_claims": ({"resource": "other", "access": ResourceAccess.WRITE},)},
    ],
)
def test_execution_contract_changes_change_content_fingerprint(
    change: dict[str, object],
) -> None:
    base = compile_(proposal(_node("finish")))
    changed = compile_(proposal(_node("finish", **change)))
    assert isinstance(base, CompiledPlan)
    assert isinstance(changed, CompiledPlan)
    assert base.nodes[0].content_fingerprint != changed.nodes[0].content_fingerprint


def test_dependency_lineage_changes_change_dependency_and_content_fingerprints() -> None:
    first = compile_(
        PlanProposal(
            summary="same graph",
            final_node="finish",
            nodes=(
                PlanNode(key="base", lineage_key="base-v1", name="Base", instruction="base"),
                PlanNode(key="finish", name="Finish", instruction="finish", depends_on=("base",)),
            ),
        )
    )
    second = compile_(
        PlanProposal(
            summary="same graph",
            final_node="finish",
            nodes=(
                PlanNode(key="base", lineage_key="base-v2", name="Base", instruction="base"),
                PlanNode(key="finish", name="Finish", instruction="finish", depends_on=("base",)),
            ),
        )
    )
    assert isinstance(first, CompiledPlan)
    assert isinstance(second, CompiledPlan)
    first_finish = next(node for node in first.nodes if node.node_key == "finish")
    second_finish = next(node for node in second.nodes if node.node_key == "finish")
    assert first_finish.dependency_fingerprint != second_finish.dependency_fingerprint
    assert first_finish.content_fingerprint != second_finish.content_fingerprint


def test_declared_final_node_must_be_a_terminal_node() -> None:
    result = compile_plan(
        PlanProposal(
            summary="choose the output leaf",
            final_node="root",
            nodes=(
                PlanNode(
                    key="root",
                    name="Root",
                    instruction="produce root",
                ),
                PlanNode(
                    key="leaf",
                    name="Leaf",
                    instruction="produce leaf",
                    depends_on=("root",),
                ),
            ),
        ),
        run_id=RUN_ID,
        graph_version=3,
    )
    assert isinstance(result, PlanRejection)
    assert result.reasons == ("final node must have no dependents",)


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
            lineage_key=node["key"],
            name=node["name"],
            instruction=node["instruction"],
            acceptance_criteria=node.get("acceptance_criteria"),
            output={"kind": ArtifactKind.TEXT},
            depends_on=tuple(node.get("depends_on", ())),
            resource_claims=(),
        )
        for node in nodes
    )
    raw = PlanProposal.model_construct(
        summary="unsafe fixture",
        final_node=raw_nodes[-1].key,
        nodes=raw_nodes,
    )
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
    result = compile_(
        PlanProposal.model_construct(
            summary="unsafe",
            final_node="conflict",
            nodes=(node,),
        )
    )
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
