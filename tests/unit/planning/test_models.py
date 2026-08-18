"""Planner contracts are bounded, closed, and contain no executable payload."""

import json

import pytest
from pydantic import ValidationError

import prp_runtime.planning as planning
from prp_runtime.domain.enums import ResourceAccess
from prp_runtime.domain.models import ArtifactKind
from prp_runtime.json_support import StrictJsonError
from prp_runtime.planning.models import (
    MAX_PLAN_NODES,
    PlanNode,
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)


def _node(key: str = "draft", **overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "key": key,
        "name": f"Node {key}",
        "instruction": f"Produce {key}",
    }
    values.update(overrides)
    return values


def _proposal(*nodes: dict[str, object], **overrides: object) -> PlanProposal:
    values: dict[str, object] = {
        "summary": "Prepare and verify the requested result",
        "final_node": nodes[-1]["key"] if nodes else "draft",
        "nodes": nodes or (_node(),),
    }
    values.update(overrides)
    return PlanProposal.model_validate(values)


def test_valid_proposal_has_explicit_safe_defaults() -> None:
    proposal = _proposal(
        _node("draft"),
        _node(
            "review",
            depends_on=("draft",),
            acceptance_criteria="The result satisfies the declared output",
            output={"kind": ArtifactKind.JSON},
            resource_claims=(
                {"resource": "result", "access": ResourceAccess.WRITE},
            ),
        ),
    )
    assert [node.key for node in proposal.nodes] == ["draft", "review"]
    assert proposal.final_node == "review"
    assert [node.lineage_key for node in proposal.nodes] == ["draft", "review"]
    assert proposal.nodes[0].depends_on == ()
    assert proposal.nodes[0].resource_claims == ()
    assert proposal.nodes[0].output.kind is ArtifactKind.TEXT
    assert proposal.nodes[1].depends_on == ("draft",)
    assert proposal.nodes[1].output.kind is ArtifactKind.JSON


def test_models_are_frozen_and_reject_unknown_fields() -> None:
    proposal = _proposal()
    with pytest.raises(ValidationError, match="frozen"):
        proposal.summary = "replace"  # type: ignore[misc]
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _proposal(reasoning="private analysis")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlanNode.model_validate(_node(code="print('x')"))


@pytest.mark.parametrize(
    "field",
    [
        "reasoning",
        "chain_of_thought",
        "code",
        "command",
        "provider",
        "model",
        "base_url",
        "api_key",
    ],
)
def test_sensitive_or_executable_fields_are_not_in_the_contract(field: str) -> None:
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlanNode.model_validate(_node(**{field: "forbidden"}))


def test_proposal_requires_a_bounded_nonempty_graph() -> None:
    with pytest.raises(ValidationError):
        _proposal(nodes=())
    with pytest.raises(ValidationError):
        _proposal(*(_node(f"node_{index}") for index in range(MAX_PLAN_NODES + 1)))


def test_proposal_requires_one_known_final_node() -> None:
    with pytest.raises(ValidationError, match="Field required"):
        PlanProposal.model_validate({"summary": "missing", "nodes": (_node(),)})
    with pytest.raises(ValidationError, match="unknown plan final node"):
        _proposal(final_node="absent")
    with pytest.raises(ValidationError, match="extra_forbidden"):
        _proposal(final_nodes=("draft",))


def test_public_text_fields_have_hard_size_limits() -> None:
    with pytest.raises(ValidationError):
        _proposal(summary="s" * 4_097)
    with pytest.raises(ValidationError):
        PlanNode.model_validate(_node(instruction="i" * 16_385))


def test_duplicate_and_unknown_dependencies_are_rejected() -> None:
    with pytest.raises(ValidationError, match="duplicate plan node key"):
        _proposal(_node("same"), _node("same"))
    with pytest.raises(ValidationError, match="unknown plan node dependency"):
        _proposal(_node("only", depends_on=("absent",)))
    with pytest.raises(ValidationError, match="depend on itself"):
        _proposal(_node("self", depends_on=("self",)))
    with pytest.raises(ValidationError, match="duplicate plan node dependency"):
        _proposal(
            _node("base"),
            _node("next", depends_on=("base", "base")),
        )


def test_lineage_keys_are_unique_and_proposal_local() -> None:
    with pytest.raises(ValidationError, match="duplicate plan node lineage key"):
        _proposal(_node("first", lineage_key="same"), _node("second", lineage_key="same"))
    for lineage_key in ("wu_old", "run_old", "art_old"):
        with pytest.raises(ValidationError, match="proposal-local"):
            PlanNode.model_validate(_node("node", lineage_key=lineage_key))


@pytest.mark.parametrize(
    "dependency",
    ["has space", "UPPER", "../escape", "shell;run", "", "wu_persisted"],
)
def test_dependency_keys_are_local_restricted_text(dependency: str) -> None:
    with pytest.raises(ValidationError):
        PlanNode.model_validate(_node("node", depends_on=(dependency,)))


@pytest.mark.parametrize(
    "key", ["wu_persisted", "run_persisted", "att_persisted", "art_x", "ev_x"]
)
def test_node_keys_never_claim_to_be_persisted_ids(key: str) -> None:
    with pytest.raises(ValidationError, match="proposal-local"):
        PlanNode.model_validate(_node(key))


def test_duplicate_resource_claims_are_rejected() -> None:
    claim = {"resource": "document", "access": ResourceAccess.WRITE}
    with pytest.raises(ValidationError, match="duplicate plan node resource claim"):
        PlanNode.model_validate(_node(resource_claims=(claim, claim)))


def test_strict_json_entry_rejects_nonstandard_json_and_nonobjects() -> None:
    text = json.dumps(
        {
            "summary": "One node",
            "final_node": "draft",
            "nodes": [_node()],
        }
    )
    assert PlanProposal.from_json(text).nodes[0].key == "draft"
    with pytest.raises(StrictJsonError):
        PlanProposal.from_json('{"summary":"x","nodes":[],"score":NaN}')
    with pytest.raises(ValueError, match="JSON object"):
        PlanProposal.from_json("[]")


def test_revision_and_rejection_only_carry_public_structured_content() -> None:
    proposal = _proposal()
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
        summary="Replace the failed node arrangement",
        proposal=proposal,
    )
    rejection = PlanRejection(
        summary="The proposal cannot be compiled",
        reasons=("It references an unavailable resource",),
    )
    assert revision.base_graph_version == 2
    assert revision.proposal is proposal
    assert rejection.reasons == ("It references an unavailable resource",)
    assert (
        revision.model_validate_json(revision.model_dump_json()).proposal.nodes[0].lineage_key
        == "draft"
    )
    with pytest.raises(ValidationError, match="extra_forbidden"):
        PlanRevision.model_validate(revision.model_dump() | {"reasoning": "hidden"})
    with pytest.raises(ValidationError):
        PlanRejection(summary="Rejected", reasons=())


def test_planning_package_exports_only_the_contract_surface() -> None:
    assert set(planning.__all__) == {
        "MAX_PLAN_NODES",
        "PlanNode",
        "PlanNodeKey",
        "PlanProposal",
        "PlanRejection",
        "PlanRevision",
        "PlanRevisionReason",
        "Planner",
        "CompiledPlan",
        "CompiledWorkUnit",
        "NodeMapping",
        "compile_plan",
        "compile_proposal",
    }
