"""Pure deterministic compilation from Planner proposals to WorkUnit drafts."""

import hashlib
import heapq
from collections import defaultdict
from typing import Annotated, Any

from pydantic import Field, StringConstraints

from prp_runtime.domain.models import DomainModel, OutputRequirement
from prp_runtime.domain.values import ResourceClaim, RunId, WorkUnitId, validate_run_id
from prp_runtime.json_support import canonical_json_dumps
from prp_runtime.planning.models import (
    MAX_PLAN_NODES,
    PlanNode,
    PlanNodeKey,
    PlanProposal,
    PlanRejection,
)

__all__ = [
    "CompiledPlan",
    "CompiledWorkUnit",
    "NodeMapping",
    "compile_plan",
    "compile_proposal",
]

DraftName = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_096),
]
DraftInstruction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384),
]


class NodeMapping(DomainModel):
    """Stable proposal-local key to generated WorkUnit ID mapping."""

    node_key: PlanNodeKey
    work_unit_id: WorkUnitId


class CompiledWorkUnit(DomainModel):
    """A serializable WorkUnit draft, not a persisted state row."""

    work_unit_id: WorkUnitId
    run_id: RunId
    graph_version: int = Field(ge=1)
    node_key: PlanNodeKey
    lineage_key: PlanNodeKey
    dependency_fingerprint: str
    content_fingerprint: str
    name: DraftName
    instruction: DraftInstruction
    acceptance_criteria: DraftName | None = None
    output: OutputRequirement = Field(default_factory=OutputRequirement)
    depends_on: tuple[WorkUnitId, ...] = ()
    resource_claims: tuple[ResourceClaim, ...] = ()


class CompiledPlan(DomainModel):
    """A complete ordered DAG ready for a later controller commit."""

    run_id: RunId
    graph_version: int = Field(ge=1)
    final_node: PlanNodeKey
    final_work_unit_id: WorkUnitId
    nodes: tuple[CompiledWorkUnit, ...] = Field(
        min_length=1,
        max_length=MAX_PLAN_NODES,
    )
    node_map: tuple[NodeMapping, ...] = Field(
        min_length=1,
        max_length=MAX_PLAN_NODES,
    )

    @property
    def work_unit_ids(self) -> tuple[WorkUnitId, ...]:
        return tuple(node.work_unit_id for node in self.nodes)


def _reject(*reasons: str) -> PlanRejection:
    return PlanRejection(summary="The plan proposal was rejected", reasons=tuple(reasons))


def _stable_work_unit_id(run_id: str, graph_version: int, node_key: str) -> str:
    digest = hashlib.sha256(
        f"{run_id}:{graph_version}:{node_key}".encode()
    ).hexdigest()[:32]
    return f"wu_{digest}"


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


def _fingerprints(node: PlanNode, dependency_lineages: tuple[str, ...]) -> tuple[str, str]:
    dependency_fingerprint = _sha256_json(
        {"dependencies": dependency_lineages}
    )
    content_fingerprint = _sha256_json(
        {
            "acceptance_criteria": node.acceptance_criteria,
            "dependencies": dependency_lineages,
            "dependency_fingerprint": dependency_fingerprint,
            "instruction": node.instruction,
            "lineage_key": node.lineage_key,
            "output": node.output.model_dump(mode="json"),
            "resource_claims": [
                claim.model_dump(mode="json") for claim in node.resource_claims
            ],
        }
    )
    return dependency_fingerprint, content_fingerprint


def _topological_order(nodes: tuple[PlanNode, ...]) -> tuple[str, ...] | PlanRejection:
    by_key = {node.key: node for node in nodes}
    lineage_keys = tuple(getattr(node, "lineage_key", node.key) for node in nodes)
    if len(set(lineage_keys)) != len(lineage_keys):
        return _reject("duplicate plan node lineage key")
    if len(by_key) != len(nodes):
        return _reject("duplicate plan node key")

    indegree = {key: 0 for key in by_key}
    dependents: dict[str, list[str]] = defaultdict(list)
    for node in nodes:
        for dependency in node.depends_on:
            if dependency not in by_key:
                return _reject(f"unknown plan node dependency: {dependency}")
            if dependency == node.key:
                return _reject(f"plan node {node.key} cannot depend on itself")
            indegree[node.key] += 1
            dependents[dependency].append(node.key)

    ready = [key for key, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered: list[str] = []
    while ready:
        key = heapq.heappop(ready)
        ordered.append(key)
        for dependent in sorted(dependents[key]):
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(ordered) != len(nodes):
        return _reject("plan proposal contains a dependency cycle")
    return tuple(ordered)


def _validate_resources(nodes: tuple[PlanNode, ...]) -> PlanRejection | None:
    for node in nodes:
        claims = node.resource_claims
        for index, claim in enumerate(claims):
            for other in claims[index + 1 :]:
                if claim.conflicts_with(other):
                    return _reject(
                        f"plan node {node.key} declares conflicting resource claims"
                    )
    return None


def compile_plan(
    proposal: PlanProposal,
    *,
    run_id: str,
    graph_version: int,
) -> CompiledPlan | PlanRejection:
    """Compile a proposal without Store access or model-text execution."""
    if graph_version < 1:
        return _reject("graph_version must be at least 1")
    try:
        validate_run_id(run_id)
    except ValueError:
        return _reject("compiled plan has an invalid run_id")
    nodes = proposal.nodes
    if not nodes:
        return _reject("plan proposal must contain at least one node")
    if len(nodes) > MAX_PLAN_NODES:
        return _reject(f"plan proposal exceeds the {MAX_PLAN_NODES}-node limit")

    resource_rejection = _validate_resources(nodes)
    if resource_rejection is not None:
        return resource_rejection
    order = _topological_order(nodes)
    if isinstance(order, PlanRejection):
        return order

    # Construct every draft only after validation has completed. This keeps a
    # rejected graph from exposing a misleading partial compilation.
    by_key = {node.key: node for node in nodes}
    if proposal.final_node not in by_key:
        return _reject("unknown plan final node: " + proposal.final_node)
    if any(proposal.final_node in node.depends_on for node in nodes):
        return _reject("final node must have no dependents")
    node_ids = {
        key: _stable_work_unit_id(run_id, graph_version, key) for key in order
    }
    lineage_by_key = {node.key: node.lineage_key for node in nodes}
    drafts = tuple(
        CompiledWorkUnit(
            work_unit_id=node_ids[node.key],
            run_id=run_id,
            graph_version=graph_version,
            node_key=node.key,
            lineage_key=getattr(node, "lineage_key", node.key),
            dependency_fingerprint=_fingerprints(
                node,
                tuple(lineage_by_key[dependency] for dependency in node.depends_on),
            )[0],
            content_fingerprint=_fingerprints(
                node,
                tuple(lineage_by_key[dependency] for dependency in node.depends_on),
            )[1],
            name=node.name,
            instruction=node.instruction,
            acceptance_criteria=node.acceptance_criteria,
            output=node.output,
            depends_on=tuple(node_ids[dependency] for dependency in node.depends_on),
            resource_claims=node.resource_claims,
        )
        for node in (by_key[key] for key in order)
    )
    mapping = tuple(
        NodeMapping(node_key=key, work_unit_id=node_ids[key]) for key in order
    )
    try:
        return CompiledPlan(
            run_id=run_id,
            graph_version=graph_version,
            final_node=proposal.final_node,
            final_work_unit_id=node_ids[proposal.final_node],
            nodes=drafts,
            node_map=mapping,
        )
    except ValueError:
        # Invalid external identifiers remain a structured compile rejection.
        return _reject("compiled plan has invalid identifiers or fields")


compile_proposal = compile_plan
