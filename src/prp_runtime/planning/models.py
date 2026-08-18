"""Closed, non-executable Planner proposal contracts."""

from enum import StrEnum, unique
from typing import Annotated, Self

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)

from prp_runtime.domain.models import OutputRequirement
from prp_runtime.domain.values import ResourceClaim
from prp_runtime.json_support import strict_json_loads

__all__ = [
    "MAX_PLAN_NODES",
    "PlanNode",
    "PlanNodeKey",
    "PlanProposal",
    "PlanRejection",
    "PlanRevision",
    "PlanRevisionReason",
]

MAX_PLAN_NODES = 64
MAX_REJECTION_REASONS = 16


def _proposal_local_key(value: str) -> str:
    if value.startswith(("run_", "wu_", "att_", "art_", "ev_")):
        raise ValueError("a plan node key must be proposal-local, not a persisted id")
    return value


PlanNodeKey = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_-]*$",
    ),
    AfterValidator(_proposal_local_key),
]
PlanText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4_096),
]
PlanInstruction = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=16_384),
]


class PlanningModel(BaseModel):
    """Immutable closed base with the runtime's strict JSON entry point."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    @classmethod
    def from_json(cls, text: str) -> Self:
        value = strict_json_loads(text)
        if not isinstance(value, dict):
            raise ValueError("a planning document must be a JSON object")
        return cls.model_validate(value)


class PlanNode(PlanningModel):
    """One proposal-local unit before persistent WorkUnit IDs are assigned."""

    key: PlanNodeKey
    lineage_key: PlanNodeKey
    name: PlanText
    instruction: PlanInstruction
    acceptance_criteria: PlanText | None = None
    output: OutputRequirement = Field(default_factory=OutputRequirement)
    depends_on: tuple[PlanNodeKey, ...] = ()
    resource_claims: tuple[ResourceClaim, ...] = ()

    @model_validator(mode="before")
    @classmethod
    def _default_lineage_to_local_key(cls, value: object) -> object:
        if isinstance(value, dict) and "lineage_key" not in value and "key" in value:
            return {**value, "lineage_key": value["key"]}
        return value

    @model_validator(mode="after")
    def _references_and_claims_are_sane(self) -> "PlanNode":
        if self.key in self.depends_on:
            raise ValueError("a plan node cannot depend on itself")
        if len(set(self.depends_on)) != len(self.depends_on):
            raise ValueError("duplicate plan node dependency")
        claim_keys = [(claim.resource, claim.access) for claim in self.resource_claims]
        if len(set(claim_keys)) != len(claim_keys):
            raise ValueError("duplicate plan node resource claim")
        return self


class PlanProposal(PlanningModel):
    """A bounded public summary plus structured graph input."""

    summary: PlanText
    final_node: PlanNodeKey
    nodes: tuple[PlanNode, ...] = Field(min_length=1, max_length=MAX_PLAN_NODES)

    @model_validator(mode="after")
    def _node_keys_and_dependencies_are_closed(self) -> "PlanProposal":
        keys = tuple(node.key for node in self.nodes)
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate plan node key")
        lineage_keys = tuple(node.lineage_key for node in self.nodes)
        if len(set(lineage_keys)) != len(lineage_keys):
            raise ValueError("duplicate plan node lineage key")
        known = set(keys)
        unknown = sorted(
            {
                dependency
                for node in self.nodes
                for dependency in node.depends_on
                if dependency not in known
            }
        )
        if unknown:
            raise ValueError("unknown plan node dependency: " + ", ".join(unknown))
        if self.final_node not in known:
            raise ValueError("unknown plan final node: " + self.final_node)
        return self


@unique
class PlanRevisionReason(StrEnum):
    """Public, deterministic reasons a replacement proposal may be requested."""

    VERIFICATION_FAILED = "VERIFICATION_FAILED"
    VERIFICATION_INCONCLUSIVE = "VERIFICATION_INCONCLUSIVE"
    DEPENDENCY_FAILED = "DEPENDENCY_FAILED"
    PROVIDER_FAILED = "PROVIDER_FAILED"
    BUDGET_CONSTRAINT = "BUDGET_CONSTRAINT"
    REQUIREMENT_CHANGED = "REQUIREMENT_CHANGED"


class PlanRevision(PlanningModel):
    """A replacement proposal tied to the graph version it revises."""

    base_graph_version: int = Field(ge=1)
    reason: PlanRevisionReason
    summary: PlanText
    proposal: PlanProposal


class PlanRejection(PlanningModel):
    """A public rejection without private reasoning or provider content."""

    summary: PlanText
    reasons: tuple[PlanText, ...] = Field(
        min_length=1,
        max_length=MAX_REJECTION_REASONS,
    )
