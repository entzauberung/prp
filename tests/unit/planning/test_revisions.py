"""Public PlanRevision reason and graph-version contracts."""

import json

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.models import NativeRunRequest, Usage
from prp_runtime.planning.models import (
    PlanNode,
    PlanProposal,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.planning.planner import Planner
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)


def proposal() -> PlanProposal:
    return PlanProposal(
        summary="replace the failed graph",
        nodes=(
            PlanNode(
                key="replacement",
                name="Replacement",
                instruction="produce a corrected result",
            ),
        ),
    )


@pytest.mark.parametrize(
    "reason",
    list(PlanRevisionReason),
)
def test_revision_reasons_are_public_and_structured(
    reason: PlanRevisionReason,
) -> None:
    revision = PlanRevision(
        base_graph_version=3,
        reason=reason,
        summary="bounded public revision",
        proposal=proposal(),
    )
    rendered = revision.model_dump_json()
    assert "reasoning" not in rendered
    assert "chain-of-thought" not in rendered
    assert revision.base_graph_version == 3


def test_revision_requires_a_positive_base_graph_version() -> None:
    with pytest.raises(ValidationError):
        PlanRevision(
            base_graph_version=0,
            reason=PlanRevisionReason.VERIFICATION_FAILED,
            summary="invalid base",
            proposal=proposal(),
        )


def test_inconclusive_verification_has_a_distinct_revision_reason() -> None:
    assert (
        PlanRevisionReason.VERIFICATION_INCONCLUSIVE.value
        == "VERIFICATION_INCONCLUSIVE"
    )


class RevisionAdapter:
    def __init__(self, revision: PlanRevision) -> None:
        self.revision = revision
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "revision-planner"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        return ProviderResponse(
            text=self.revision.model_dump_json(),
            usage=Usage(input_tokens=2, output_tokens=2),
            finish_reason=FinishReason.STOP,
        )


@pytest.mark.asyncio
async def test_planner_revision_request_is_strict_and_public() -> None:
    revision = PlanRevision(
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_INCONCLUSIVE,
        summary="replace the undecidable graph",
        proposal=proposal(),
    )
    adapter = RevisionAdapter(revision)
    planner = Planner(
        adapter,  # type: ignore[arg-type]
        ModelProfile(
            alias="planner",
            provider="fake",
            model="planner-model",
            role=ModelRole.PLANNER,
            base_url="https://models.invalid/v1",
            supports_structured_output=True,
            context_window_tokens=8_000,
            max_output_tokens=1_000,
        ),
    )

    result = await planner.revise(
        NativeRunRequest(input="repair the result"),
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_INCONCLUSIVE,
        feedback="the deterministic checker could not decide",
    )

    assert result == revision
    assert len(adapter.requests) == 1
    sent = adapter.requests[0]
    payload = json.loads(sent.input)
    assert payload["base_graph_version"] == 2
    assert payload["reason"] == "VERIFICATION_INCONCLUSIVE"
    assert payload["feedback"] == "the deterministic checker could not decide"
    assert "reasoning" not in sent.input
    assert "api_key" not in sent.input
