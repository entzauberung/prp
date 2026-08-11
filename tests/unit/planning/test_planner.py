"""The Planner is a stateless, strict adapter around one fake provider call."""

import asyncio
import json

import pytest

from prp_runtime.domain.enums import ModelRole, WorkUnitStatus
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import ArtifactKind, ErrorCategory, NativeRunRequest, Usage
from prp_runtime.json_support import strict_json_loads
from prp_runtime.planning.models import (
    PlanProposal,
    PlanRejection,
    PlanRevision,
    PlanRevisionReason,
)
from prp_runtime.planning.planner import (
    PLANNING_GRAPH_VERSION,
    PLANNING_WORK_UNIT_NAME,
    Planner,
    new_planning_work_unit,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderRequest,
    ProviderResponse,
)


def _profile(**overrides: object) -> ModelProfile:
    values: dict[str, object] = {
        "alias": "planner",
        "provider": "openai_compatible",
        "model": "planning-model",
        "role": ModelRole.PLANNER,
        "base_url": "https://models.internal/v1",
        "api_key": "sk-server-only",
        "supports_structured_output": True,
        "context_window_tokens": 64_000,
        "max_output_tokens": 8_000,
    }
    values.update(overrides)
    return ModelProfile.model_validate(values)


def _proposal_text(**overrides: object) -> str:
    values: dict[str, object] = {
        "summary": "Draft then review the result",
        "nodes": [
            {
                "key": "draft",
                "name": "Draft",
                "instruction": "Create the requested result",
            },
            {
                "key": "review",
                "name": "Review",
                "instruction": "Check the result",
                "acceptance_criteria": "The declared output is satisfied",
                "depends_on": ["draft"],
            },
        ],
    }
    values.update(overrides)
    return json.dumps(values)


class FakeAdapter:
    def __init__(self, outcome: object) -> None:
        self.outcome = outcome
        self.requests: list[ProviderRequest] = []

    @property
    def name(self) -> str:
        return "planner-fake"

    async def complete(self, request: ProviderRequest) -> ProviderResponse:
        self.requests.append(request)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        assert isinstance(self.outcome, ProviderResponse)
        return self.outcome


def _response(
    text: str,
    *,
    finish_reason: FinishReason = FinishReason.STOP,
) -> ProviderResponse:
    return ProviderResponse(
        text=text,
        usage=Usage(input_tokens=20, output_tokens=30, elapsed_ms=4),
        finish_reason=finish_reason,
    )


@pytest.mark.asyncio
async def test_planner_builds_one_structured_request_and_returns_a_proposal() -> None:
    adapter = FakeAdapter(_response(_proposal_text()))
    planner = Planner(adapter, _profile())  # type: ignore[arg-type]
    request = NativeRunRequest(
        input="prepare a report",
        instructions="use the declared evidence",
        output={"kind": ArtifactKind.JSON},
    )

    result = await planner.propose(request)

    assert isinstance(result, PlanProposal)
    assert [node.key for node in result.nodes] == ["draft", "review"]
    assert len(adapter.requests) == 1
    sent = adapter.requests[0]
    assert sent.alias == "planner"
    assert sent.json_schema is not None
    assert isinstance(strict_json_loads(sent.json_schema), dict)
    summary = strict_json_loads(sent.input)
    assert summary == {
        "objective": "prepare a report",
        "user_instructions": "use the declared evidence",
        "output": {"kind": "JSON", "json_schema": None},
        "max_nodes": 64,
    }
    assert "reasoning" in (sent.instructions or "")
    assert "base_url" not in ProviderRequest.model_fields
    assert "api_key" not in ProviderRequest.model_fields
    assert not hasattr(planner, "_store")


@pytest.mark.asyncio
async def test_propose_call_preserves_accounting_facts_without_raw_response() -> None:
    response = ProviderResponse(
        text=_proposal_text(),
        usage=Usage(input_tokens=20, output_tokens=30, elapsed_ms=4),
        finish_reason=FinishReason.STOP,
        provider_request_id="planner-request-1",
    )
    result = await Planner(FakeAdapter(response), _profile()).propose_call(  # type: ignore[arg-type]
        NativeRunRequest(input="prepare a report")
    )

    assert result.role is ModelRole.PLANNER
    assert result.model == _profile().model_ref
    assert isinstance(result.proposal, PlanProposal)
    assert result.rejection is None
    assert result.usage == Usage(
        input_tokens=20,
        output_tokens=30,
        strong_model_tokens=50,
        elapsed_ms=4,
    )
    assert result.finish_reason is FinishReason.STOP
    assert result.provider_request_id == "planner-request-1"
    assert result.error is None
    assert result.succeeded is True
    assert "text" not in type(result).model_fields


@pytest.mark.asyncio
async def test_propose_call_keeps_unreported_usage_unknown() -> None:
    response = ProviderResponse(
        text=_proposal_text(),
        finish_reason=FinishReason.STOP,
    )
    result = await Planner(FakeAdapter(response), _profile()).propose_call(  # type: ignore[arg-type]
        NativeRunRequest(input="prepare a report")
    )

    assert result.succeeded is True
    assert result.usage is None


@pytest.mark.asyncio
async def test_propose_call_classifies_provider_failure_without_sensitive_text() -> None:
    secret = "sk-secret at /home/private/request.json"
    result = await Planner(  # type: ignore[arg-type]
        FakeAdapter(ProviderError(secret, code=ErrorCode.PROVIDER_TIMEOUT)),
        _profile(),
    ).propose_call(NativeRunRequest(input="prepare a report"))

    assert result.rejection is not None
    assert result.error is not None
    assert result.error.category is ErrorCategory.TIMEOUT
    assert result.finish_reason is None
    assert result.usage is None
    assert result.succeeded is False
    assert secret not in result.model_dump_json()


@pytest.mark.asyncio
async def test_revise_call_returns_a_parsed_revision_with_usage() -> None:
    revision_text = json.dumps(
        {
            "base_graph_version": 2,
            "reason": "VERIFICATION_FAILED",
            "summary": "Replace the failed graph",
            "proposal": json.loads(_proposal_text()),
        }
    )
    result = await Planner(  # type: ignore[arg-type]
        FakeAdapter(_response(revision_text)),
        _profile(),
    ).revise_call(
        NativeRunRequest(input="prepare a report"),
        base_graph_version=2,
        reason=PlanRevisionReason.VERIFICATION_FAILED,
    )

    assert isinstance(result.proposal, PlanRevision)
    assert result.proposal.base_graph_version == 2
    assert result.usage == Usage(
        input_tokens=20,
        output_tokens=30,
        strong_model_tokens=50,
        elapsed_ms=4,
    )
    assert result.error is None


def test_internal_planning_work_unit_is_generic_and_graph_isolated() -> None:
    unit = new_planning_work_unit("run_123")

    assert unit.graph_version == PLANNING_GRAPH_VERSION == 1
    assert unit.name == PLANNING_WORK_UNIT_NAME
    assert unit.status is WorkUnitStatus.RUNNING
    assert unit.depends_on == ()
    assert unit.resource_claims == ()
    assert "run_123" not in unit.instruction


@pytest.mark.parametrize(
    "text",
    [
        "not json",
        '{"summary":"bad","nodes":[],"score":NaN}',
        _proposal_text(reasoning="private"),
        _proposal_text(api_key="test-token-must-not-be-retained"),
        _proposal_text(tools=[{"type": "shell"}]),
        _proposal_text(nodes=[]),
        _proposal_text(
            nodes=[
                {
                    "key": "only",
                    "name": "Only",
                    "instruction": "Work",
                    "depends_on": ["missing"],
                }
            ]
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_planner_output_becomes_a_stable_rejection(text: str) -> None:
    adapter = FakeAdapter(_response(text))
    result = await Planner(adapter, _profile()).propose(  # type: ignore[arg-type]
        NativeRunRequest(input="hello")
    )
    assert result == PlanRejection(
        summary="The Planner response was rejected",
        reasons=("response is not a valid PlanProposal",),
    )
    assert text not in result.model_dump_json()


@pytest.mark.asyncio
async def test_provider_error_is_redacted_to_its_stable_code() -> None:
    secret_message = "sk-secret at /home/private/request.json"
    adapter = FakeAdapter(
        ProviderError(secret_message, code=ErrorCode.PROVIDER_TIMEOUT)
    )
    result = await Planner(adapter, _profile()).propose(  # type: ignore[arg-type]
        NativeRunRequest(input="hello")
    )
    assert isinstance(result, PlanRejection)
    rendered = result.model_dump_json()
    assert result.reasons == ("provider error: provider_timeout",)
    assert secret_message not in rendered
    assert "sk-secret" not in rendered
    assert "/home/" not in rendered


@pytest.mark.asyncio
async def test_wrong_role_or_missing_structured_capability_is_rejected_before_call() -> None:
    response = _response(_proposal_text())
    wrong_role = FakeAdapter(response)
    role_result = await Planner(  # type: ignore[arg-type]
        wrong_role,
        _profile(role=ModelRole.WORKER),
    ).propose(NativeRunRequest(input="hello"))
    assert isinstance(role_result, PlanRejection)
    assert role_result.reasons == ("provider error: provider_not_configured",)
    assert wrong_role.requests == []

    no_schema = FakeAdapter(response)
    schema_result = await Planner(  # type: ignore[arg-type]
        no_schema,
        _profile(supports_structured_output=False),
    ).propose(NativeRunRequest(input="hello"))
    assert isinstance(schema_result, PlanRejection)
    assert schema_result.reasons == (
        "request error: invalid_output_requirement",
    )
    assert no_schema.requests == []


@pytest.mark.asyncio
async def test_non_stop_finish_reason_is_a_rejection() -> None:
    adapter = FakeAdapter(
        _response(_proposal_text(), finish_reason=FinishReason.LENGTH)
    )
    result = await Planner(adapter, _profile()).propose(  # type: ignore[arg-type]
        NativeRunRequest(input="hello")
    )
    assert result == PlanRejection(
        summary="The Planner did not complete a proposal",
        reasons=("provider finish reason: LENGTH",),
    )


@pytest.mark.asyncio
async def test_cancelled_error_propagates_without_a_rejection() -> None:
    adapter = FakeAdapter(asyncio.CancelledError())
    planner = Planner(adapter, _profile())  # type: ignore[arg-type]
    with pytest.raises(asyncio.CancelledError):
        await planner.propose(NativeRunRequest(input="hello"))
