"""Shared inbound binding normalization contract tests."""

import pytest
from pydantic import ValidationError

from prp_runtime.api.bindings import (
    BindingOperation,
    normalize_cancel,
    normalize_query,
    normalize_request,
)
from prp_runtime.domain.enums import RoutingPolicy
from prp_runtime.domain.errors import ErrorCode, PrpError
from prp_runtime.domain.models import ArtifactKind


def test_normalize_request_builds_one_native_shape() -> None:
    result = normalize_request(
        {
            "input": "summarize this",
            "instructions": "be concise",
            "budget": {"max_total_tokens": 100},
            "output": {"kind": "JSON", "json_schema": '{"type":"object"}'},
        }
    )

    assert result.operation is BindingOperation.CREATE
    assert result.request is not None
    assert result.request.input == "summarize this"
    assert result.request.instructions == "be concise"
    assert result.request.budget.max_total_tokens == 100
    assert result.request.output.kind is ArtifactKind.JSON
    assert result.request.routing_policy is RoutingPolicy.AUTO


def test_normalize_request_accepts_public_routing_intent() -> None:
    result = normalize_request(
        {
            "input": "summarize this",
            "routing": {"requires_plan": True, "desired_parallelism": 2},
        }
    )

    assert result.request is not None
    assert result.request.routing is not None
    assert result.request.routing.requires_plan is True
    assert result.request.routing.desired_parallelism == 2


@pytest.mark.parametrize("field", ["retryable_failure", "api_key", "provider_alias"])
def test_routing_intent_rejects_non_public_facts_without_reading_values(
    field: str,
) -> None:
    with pytest.raises(PrpError) as excinfo:
        normalize_request({"input": "hello", "routing": {field: "redacted"}})

    assert excinfo.value.code is ErrorCode.UNSUPPORTED_FIELD
    assert "redacted" not in str(excinfo.value)


def test_manual_binding_rejects_routing_intent() -> None:
    with pytest.raises(PrpError) as excinfo:
        normalize_request(
            {
                "input": "hello",
                "routing_policy": "MANUAL",
                "strategy": "DIRECT",
                "routing": {"requires_cascade": True},
            }
        )

    assert excinfo.value.code is ErrorCode.INVALID_REQUEST


@pytest.mark.parametrize("field", ["unknown", "api_key", "base_url", "model"])
def test_unknown_or_credential_fields_are_rejected_without_reading_values(
    field: str,
) -> None:
    with pytest.raises(PrpError) as excinfo:
        normalize_request({"input": "hello", field: "redacted"})

    assert excinfo.value.code is ErrorCode.UNSUPPORTED_FIELD
    assert "redacted" not in str(excinfo.value)


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("stream", ErrorCode.UNSUPPORTED_STREAM_MODE),
        ("tools", ErrorCode.UNSUPPORTED_TOOLS),
        ("modalities", ErrorCode.UNSUPPORTED_MODALITY),
    ],
)
def test_unsupported_capabilities_have_stable_codes(field: str, code: ErrorCode) -> None:
    with pytest.raises(PrpError) as excinfo:
        normalize_request({"input": "hello", field: True})
    assert excinfo.value.code is code


def test_invalid_budget_and_output_are_structured() -> None:
    with pytest.raises(PrpError) as budget_error:
        normalize_request({"input": "hello", "budget": {"max_attempts": 0}})
    with pytest.raises(PrpError) as output_error:
        normalize_request({"input": "hello", "output": {"json_schema": "not json"}})
    assert budget_error.value.code is ErrorCode.INVALID_BUDGET
    assert output_error.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT


def test_query_and_cancel_only_validate_identifiers() -> None:
    query = normalize_query("run_123")
    cancel = normalize_cancel("run_123")
    assert query.operation is BindingOperation.QUERY
    assert cancel.operation is BindingOperation.CANCEL
    assert query.run_id == cancel.run_id == "run_123"
    assert query.request is None
    assert cancel.request is None

    with pytest.raises(PrpError) as excinfo:
        normalize_cancel("not-a-run")
    assert excinfo.value.code is ErrorCode.INVALID_REQUEST


def test_result_is_closed_and_immutable() -> None:
    result = normalize_request({"input": "hello"})
    with pytest.raises(ValidationError):
        result.operation = BindingOperation.QUERY  # type: ignore[misc]
