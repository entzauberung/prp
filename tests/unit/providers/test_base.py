"""Targeted tests for the provider contract and server-side model profiles."""

import json
from decimal import Decimal

import pytest
from pydantic import SecretStr, ValidationError

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import DomainValidationError, ErrorCode, ProviderError
from prp_runtime.domain.models import (
    MAX_PROVIDER_TOOL_COUNT,
    MAX_PROVIDER_TOOL_DESCRIPTOR_BYTES,
    AgentToolCall,
    ProviderToolDescriptor,
    Usage,
)
from prp_runtime.providers.base import (
    FinishReason,
    ModelProfile,
    ProviderAdapter,
    ProviderRequest,
    ProviderResponse,
)
from prp_runtime.settings import Settings

SECRET = "sk-not-a-real-key-0123456789"

LEADER = {
    "alias": "leader",
    "provider": "openai_compatible",
    "model": "strong-model",
    "role": ModelRole.PLANNER,
    "base_url": "https://models.internal/v1",
    "api_key": SECRET,
    "supports_structured_output": True,
    "context_window_tokens": 128_000,
    "max_output_tokens": 8_000,
    "input_price_per_million_tokens": 3.0,
    "output_price_per_million_tokens": 15.0,
    "max_concurrency": 2,
}

WORKER = {
    "alias": "worker",
    "provider": "openai_compatible",
    "model": "weak-model",
    "role": ModelRole.WORKER,
    "base_url": "https://models.internal/v1",
    "context_window_tokens": 32_000,
    "max_output_tokens": 4_000,
    "max_concurrency": 4,
}


def leader_profile(**overrides: object) -> ModelProfile:
    return ModelProfile(**{**LEADER, **overrides})  # type: ignore[arg-type]


def worker_profile(**overrides: object) -> ModelProfile:
    return ModelProfile(**{**WORKER, **overrides})  # type: ignore[arg-type]


# --- model profile --------------------------------------------------------------


def test_profile_exposes_the_domain_model_reference() -> None:
    profile = worker_profile()
    assert profile.model_ref.provider == "openai_compatible"
    assert profile.model_ref.model == "weak-model"
    assert profile.model_ref.identifier == "openai_compatible/weak-model"
    assert profile.timeout_seconds == 60.0
    assert profile.supports_structured_output is False
    assert profile.input_price_per_million_tokens == Decimal("0")
    assert profile.output_price_per_million_tokens == Decimal("0")


def test_profile_cost_uses_exact_decimal_arithmetic_and_preserves_unknown_usage() -> None:
    profile = worker_profile(
        input_price_per_million_tokens=Decimal("0.1"),
        output_price_per_million_tokens=Decimal("0.2"),
    )
    first = profile.cost_for_usage(Usage(input_tokens=1, output_tokens=2))
    second = profile.cost_for_usage(Usage(input_tokens=3, output_tokens=4))
    assert first is not None and second is not None
    assert first.total_cost + second.total_cost == Decimal("0.0000016")
    assert profile.cost_for_usage(None) is None


def test_profile_prices_round_trip_as_decimal_values() -> None:
    profile = worker_profile(input_price_per_million_tokens=Decimal("0.10"))
    restored = ModelProfile.model_validate_json(profile.model_dump_json())
    assert restored == profile
    assert restored.input_price_per_million_tokens == Decimal("0.10")


def test_api_key_never_leaks_into_repr_or_serialisation() -> None:
    profile = leader_profile()
    assert isinstance(profile.api_key, SecretStr)
    assert profile.api_key.get_secret_value() == SECRET
    assert SECRET not in repr(profile)
    assert SECRET not in str(profile)
    assert SECRET not in profile.model_dump_json()
    assert SECRET not in json.dumps(profile.model_dump(mode="json"))


def test_profile_declares_no_tool_or_multimodal_capability() -> None:
    forbidden = {"tools", "tool_choice", "functions", "images", "audio", "vision", "modalities"}
    assert forbidden.isdisjoint(ModelProfile.model_fields)


def test_profile_rejects_unknown_fields_and_is_frozen() -> None:
    with pytest.raises(ValidationError):
        worker_profile(supports_tools=True)
    profile = worker_profile()
    with pytest.raises(ValidationError):
        profile.model = "other"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_price_per_million_tokens", -0.1),
        ("output_price_per_million_tokens", -1.0),
        ("max_concurrency", 0),
        ("context_window_tokens", 0),
        ("max_output_tokens", 0),
        ("timeout_seconds", 0.0),
    ],
)
def test_profile_rejects_non_positive_limits_and_negative_prices(
    field: str, value: object
) -> None:
    with pytest.raises(ValidationError):
        worker_profile(**{field: value})


def test_profile_output_limit_must_fit_the_context_window() -> None:
    with pytest.raises(ValidationError):
        worker_profile(context_window_tokens=1_000, max_output_tokens=2_000)


@pytest.mark.parametrize("bad_url", ["models.internal", "ftp://models.internal", ""])
def test_profile_requires_an_http_endpoint(bad_url: str) -> None:
    with pytest.raises(ValidationError):
        worker_profile(base_url=bad_url)


@pytest.mark.parametrize("bad_alias", ["", "Leader", "lead er", "-leader"])
def test_profile_alias_is_a_strict_label(bad_alias: str) -> None:
    with pytest.raises(ValidationError):
        worker_profile(alias=bad_alias)


# --- provider request and response ----------------------------------------------


def test_request_cannot_carry_an_endpoint_or_a_credential() -> None:
    assert "base_url" not in ProviderRequest.model_fields
    assert "api_key" not in ProviderRequest.model_fields
    with pytest.raises(ValidationError):
        ProviderRequest(
            alias="worker",
            model="weak-model",
            input="hello",
            max_output_tokens=10,
            timeout_seconds=5.0,
            base_url="https://attacker.example",
        )


def test_request_for_profile_uses_server_side_values() -> None:
    profile = worker_profile()
    request = ProviderRequest.for_profile(profile, input="summarise", instructions="be terse")
    assert request.alias == "worker"
    assert request.model == "weak-model"
    assert request.max_output_tokens == profile.max_output_tokens
    assert request.timeout_seconds == profile.timeout_seconds
    assert request.json_schema is None
    assert request.tools == ()


def test_provider_tool_descriptor_is_public_metadata_only_and_json_serialisable() -> None:
    descriptor = ProviderToolDescriptor(
        name="read_file",
        description="Read one relative file.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    )
    request = ProviderRequest.for_profile(worker_profile(), input="x", tools=(descriptor,))

    assert request.tools == (descriptor,)
    assert request.model_dump(mode="json")["tools"] == [descriptor.model_dump(mode="json")]
    assert "handler" not in descriptor.model_dump(mode="json")
    assert "effect" not in descriptor.model_dump(mode="json")
    assert "executable" not in descriptor.model_dump(mode="json")
    assert "path_root" not in descriptor.model_dump(mode="json")


@pytest.mark.parametrize("field", ["handler", "effect", "executable", "path_root", "api_key"])
def test_provider_tool_descriptor_rejects_execution_and_secret_fields(field: str) -> None:
    with pytest.raises(ValidationError):
        ProviderToolDescriptor(name="read_file", **{field: "forbidden"})  # type: ignore[arg-type]


def test_provider_tool_descriptor_is_frozen_and_rejects_oversized_schema() -> None:
    descriptor = ProviderToolDescriptor(name="read_file")
    with pytest.raises(ValidationError):
        descriptor.name = "other"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ProviderToolDescriptor(
            name="read_file",
            input_schema={"description": "x" * MAX_PROVIDER_TOOL_DESCRIPTOR_BYTES},
        )


def test_provider_request_rejects_duplicate_and_unbounded_tools() -> None:
    descriptor = ProviderToolDescriptor(name="read_file")
    with pytest.raises(ValidationError, match="unique"):
        ProviderRequest.for_profile(
            worker_profile(), input="x", tools=(descriptor, descriptor)
        )
    with pytest.raises(ValidationError, match="too many"):
        ProviderRequest.for_profile(
            worker_profile(),
            input="x",
            tools=tuple(
                ProviderToolDescriptor(name=f"tool_{index}")
                for index in range(MAX_PROVIDER_TOOL_COUNT + 1)
            ),
        )
    with pytest.raises(ValidationError, match="size limit"):
        ProviderRequest.for_profile(
            worker_profile(),
            input="x",
            tools=tuple(
                ProviderToolDescriptor(name=f"tool_{index}", description="x" * 1_000)
                for index in range(MAX_PROVIDER_TOOL_COUNT)
            ),
        )


def test_request_rejects_structured_output_when_unsupported() -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        ProviderRequest.for_profile(
            worker_profile(), input="x", json_schema='{"type":"object"}'
        )
    assert excinfo.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT
    assert excinfo.value.detail.field == "json_schema"
    allowed = ProviderRequest.for_profile(
        leader_profile(), input="x", json_schema='{"type":"object"}'
    )
    assert allowed.json_schema == '{"type":"object"}'


def test_request_rejects_an_output_limit_above_the_profile() -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        ProviderRequest.for_profile(worker_profile(), input="x", max_output_tokens=99_999)
    assert excinfo.value.code is ErrorCode.INVALID_REQUEST
    assert (
        ProviderRequest.for_profile(
            worker_profile(), input="x", max_output_tokens=10
        ).max_output_tokens
        == 10
    )


def test_request_requires_non_empty_input() -> None:
    with pytest.raises(ValidationError):
        ProviderRequest.for_profile(worker_profile(), input="")


def test_response_carries_text_usage_finish_reason_and_request_id() -> None:
    response = ProviderResponse(
        text="the answer",
        usage=Usage(input_tokens=10, output_tokens=4, elapsed_ms=25),
        finish_reason=FinishReason.STOP,
        provider_request_id="req_1",
    )
    assert set(ProviderResponse.model_fields) == {
        "text",
        "tool_calls",
        "usage",
        "finish_reason",
        "provider_request_id",
    }
    assert ProviderResponse.model_validate_json(response.model_dump_json()) == response
    assert [member.value for member in FinishReason] == [
        "STOP",
        "LENGTH",
        "CONTENT_FILTER",
        "TOOL_CALLS",
        "OTHER",
    ]


def test_provider_response_rejects_blank_or_mislabeled_tool_turns() -> None:
    call = ProviderToolDescriptor(name="read_file")
    tool_call = AgentToolCall(call_id="call-1", tool_name=call.name)
    with pytest.raises(ValidationError):
        ProviderResponse(text=" \n", finish_reason=FinishReason.STOP)
    with pytest.raises(ValidationError):
        ProviderResponse(tool_calls=(tool_call,), finish_reason=FinishReason.STOP)
    with pytest.raises(ValidationError):
        ProviderResponse(
            tool_calls=(tool_call, tool_call),
            finish_reason=FinishReason.TOOL_CALLS,
        )


def test_provider_request_rejects_orphaned_tool_results() -> None:
    with pytest.raises(ValidationError, match="orphaned"):
        ProviderRequest(
            alias="worker",
            model="model",
            input="continue",
            max_output_tokens=10,
            timeout_seconds=5,
            history=(
                {
                    "kind": "tool_result",
                    "call_id": "call-1",
                    "status": "SUCCEEDED",
                },
            ),
        )


def test_adapter_protocol_matches_a_minimal_implementation() -> None:
    class StubAdapter:
        @property
        def name(self) -> str:
            return "stub"

        async def aclose(self) -> None:
            return None

        async def complete(self, request: ProviderRequest) -> ProviderResponse:
            return ProviderResponse(
                text="stub", usage=Usage(), finish_reason=FinishReason.STOP
            )

    assert isinstance(StubAdapter(), ProviderAdapter)
    assert not isinstance(object(), ProviderAdapter)


# --- settings integration -------------------------------------------------------


def test_settings_hold_optional_leader_and_worker_profiles() -> None:
    settings = Settings()
    assert settings.profiles == ()
    configured = Settings(
        leader_profile=leader_profile(),
        worker_profile=worker_profile(),
        cascade_profiles=(worker_profile(alias="worker-large"),),
    )
    assert [profile.alias for profile in configured.profiles] == [
        "leader",
        "worker",
        "worker-large",
    ]
    assert configured.require_profile(ModelRole.PLANNER).model == "strong-model"
    assert configured.require_profile(ModelRole.WORKER).model == "weak-model"
    assert configured.profile_by_alias("worker").alias == "worker"
    assert configured.profile_by_alias("worker-large").alias == "worker-large"


def test_settings_reject_a_role_mismatch_and_duplicate_alias() -> None:
    with pytest.raises(ValidationError):
        Settings(leader_profile=worker_profile(alias="leader"))
    with pytest.raises(ValidationError):
        Settings(worker_profile=leader_profile(alias="worker"))
    with pytest.raises(ValidationError):
        Settings(
            leader_profile=leader_profile(alias="same"),
            worker_profile=worker_profile(alias="same"),
        )
    with pytest.raises(ValidationError):
        Settings(
            worker_profile=worker_profile(alias="same"),
            cascade_profiles=(worker_profile(alias="same"),),
        )
    with pytest.raises(ValidationError):
        Settings(
            cascade_profiles=(
                worker_profile(alias="same"),
                worker_profile(alias="same"),
            )
        )


def test_unknown_alias_and_missing_role_fail_with_structured_errors() -> None:
    settings = Settings(worker_profile=worker_profile())
    with pytest.raises(ProviderError) as alias_error:
        settings.profile_by_alias("leader")
    assert alias_error.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
    assert alias_error.value.retryable is False
    with pytest.raises(ProviderError) as role_error:
        settings.require_profile(ModelRole.PLANNER)
    assert role_error.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED


def test_profiles_load_from_server_environment_only() -> None:
    settings = Settings.from_env(
        {
            "PRP_LEADER_PROFILE": json.dumps({**LEADER, "role": "PLANNER"}),
            "PRP_WORKER_PROFILE": json.dumps({**WORKER, "role": "WORKER"}),
        }
    )
    assert settings.leader_profile is not None
    assert settings.worker_profile is not None
    assert settings.leader_profile.api_key is not None
    assert settings.leader_profile.api_key.get_secret_value() == SECRET
    assert SECRET not in settings.model_dump_json()


def test_malformed_profile_environment_value_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_WORKER_PROFILE": "{not json"})
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_WORKER_PROFILE": json.dumps({"alias": "worker"})})
    with pytest.raises(ValueError, match="PRP_LEADER_URL"):
        Settings.from_env({"PRP_LEADER_URL": "https://models.internal"})


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999"])
def test_profile_environment_value_rejects_non_standard_numbers(token: str) -> None:
    """A non-finite number inside an otherwise valid profile document is rejected.

    The token replaces a real numeric field's literal so the strict parser sees
    it while parsing a document that is legal JSON in every other respect: the
    test proves the rejection is about the number, not a missing field.
    """
    payload = json.dumps({**WORKER, "role": "WORKER"})
    target = '"context_window_tokens": 32000'
    assert target in payload
    tampered = payload.replace(target, f'"context_window_tokens": {token}')
    with pytest.raises(ValidationError):
        Settings.from_env({"PRP_WORKER_PROFILE": tampered})
