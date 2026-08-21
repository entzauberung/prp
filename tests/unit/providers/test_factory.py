"""Targeted tests for closed protocol adapter construction."""

import pytest

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.providers.base import ModelProfile, ProviderProtocol
from prp_runtime.providers.anthropic import AnthropicMessagesProvider
from prp_runtime.providers.factory import build_provider_adapter
from prp_runtime.providers.openai_responses import OpenAIResponsesProvider


def _profile(protocol: ProviderProtocol) -> ModelProfile:
    version = "2023-06-01" if protocol is ProviderProtocol.ANTHROPIC_MESSAGES else None
    return ModelProfile(
        alias="worker",
        provider="test-provider",
        model="test-model",
        role=ModelRole.WORKER,
        base_url="https://models.example/v1",
        protocol=protocol,
        anthropic_version=version,
        context_window_tokens=1_024,
        max_output_tokens=128,
    )


class StubAdapter:
    def __init__(self, profile: ModelProfile) -> None:
        self.profile = profile

    @property
    def name(self) -> str:
        return "stub"

    async def aclose(self) -> None:
        return None

    async def complete(self, request: object) -> object:
        del request
        raise AssertionError("factory unit test adapter must not be called")


@pytest.mark.parametrize("protocol", list(ProviderProtocol))
def test_factory_dispatches_each_closed_protocol_with_registered_constructor(
    protocol: ProviderProtocol,
) -> None:
    constructed: list[ProviderProtocol] = []

    def constructor(profile: ModelProfile) -> StubAdapter:
        constructed.append(profile.protocol)
        return StubAdapter(profile)

    adapter = build_provider_adapter(
        _profile(protocol),
        registry={candidate: constructor for candidate in ProviderProtocol},
    )
    assert isinstance(adapter, StubAdapter)
    assert constructed == [protocol]


def test_factory_keeps_chat_default_and_rejects_unregistered_protocol() -> None:
    chat = build_provider_adapter(_profile(ProviderProtocol.OPENAI_CHAT))
    assert chat.name == "test-provider"
    responses = build_provider_adapter(_profile(ProviderProtocol.OPENAI_RESPONSES))
    assert isinstance(responses, OpenAIResponsesProvider)
    anthropic = build_provider_adapter(_profile(ProviderProtocol.ANTHROPIC_MESSAGES))
    assert isinstance(anthropic, AnthropicMessagesProvider)


def test_factory_rejects_an_unregistered_protocol() -> None:
    with pytest.raises(ProviderError) as excinfo:
        build_provider_adapter(
            _profile(ProviderProtocol.OPENAI_RESPONSES),
            registry={ProviderProtocol.OPENAI_CHAT: lambda profile: StubAdapter(profile)},
        )
    assert excinfo.value.code is ErrorCode.PROVIDER_NOT_CONFIGURED
