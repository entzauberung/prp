"""Protocol-level tests for message ingress and response egress."""

import pytest


@pytest.mark.live_protocols
@pytest.mark.asyncio
async def test_ingress_anthropic_messages(live_profile_anthropic):
    """Test Anthropic Messages protocol ingress."""
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.anthropic import AnthropicProvider
    from prp_runtime.domain.enums import ModelRole

    profile = ModelProfile(
        alias="test_anthropic",
        provider="anthropic",
        model=live_profile_anthropic["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.ANTHROPIC_MESSAGES,
        api_key=live_profile_anthropic["api_key"],
        base_url=live_profile_anthropic["base_url"],
        anthropic_version="2023-06-01",
        context_window_tokens=200000,
        max_output_tokens=8192,
    )

    provider = AnthropicProvider(profile)

    request = ProviderRequest(
        alias="test_anthropic",
        model=profile.model,
        input="Reply with exactly: TEST",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    assert response.text
    assert "TEST" in response.text.upper()


@pytest.mark.live_protocols
@pytest.mark.asyncio
async def test_ingress_openai_responses(live_profile_deepseek):
    """Test OpenAI Responses protocol ingress."""
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider
    from prp_runtime.domain.enums import ModelRole

    profile = ModelProfile(
        alias="test_deepseek",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=8000,
    )

    provider = OpenAIResponsesProvider(profile)

    request = ProviderRequest(
        alias="test_deepseek",
        model=profile.model,
        input="Reply with exactly: TEST",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    assert response.text
    assert "TEST" in response.text.upper()
