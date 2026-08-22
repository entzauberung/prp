"""Regression tests to verify baseline functionality remains intact."""

import pytest


@pytest.mark.live_regression
@pytest.mark.asyncio
async def test_regression_basic_completion(live_profile_deepseek):
    """Regression: basic completion flow must work."""
    from prp_runtime.domain.enums import ModelRole
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider

    profile = ModelProfile(
        alias="regression_deepseek",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=128,
    )

    provider = OpenAIResponsesProvider(profile)

    request = ProviderRequest(
        alias="regression",
        model=profile.model,
        input="Say hello.",
        max_output_tokens=20,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    # Basic assertions
    assert response.text
    assert len(response.text) > 0
    assert response.usage
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0


@pytest.mark.live_regression
@pytest.mark.asyncio
async def test_regression_error_handling(live_profile_deepseek):
    """Regression: error handling must be robust."""
    from prp_runtime.domain.enums import ModelRole
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider

    profile = ModelProfile(
        alias="regression_error",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=1,  # Extremely low limit
    )

    provider = OpenAIResponsesProvider(profile)

    request = ProviderRequest(
        alias="regression",
        model=profile.model,
        input="Write a long story.",
        max_output_tokens=1,
        timeout_seconds=30.0,
    )

    # Should complete without crashing even with tiny token limit
    response = await provider.complete(request)
    assert response is not None


@pytest.mark.live_regression
@pytest.mark.asyncio
async def test_regression_protocol_compatibility(live_profile_anthropic):
    """Regression: Anthropic Messages protocol must remain compatible."""
    from prp_runtime.domain.enums import ModelRole
    from prp_runtime.providers.anthropic import AnthropicProvider
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest

    profile = ModelProfile(
        alias="regression_anthropic",
        provider="anthropic",
        model=live_profile_anthropic["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.ANTHROPIC_MESSAGES,
        api_key=live_profile_anthropic["api_key"],
        base_url=live_profile_anthropic["base_url"],
        anthropic_version="2023-06-01",
        context_window_tokens=200000,
        max_output_tokens=128,
    )

    provider = AnthropicProvider(profile)

    request = ProviderRequest(
        alias="regression",
        model=profile.model,
        input="Reply: OK",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    assert response.text
    assert response.usage
    assert response.usage.input_tokens > 0
    assert response.usage.output_tokens > 0
