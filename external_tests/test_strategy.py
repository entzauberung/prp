"""Strategy inference tests for multi-protocol routing and decision logic."""

import pytest


@pytest.mark.live_strategy
@pytest.mark.asyncio
async def test_multi_protocol_fallback_chain(
    live_profile_anthropic, live_profile_deepseek
):
    """Test strategy layer can route across multiple protocols."""
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.anthropic import AnthropicProvider
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider
    from prp_runtime.domain.enums import ModelRole

    # Create two profiles with different protocols
    anthropic_profile = ModelProfile(
        alias="strategy_anthropic",
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

    deepseek_profile = ModelProfile(
        alias="strategy_deepseek",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=128,
    )

    # Test both providers can handle the same input
    request = ProviderRequest(
        alias="test",
        model="test",
        input="Count to three.",
        max_output_tokens=50,
        timeout_seconds=30.0,
    )

    anthropic_provider = AnthropicProvider(anthropic_profile)
    deepseek_provider = OpenAIResponsesProvider(deepseek_profile)

    anthropic_response = await anthropic_provider.complete(request)
    deepseek_response = await deepseek_provider.complete(request)

    # Both should succeed
    assert anthropic_response.text
    assert deepseek_response.text
    assert anthropic_response.usage.output_tokens > 0
    assert deepseek_response.usage.output_tokens > 0


@pytest.mark.live_strategy
@pytest.mark.asyncio
async def test_protocol_selection_via_factory(external_config):
    """Test factory can instantiate correct provider based on protocol."""
    from prp_runtime.providers.factory import create_provider
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol
    from prp_runtime.domain.enums import ModelRole

    # Find one profile of each protocol type
    anthropic_prof = None
    openai_prof = None

    for profile in external_config.profiles:
        if profile.protocol == "ANTHROPIC_MESSAGES" and not anthropic_prof:
            anthropic_prof = profile
        elif profile.protocol == "OPENAI_RESPONSES" and not openai_prof:
            openai_prof = profile

    if not anthropic_prof and not openai_prof:
        pytest.skip("Need at least one profile to test factory")

    # Test Anthropic factory
    if anthropic_prof:
        profile = ModelProfile(
            alias="factory_anthropic",
            provider="anthropic",
            model=anthropic_prof.model_id,
            role=ModelRole.WORKER,
            protocol=ProviderProtocol.ANTHROPIC_MESSAGES,
            api_key=anthropic_prof.api_key,
            base_url=anthropic_prof.base_url,
            anthropic_version="2023-06-01",
            context_window_tokens=200000,
            max_output_tokens=128,
        )
        provider = create_provider(profile)
        assert provider is not None
        assert hasattr(provider, "complete")

    # Test OpenAI factory
    if openai_prof:
        profile = ModelProfile(
            alias="factory_openai",
            provider="openai",
            model=openai_prof.model_id,
            role=ModelRole.WORKER,
            protocol=ProviderProtocol.OPENAI_RESPONSES,
            api_key=openai_prof.api_key,
            base_url=openai_prof.base_url,
            context_window_tokens=64000,
            max_output_tokens=128,
        )
        provider = create_provider(profile)
        assert provider is not None
        assert hasattr(provider, "complete")
