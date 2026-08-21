"""Agent workflow tests for end-to-end multi-step orchestration."""

import pytest


@pytest.mark.live_agent
@pytest.mark.asyncio
async def test_agent_single_turn_completion(live_profile_deepseek):
    """Test agent can complete a single-turn task."""
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider
    from prp_runtime.domain.enums import ModelRole

    profile = ModelProfile(
        alias="agent_deepseek",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=256,
    )

    provider = OpenAIResponsesProvider(profile)

    # Simple reasoning task
    request = ProviderRequest(
        alias="agent_test",
        model=profile.model,
        input="What is 15 + 27? Reply with just the number.",
        max_output_tokens=20,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    assert response.text
    assert "42" in response.text


@pytest.mark.live_agent
@pytest.mark.asyncio
async def test_agent_multi_protocol_workflow(
    live_profile_anthropic, live_profile_deepseek
):
    """Test agent workflow can coordinate across protocols."""
    from prp_runtime.providers.base import ModelProfile, ProviderProtocol, ProviderRequest
    from prp_runtime.providers.anthropic import AnthropicProvider
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider
    from prp_runtime.domain.enums import ModelRole

    # Setup profiles
    anthropic_profile = ModelProfile(
        alias="agent_anthropic",
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
        alias="agent_deepseek",
        provider="deepseek",
        model=live_profile_deepseek["model"],
        role=ModelRole.WORKER,
        protocol=ProviderProtocol.OPENAI_RESPONSES,
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
        context_window_tokens=64000,
        max_output_tokens=128,
    )

    # Step 1: Use Anthropic for initial task
    anthropic_provider = AnthropicProvider(anthropic_profile)
    step1_request = ProviderRequest(
        alias="step1",
        model=anthropic_profile.model,
        input="Name a color. Reply with just one word.",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )
    step1_response = await anthropic_provider.complete(step1_request)

    # Step 2: Use DeepSeek for follow-up
    deepseek_provider = OpenAIResponsesProvider(deepseek_profile)
    step2_request = ProviderRequest(
        alias="step2",
        model=deepseek_profile.model,
        input=f"The previous answer was: {step1_response.text}. Reply with OK.",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )
    step2_response = await deepseek_provider.complete(step2_request)

    # Both steps should succeed
    assert step1_response.text
    assert step2_response.text
    assert step1_response.usage.output_tokens > 0
    assert step2_response.usage.output_tokens > 0
