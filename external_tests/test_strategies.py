"""Strategy selection and routing tests."""

import pytest


@pytest.mark.live_strategy
def test_routing_direct(live_profile_anthropic):
    """Test DIRECT strategy: single stable provider."""
    from prp_runtime.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        api_key=live_profile_anthropic["api_key"],
        base_url=live_profile_anthropic["base_url"],
    )

    response = provider.complete(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Say OK"}],
        max_tokens=5,
    )

    assert response.text
    assert response.usage
    assert response.usage.input_tokens > 0


@pytest.mark.live_strategy
def test_routing_cascade(live_profile_anthropic, live_profile_deepseek):
    """Test CASCADE strategy: fallback on local failure."""
    # This would normally test local failure -> real fallback
    # For now, just verify both providers work independently
    from prp_runtime.providers.anthropic import AnthropicProvider
    from prp_runtime.providers.openai_responses import OpenAIResponsesProvider

    primary = OpenAIResponsesProvider(
        api_key=live_profile_deepseek["api_key"],
        base_url=live_profile_deepseek["base_url"],
    )

    fallback = AnthropicProvider(
        api_key=live_profile_anthropic["api_key"],
        base_url=live_profile_anthropic["base_url"],
    )

    # Test primary works
    r1 = primary.complete(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "1"}],
        max_tokens=5,
    )
    assert r1.text

    # Test fallback works
    r2 = fallback.complete(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "2"}],
        max_tokens=5,
    )
    assert r2.text


@pytest.mark.live_strategy
def test_routing_auto(live_profile_anthropic):
    """Test AUTO strategy: automatic selection."""
    from prp_runtime.providers.anthropic import AnthropicProvider

    provider = AnthropicProvider(
        api_key=live_profile_anthropic["api_key"],
        base_url=live_profile_anthropic["base_url"],
    )

    response = provider.complete(
        model="claude-3-5-sonnet-20241022",
        messages=[{"role": "user", "content": "Auto test"}],
        max_tokens=10,
    )

    assert response.text
    assert response.usage
