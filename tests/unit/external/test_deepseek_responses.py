"""Smoke test for DeepSeek via OpenAI Responses protocol."""

import os
import pytest
from prp_runtime.providers.factory import create_provider
from prp_runtime.providers.base import ProviderRequest

pytestmark = pytest.mark.skipif(
    not os.environ.get("PRP_TEST_CREDENTIAL_FILE"),
    reason="external smoke requires explicit credential opt-in",
)


@pytest.mark.external
@pytest.mark.live_provider
@pytest.mark.asyncio
async def test_deepseek_responses_smoke():
    """Smoke test: DeepSeek deepseek-v4-flash via OpenAI Responses protocol."""
    credential_file = os.environ.get("PRP_TEST_CREDENTIAL_FILE")
    assert credential_file, "PRP_TEST_CREDENTIAL_FILE not set"

    provider = create_provider(
        profile_name="DEEPSEEK_FLASH_RESPONSES",
        credential_file=credential_file,
    )

    request = ProviderRequest(
        alias=provider.profile.alias,
        model=provider.profile.model,
        input="Say 'OK' if you can read this.",
        max_output_tokens=10,
        timeout_seconds=30.0,
    )

    response = await provider.complete(request)

    assert response.output_text, "Response output_text is empty"
    assert "OK" in response.output_text.upper(), "Response does not contain 'OK'"
