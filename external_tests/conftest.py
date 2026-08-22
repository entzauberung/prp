"""Pytest configuration for external live tests."""

import os
from dataclasses import dataclass
from pathlib import Path

import pytest

PREFLIGHT_EXPECTED_ALIASES = (
    "DEEPSEEK_FLASH_CHAT",
    "DEEPSEEK_FLASH_RESPONSES",
    "DEEPSEEK_FLASH_ANTHROPIC",
    "LUNA_GPT_56",
    "CLAUDE_SONNET_5",
)


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "live_provider: marks tests as live provider tests")
    config.addinivalue_line("markers", "live_integration: marks tests as live integration tests")
    config.addinivalue_line("markers", "live_protocols: marks tests as live protocol tests")
    config.addinivalue_line("markers", "live_strategy: marks tests as live strategy tests")
    config.addinivalue_line("markers", "live_agent: marks tests as live agent tests")
    config.addinivalue_line("markers", "live_regression: marks tests as live regression tests")


def pytest_collection_modifyitems(config, items):
    """Fail closed if preflight does not collect exactly the active smoke set."""
    if os.environ.get("PRP_LIVE_PREFLIGHT") != "1":
        return

    expected = int(os.environ.get("PRP_LIVE_EXPECTED_SCENARIOS", "0"))
    nodeids = " ".join(item.nodeid for item in items)
    missing = [alias for alias in PREFLIGHT_EXPECTED_ALIASES if alias not in nodeids]
    if expected != len(PREFLIGHT_EXPECTED_ALIASES) or len(items) != expected or missing:
        raise pytest.UsageError(
            "preflight scenario guard failed: "
            f"expected={expected} collected={len(items)} missing={','.join(missing) or 'none'}"
        )
    if any(item.get_closest_marker("live_provider") is None for item in items):
        raise pytest.UsageError("preflight scenario guard found a non-live-provider item")


@pytest.fixture
def external_config():
    """Load external config with proper validation."""
    from external_tests.support import load_external_config

    try:
        return load_external_config()
    except Exception as e:
        pytest.skip(f"External config not available: {e}")


@pytest.fixture
def live_profile_anthropic(external_config):
    """Provide Anthropic profile configuration as dict."""
    # Find Luna or Claude profile
    for profile in external_config.profiles:
        if profile.alias in ("LUNA_GPT_56", "CLAUDE_SONNET_5"):
            return {
                "alias": profile.alias,
                "model": profile.model_id,
                "api_key": profile.api_key,
                "base_url": profile.base_url,
                "protocol": profile.protocol,
            }

    pytest.skip("No Anthropic profile (LUNA or CLAUDE) available")


@pytest.fixture
def live_profile_deepseek(external_config):
    """Provide DeepSeek profile configuration as dict."""
    for profile in external_config.profiles:
        if "DEEPSEEK" in profile.alias:
            return {
                "alias": profile.alias,
                "model": profile.model_id,
                "api_key": profile.api_key,
                "base_url": profile.base_url,
                "protocol": profile.protocol,
            }

    pytest.skip("No DeepSeek profile available")


@pytest.fixture
def live_profile_zhipu(external_config):
    """Provide Zhipu profile configuration as dict."""
    for profile in external_config.profiles:
        if "ZHIPU" in profile.alias:
            return {
                "alias": profile.alias,
                "model": profile.model_id,
                "api_key": profile.api_key,
                "base_url": profile.base_url,
                "protocol": profile.protocol,
            }

    pytest.skip("No Zhipu profile available")


@dataclass
class TemporaryResources:
    """Temporary resources for external tests."""
    database_path: Path


@pytest.fixture
def temporary_resources(tmp_path):
    """Provide temporary resources for external tests."""
    db_path = tmp_path / "prp_test.db"
    return TemporaryResources(database_path=db_path)
