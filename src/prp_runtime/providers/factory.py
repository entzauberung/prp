"""Closed provider-protocol to adapter construction registry.

Adapters are selected by protocol, not by role. Analyzer and Verifier profiles
never silently construct a Worker adapter.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path

from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.providers.anthropic import AnthropicMessagesProvider
from prp_runtime.providers.base import ModelProfile, ProviderAdapter, ProviderProtocol
from prp_runtime.providers.openai_compatible import OpenAICompatibleProvider
from prp_runtime.providers.openai_responses import OpenAIResponsesProvider

__all__ = [
    "AdapterConstructor",
    "DEFAULT_ADAPTER_REGISTRY",
    "build_provider_adapter",
    "create_provider",
]

AdapterConstructor = Callable[[ModelProfile], ProviderAdapter]

DEFAULT_ADAPTER_REGISTRY: Mapping[ProviderProtocol, AdapterConstructor] = {
    ProviderProtocol.OPENAI_CHAT: OpenAICompatibleProvider,
    ProviderProtocol.OPENAI_RESPONSES: OpenAIResponsesProvider,
    ProviderProtocol.ANTHROPIC_MESSAGES: AnthropicMessagesProvider,
}


def build_provider_adapter(
    profile: ModelProfile,
    *,
    registry: Mapping[ProviderProtocol, AdapterConstructor] | None = None,
) -> ProviderAdapter:
    """Construct the registered adapter for one server-owned model profile."""

    registered = DEFAULT_ADAPTER_REGISTRY if registry is None else registry
    constructor: AdapterConstructor | None
    match profile.protocol:
        case ProviderProtocol.OPENAI_CHAT:
            constructor = registered.get(ProviderProtocol.OPENAI_CHAT)
        case ProviderProtocol.OPENAI_RESPONSES:
            constructor = registered.get(ProviderProtocol.OPENAI_RESPONSES)
        case ProviderProtocol.ANTHROPIC_MESSAGES:
            constructor = registered.get(ProviderProtocol.ANTHROPIC_MESSAGES)
        case _:
            raise ProviderError(
                "unsupported outbound provider protocol",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
    if constructor is None:
        raise ProviderError(
            f"no adapter is registered for protocol {profile.protocol.value}",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
        )
    return constructor(profile)


def create_provider(
    profile_name: str,
    credential_file: str | Path,
    *,
    registry: Mapping[ProviderProtocol, AdapterConstructor] | None = None,
) -> ProviderAdapter:
    """Create a provider adapter by loading profile from credential file.

    Args:
        profile_name: Name of the profile to load
        credential_file: Path to credential file
        registry: Optional custom adapter registry

    Returns:
        Configured provider adapter

    Raises:
        ProviderError: If profile not found or protocol not supported
    """
    from prp_runtime.settings import load_credential_profiles

    profiles = load_credential_profiles(Path(credential_file))

    if profile_name not in profiles:
        raise ProviderError(
            f"profile '{profile_name}' not found in credential file",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
        )

    profile = profiles[profile_name]
    return build_provider_adapter(profile, registry=registry)
