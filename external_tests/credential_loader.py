"""Environment-only credentials for the optional external test harness.

No credential file format is supported. Live runs receive provider keys only
through ``PRP_EXTERNAL_*_API_KEY`` environment variables; example configuration
uses placeholders and never contains a real key.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Final
from urllib.parse import urlsplit

_PROFILE_PREFIX: Final = "PRP_EXTERNAL_"
_PLACEHOLDER_PREFIXES: Final = ("<", "YOUR_", "REPLACE_", "CHANGEME")

TERRA_ALIAS: Final = "TERRA_GPT"
TERRA_PROTOCOL: Final = "OPENAI_RESPONSES"
TERRA_ENV_NAMES: Final = {
    "model": "PRP_EXTERNAL_TERRA_GPT_MODEL",
    "base_url": "PRP_EXTERNAL_TERRA_GPT_BASE_URL",
    "api_key": "PRP_EXTERNAL_TERRA_GPT_API_KEY",
    "allowed_host": "PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST",
}
TERRA_NOT_CONFIGURED: Final = "TERRA_NOT_CONFIGURED"
TERRA_READY: Final = "READY"
TERRA_FALLBACK_NOT_ALLOWED: Final = "FALLBACK_NOT_ALLOWED"
TERRA_RETRYABLE_FAILURES: Final = frozenset(
    {"NETWORK", "TIMEOUT", "UPSTREAM_TRANSIENT", "PROVIDER_UNAVAILABLE"}
)

PROFILE_CONTRACTS: Final = {
    "DEEPSEEK_FLASH_CHAT": {
        "credential": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "DEEPSEEK_FLASH_RESPONSES": {
        "credential": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com",
    },
    "DEEPSEEK_FLASH_ANTHROPIC": {
        "credential": "deepseek",
        "model": "deepseek-v4-flash",
        "base_url": "https://api.deepseek.com/anthropic",
    },
    "LUNA_GPT_56": {
        "credential": "openai",
        "model": "gpt-5.6-luna",
        "base_url": "https://fast.vanyospace.com",
    },
    "CLAUDE_SONNET_5": {
        "credential": "anthropic",
        "model": "claude-sonnet-5",
        "base_url": "https://fast.vanyospace.com",
    },
}


class CredentialError(ValueError):
    """A non-sensitive environment credential configuration error."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(f"credential error code={code}")


class _SecretStr:
    __slots__ = ("__value",)

    def __init__(self, value: str) -> None:
        self.__value = value

    def __repr__(self) -> str:
        return "SecretStr('<redacted>')"

    def __str__(self) -> str:
        return "<redacted>"

    def reveal(self) -> str:
        return self.__value


class CredentialSet:
    """Opaque provider keys with controlled per-profile environment export."""

    __slots__ = ("__secrets",)

    def __init__(self, secrets: Mapping[str, _SecretStr]) -> None:
        self.__secrets = dict(secrets)

    def __repr__(self) -> str:
        return "CredentialSet(profiles=5, secrets=<redacted>)"

    def __str__(self) -> str:
        return "CredentialSet(<redacted>)"

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(PROFILE_CONTRACTS)

    def profile_env(self, alias: str) -> dict[str, str]:
        try:
            contract = PROFILE_CONTRACTS[alias]
            secret = self.__secrets[contract["credential"]].reveal()
        except KeyError as error:
            raise CredentialError("UNKNOWN_PROFILE") from error
        prefix = f"{_PROFILE_PREFIX}{alias}_"
        return {
            f"{prefix}API_KEY": secret,
            f"{prefix}MODEL": contract["model"],
            f"{prefix}BASE_URL": contract["base_url"],
        }

    def env_for(self, alias: str) -> dict[str, str]:
        return self.profile_env(alias)


def _is_placeholder(value: str) -> bool:
    upper = value.strip().upper()
    return not upper or any(upper.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


def load_credentials_from_env(environ: Mapping[str, str] | None = None) -> CredentialSet:
    """Load all active provider keys from environment variables only."""
    source = os.environ if environ is None else environ
    values: dict[str, _SecretStr] = {}
    missing: set[str] = set()
    for alias, contract in PROFILE_CONTRACTS.items():
        env_name = f"{_PROFILE_PREFIX}{alias}_API_KEY"
        raw = source.get(env_name, "").strip()
        provider = contract["credential"]
        if _is_placeholder(raw):
            missing.add(env_name)
            continue
        previous = values.get(provider)
        if previous is not None and previous.reveal() != raw:
            raise CredentialError(f"CONFLICTING_ENV:{provider}")
        values[provider] = _SecretStr(raw)
    if missing:
        raise CredentialError("MISSING_ENV:" + ",".join(sorted(missing)))
    return CredentialSet(values)


class OptionalProfileResolution:
    """Public, non-secret result of resolving optional Terra metadata."""

    __slots__ = ("alias", "status", "reason", "model", "base_url", "allowed_host", "api_key")

    def __init__(
        self,
        *,
        alias: str,
        status: str,
        reason: str,
        model: str | None = None,
        base_url: str | None = None,
        allowed_host: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.alias = alias
        self.status = status
        self.reason = reason
        self.model = model
        self.base_url = base_url
        self.allowed_host = allowed_host
        self.api_key = api_key

    def redacted(self) -> dict[str, str | None]:
        return {
            "alias": self.alias,
            "status": self.status,
            "reason": self.reason,
            "model": self.model,
            "base_url_host": _url_host(self.base_url) if self.base_url else None,
            "allowed_host": self.allowed_host,
            "api_key": "<redacted>" if self.api_key else None,
        }


def _url_host(value: str) -> str | None:
    try:
        return urlsplit(value).hostname
    except ValueError:
        return None


def resolve_terra_profile(
    environ: Mapping[str, str],
    *,
    allowed_hosts: set[str] | frozenset[str] | tuple[str, ...],
    fallback_from: str | None = None,
    failure_classification: str | None = None,
) -> OptionalProfileResolution:
    """Resolve Terra only from complete explicit environment metadata."""
    if fallback_from is not None and fallback_from != "LUNA_GPT_56":
        return OptionalProfileResolution(
            alias=TERRA_ALIAS,
            status=TERRA_FALLBACK_NOT_ALLOWED,
            reason="Terra fallback is only permitted after LUNA_GPT_56",
        )
    if fallback_from is not None:
        failure = failure_classification or environ.get(
            "PRP_EXTERNAL_LUNA_GPT_56_FAILURE", ""
        ).strip()
        if failure not in TERRA_RETRYABLE_FAILURES:
            return OptionalProfileResolution(
                alias=TERRA_ALIAS,
                status=TERRA_FALLBACK_NOT_ALLOWED,
                reason="Luna failure is not classified as retryable",
            )

    values = {field: environ.get(name, "").strip() for field, name in TERRA_ENV_NAMES.items()}
    missing = tuple(field for field, value in values.items() if not value)
    if missing:
        return OptionalProfileResolution(
            alias=TERRA_ALIAS,
            status=TERRA_NOT_CONFIGURED,
            reason=f"missing Terra metadata: {','.join(missing)}",
        )
    if fallback_from is None:
        return OptionalProfileResolution(
            alias=TERRA_ALIAS,
            status=TERRA_FALLBACK_NOT_ALLOWED,
            reason="Terra requires an explicit Luna fallback decision",
        )

    base_url = values["base_url"]
    allowed_host = values["allowed_host"]
    try:
        parsed = urlsplit(base_url)
        valid_url = (
            parsed.scheme == "https"
            and parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid_url = False
        parsed = None
    if not valid_url or parsed is None or parsed.hostname != allowed_host:
        return OptionalProfileResolution(
            alias=TERRA_ALIAS,
            status=TERRA_NOT_CONFIGURED,
            reason="Terra base URL must be HTTPS and match its explicit host",
        )
    if allowed_host.lower() not in {host.lower() for host in allowed_hosts}:
        return OptionalProfileResolution(
            alias=TERRA_ALIAS,
            status=TERRA_NOT_CONFIGURED,
            reason="Terra host is not explicitly admitted to the active allowlist",
            model=values["model"],
            base_url=base_url,
            allowed_host=allowed_host,
        )
    return OptionalProfileResolution(
        alias=TERRA_ALIAS,
        status=TERRA_READY,
        reason="complete Terra metadata is explicitly admitted",
        model=values["model"],
        base_url=base_url,
        allowed_host=allowed_host,
        api_key=values["api_key"],
    )


__all__ = [
    "CredentialError",
    "CredentialSet",
    "OptionalProfileResolution",
    "PROFILE_CONTRACTS",
    "TERRA_ALIAS",
    "TERRA_ENV_NAMES",
    "TERRA_FALLBACK_NOT_ALLOWED",
    "TERRA_NOT_CONFIGURED",
    "TERRA_PROTOCOL",
    "TERRA_READY",
    "TERRA_RETRYABLE_FAILURES",
    "load_credentials_from_env",
    "resolve_terra_profile",
]
