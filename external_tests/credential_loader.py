"""Strict, secret-safe credentials for the unattended external runner.

This module deliberately has no network, subprocess, environment, or logging
side effects. Only the runner should call ``load_credentials`` on the
authorized external file.
"""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Mapping
from typing import Final


MAX_CREDENTIAL_FILE_BYTES: Final = 64 * 1024
_PROFILE_PREFIX: Final = "PRP_EXTERNAL_"

_BLOCK_ALIASES: Final = {
    "DEEPSEEK": "deepseek",
    "DEEPSEEK_API": "deepseek",
    "DEEPSEEK_PROVIDER": "deepseek",
    "DEEPSEEK_V4": "deepseek",
    "DEEPSEEK_V4_FLASH": "deepseek",
    "DEEPSEEK_FLASH": "deepseek",
    "OPENAI": "openai",
    "VANYOSPACE_OPENAI": "openai",
    "VANYOSPACE_OPENAI_API": "openai",
    "LUNA": "openai",
    "LUNA_OPENAI": "openai",
    "LUNA_GPT_56": "openai",
    "LUNA_GPT_5_6": "openai",
    "GPT_56_LUNA": "openai",
    "GPT_5_6_LUNA": "openai",
    "GPT_56": "openai",
    "VANYOSPACE_LUNA": "openai",
    "VANYOSPACE_GPT": "openai",
    "ANTHROPIC": "anthropic",
    "VANYOSPACE_ANTHROPIC": "anthropic",
    "VANYOSPACE_ANTHROPIC_API": "anthropic",
    "CLAUDE": "anthropic",
    "CLAUDE_ANTHROPIC": "anthropic",
    "CLAUDE_SONNET_5": "anthropic",
    "CLAUDE_SONNET": "anthropic",
    "VANYOSPACE_CLAUDE": "anthropic",
    "VANYOSPACE_SONNET": "anthropic",
    "VANYOSPACE": "vanyospace",
    "ZHIPU": "zhipu",
    "ZHIPU_API": "zhipu",
    "ZHIPUAI": "zhipu",
    "GLM": "zhipu",
    "BIGMODEL": "zhipu",
}
_DOCUMENTATION_HEADINGS: Final = frozenset(
    {
        "API_KEYS",
        "AUTHORIZED_CREDENTIALS",
        "CREDENTIALS",
        "EXTERNAL_CREDENTIALS",
        "TEST_KEYS",
    }
)

_KEY_ALIASES: Final = {
    "deepseek": {
        "API_KEY": "deepseek",
        "KEY": "deepseek",
        "DEEPSEEK_KEY": "deepseek",
        "DEEPSEEK_API_KEY": "deepseek",
    },
    "openai": {
        "API_KEY": "openai",
        "OPENAI_API_KEY": "openai",
        "OPENAI_KEY": "openai",
        "LUNA_API_KEY": "openai",
        "LUNA_KEY": "openai",
        "GPT_5_6_LUNA_API_KEY": "openai",
        "VANYOSPACE_OPENAI_API_KEY": "openai",
    },
    "anthropic": {
        "API_KEY": "anthropic",
        "ANTHROPIC_API_KEY": "anthropic",
        "ANTHROPIC_KEY": "anthropic",
        "CLAUDE_API_KEY": "anthropic",
        "CLAUDE_KEY": "anthropic",
        "VANYOSPACE_ANTHROPIC_API_KEY": "anthropic",
    },
    "vanyospace": {
        "LUNA_API_KEY": "openai",
        "LUNA_KEY": "openai",
        "OPENAI_API_KEY": "openai",
        "VANYOSPACE_OPENAI_API_KEY": "openai",
        "CLAUDE_API_KEY": "anthropic",
        "CLAUDE_KEY": "anthropic",
        "ANTHROPIC_API_KEY": "anthropic",
        "VANYOSPACE_ANTHROPIC_API_KEY": "anthropic",
    },
    "zhipu": {
        "API_KEY": "zhipu",
        "ZHIPU_API_KEY": "zhipu",
        "ZHIPU_KEY": "zhipu",
        "ZHIPUAI_API_KEY": "zhipu",
        "GLM_API_KEY": "zhipu",
    },
}

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

_HEADER_RE = re.compile(r"^\s*(?:#{1,6}\s+|\[)(?P<name>[^\]#]+?)(?:\]|\s*#*)\s*$")
_ASSIGNMENT_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:`)?(?P<key>[^:=`]+?)(?:`)?\s*(?:=|:)\s*(?P<value>.*?)\s*$"
)
_BLOCK_MARKER_RE = re.compile(
    r"^\s*(?:[-*]\s*)?(?:\*\*|__)?(?P<label>[A-Za-z0-9_. -]+?)(?:\*\*|__)?\s*:\s*$"
)


class CredentialError(ValueError):
    """An intentionally non-sensitive credential parsing error."""

    def __init__(self, code: str, block: str | None = None) -> None:
        self.code = code
        self.block = block
        label = f" block={block}" if block else ""
        super().__init__(f"credential error code={code}{label}")


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


def _normalise_label(label: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", label.strip().upper()).strip("_")


def _normalise_value(value: str, block: str) -> str:
    candidate = value.strip()
    if len(candidate) >= 2 and candidate[0] == candidate[-1] and candidate[0] in "'\"`":
        candidate = candidate[1:-1].strip()
    if not candidate:
        raise CredentialError("INVALID_VALUE", block)
    # Allow dots for zhipu keys, but still reject other whitespace
    if any(c.isspace() for c in candidate):
        raise CredentialError("INVALID_VALUE", block)
    return candidate


class CredentialSet:
    """Opaque parsed credentials with a controlled per-profile env export."""

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


def parse_credentials_text(text: str) -> CredentialSet:
    """Parse provider credentials from loosely-formatted Markdown text.

    Relaxed mode: extracts API keys using context hints (deepseek, zhipu, claude keywords).
    Supports both sk-* format and zhipu's {hex}.{base64} format.
    """

    if not isinstance(text, str):
        raise CredentialError("INVALID_TEXT")

    # Collect all API key candidates with their surrounding context
    candidates: list[tuple[str, str]] = []  # (key, context_line)

    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Match sk-* keys OR zhipu format (hex.base64)
        is_sk_key = stripped.startswith("sk-") and len(stripped) > 20
        is_zhipu_key = (
            "." in stripped
            and len(stripped) > 30
            and not stripped.startswith("http")
            and not " " in stripped
            and re.match(r"^[a-f0-9]{32}\.[A-Za-z0-9]+$", stripped)
        )

        if is_sk_key or is_zhipu_key:
            # Gather context: look back up to 5 lines for provider hints
            context_lines = []
            for j in range(max(0, i - 5), i + 1):
                context_lines.append(lines[j].lower())
            context = " ".join(context_lines)
            candidates.append((stripped, context))

    # Classify candidates by provider hints
    values: dict[str, _SecretStr] = {}

    # Provider-specific hints ordered by specificity (most specific first)
    provider_patterns = {
        "deepseek": ["deepseek", "api.deepseek.com"],
        "zhipu": ["zhipu", "bigmodel", "glm", "可以使用的模型有glm"],
        "anthropic": ["claude", "anthropic", "sonnet"],
        "openai": ["gpt-5.6-luna", "luna", "fast.vanyospace.com", "第三方的测试key"],
    }

    for key, context in candidates:
        # First check if the key format suggests zhipu (hex.base64)
        matched_provider = None
        if "." in key and re.match(r"^[a-f0-9]{32}\.[A-Za-z0-9]+$", key):
            matched_provider = "zhipu"

        # Then check if the key itself contains a provider hint
        if not matched_provider:
            key_lower = key.lower()
            for provider in ["deepseek", "anthropic", "openai"]:
                if provider in key_lower:
                    matched_provider = provider
                    break

        # If no match in key, check context with specificity ordering
        if not matched_provider:
            for provider, hints in provider_patterns.items():
                if any(hint in context for hint in hints):
                    matched_provider = provider
                    break

        if matched_provider and matched_provider not in values:
            try:
                values[matched_provider] = _SecretStr(_normalise_value(key, matched_provider))
            except CredentialError:
                pass

    # Only the three credential classes used by the active five-profile matrix
    # are required. Extra legacy keys in the authorized document are ignored.
    required_keys = {"deepseek", "openai", "anthropic"}
    missing = required_keys - set(values)
    if missing:
        raise CredentialError(f"MISSING_KEY: {missing}")

    return CredentialSet(values)


def load_credentials(path: str | os.PathLike[str]) -> CredentialSet:
    """Load a regular, non-symlink UTF-8 credential file without logging it."""

    try:
        file_path = os.fspath(path)
    except TypeError as error:
        raise CredentialError("INVALID_PATH") from error

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(file_path, flags)
    except OSError as error:
        raise CredentialError("FILE_UNREADABLE") from error

    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise CredentialError("NOT_REGULAR_FILE")
        if file_stat.st_size > MAX_CREDENTIAL_FILE_BYTES:
            raise CredentialError("FILE_TOO_LARGE")
        with os.fdopen(descriptor, "rb", closefd=True) as handle:
            descriptor = -1
            data = handle.read(MAX_CREDENTIAL_FILE_BYTES + 1)
    except CredentialError:
        raise
    except OSError as error:
        raise CredentialError("FILE_UNREADABLE") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if len(data) > MAX_CREDENTIAL_FILE_BYTES:
        raise CredentialError("FILE_TOO_LARGE")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise CredentialError("INVALID_ENCODING") from error
    return parse_credentials_text(text)


load_authorized_credentials = load_credentials


__all__ = [
    "CredentialError",
    "CredentialSet",
    "MAX_CREDENTIAL_FILE_BYTES",
    "PROFILE_CONTRACTS",
    "load_authorized_credentials",
    "load_credentials",
    "parse_credentials_text",
]
