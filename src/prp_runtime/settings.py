"""Runtime settings.

Settings are plain data validated by Pydantic. Reading settings has no side
effect: no network access, no database connection, no logging reconfiguration.

Model endpoints and credentials are part of this server-side configuration.
A request never carries a provider URL or an API key.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, SecretStr, model_validator

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.values import PrincipalId
from prp_runtime.json_support import strict_json_loads
from prp_runtime.providers.base import ModelProfile
from prp_runtime.workspace.isolation import (
    DEFAULT_ISOLATION_MAX_BYTES,
    DEFAULT_ISOLATION_MAX_SLOTS,
    MAX_ISOLATION_MAX_BYTES,
    MAX_ISOLATION_MAX_SLOTS,
)
from prp_runtime.workspace.models import WorkspaceRootMapping

__all__ = [
    "DEFAULT_PROCESS_MAX_ATTEMPTS",
    "DEFAULT_PROCESS_MAX_CONCURRENCY",
    "DEFAULT_PROCESS_MAX_TOTAL_TOKENS",
    "ENV_PREFIX",
    "LogLevel",
    "MAX_PROCESS_MAX_ATTEMPTS",
    "MAX_PROCESS_MAX_CONCURRENCY",
    "MAX_PROCESS_MAX_TOTAL_TOKENS",
    "ProcessResourceEnvelope",
    "Settings",
    "load_credential_profiles",
]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

ENV_PREFIX = "PRP_"

_ENV_FIELDS: dict[str, str] = {
    "PRP_DATABASE_PATH": "database_path",
    "PRP_MAX_REQUEST_BYTES": "max_request_bytes",
    "PRP_MAX_INPUT_CHARS": "max_input_chars",
    "PRP_LOG_LEVEL": "log_level",
    "PRP_LEADER_PROFILE": "leader_profile",
    "PRP_WORKER_PROFILE": "worker_profile",
    "PRP_ANALYZER_PROFILE": "analyzer_profile",
    "PRP_VERIFIER_PROFILE": "verifier_profile",
    "PRP_CASCADE_PROFILES": "cascade_profiles",
    "PRP_ALLOW_HOST_YOLO": "allow_host_yolo",
    "PRP_SERVICE_TOKEN": "service_token",
    "PRP_SERVICE_PRINCIPAL": "service_principal",
    "PRP_WORKSPACE_ROOTS": "workspace_roots",
    "PRP_ISOLATION_MAX_SLOTS": "isolation_max_slots",
    "PRP_ISOLATION_MAX_BYTES": "isolation_max_bytes",
    "PRP_PROCESS_MAX_CONCURRENCY": "process_max_concurrency",
    "PRP_PROCESS_MAX_ATTEMPTS": "process_max_attempts",
    "PRP_PROCESS_MAX_TOTAL_TOKENS": "process_max_total_tokens",
    "PRP_EXPORT_MAX_FILES": "export_max_files",
    "PRP_EXPORT_MAX_BYTES": "export_max_bytes",
    "PRP_EXPORT_MAX_NESTING": "export_max_nesting",
    "PRP_EXPORT_MAX_SECONDS": "export_max_seconds",
}

DEFAULT_PROCESS_MAX_CONCURRENCY = 1
MAX_PROCESS_MAX_CONCURRENCY = 8
DEFAULT_PROCESS_MAX_ATTEMPTS = 8
MAX_PROCESS_MAX_ATTEMPTS = 32
DEFAULT_PROCESS_MAX_TOTAL_TOKENS = 250_000
MAX_PROCESS_MAX_TOTAL_TOKENS = 1_000_000
DEFAULT_LOCAL_WAIT_SECONDS = 30.0
MAX_LOCAL_WAIT_SECONDS = 300.0
DEFAULT_EXPORT_MAX_FILES = 32
MAX_EXPORT_MAX_FILES = 128
DEFAULT_EXPORT_MAX_BYTES = 32_768
MAX_EXPORT_MAX_BYTES = 96_000
DEFAULT_EXPORT_MAX_NESTING = 6
MAX_EXPORT_MAX_NESTING = 12
DEFAULT_EXPORT_MAX_SECONDS = 5.0
MAX_EXPORT_MAX_SECONDS = 15.0


def _profile_from_json(value: object) -> object:
    """Accept a JSON document for a model profile.

    Parsed strictly like every other JSON the runtime reads: an environment
    variable is external input, and a non-finite number in a profile would travel
    into a provider payload that cannot be written back out as JSON.
    """
    if isinstance(value, str):
        return strict_json_loads(value)
    return value


def _cascade_profiles_from_json(value: object) -> object:
    """Accept a JSON array of profile objects for PRP_CASCADE_PROFILES."""
    if isinstance(value, str):
        parsed = strict_json_loads(value)
        if not isinstance(parsed, list):
            raise ValueError("PRP_CASCADE_PROFILES must be a JSON array")
        return parsed
    return value


def _workspace_roots_from_json(value: object) -> object:
    """Accept a strict JSON object for server-only workspace roots."""
    if isinstance(value, str):
        parsed = strict_json_loads(value)
        if not isinstance(parsed, dict) or not all(
            isinstance(alias, str) for alias in parsed
        ):
            raise ValueError("PRP_WORKSPACE_ROOTS must be a JSON object")
        return parsed
    return value


ProfileValue = Annotated[ModelProfile, BeforeValidator(_profile_from_json)]
CascadeProfilesValue = Annotated[
    tuple[ModelProfile, ...],
    BeforeValidator(_cascade_profiles_from_json),
]
WorkspaceRootsValue = Annotated[
    WorkspaceRootMapping,
    BeforeValidator(_workspace_roots_from_json),
]


@dataclass(frozen=True, slots=True)
class ProcessResourceEnvelope:
    """Process-local slot, byte, concurrency, attempt and token ceilings."""

    max_slots: int
    max_copied_bytes: int
    max_concurrency: int
    max_attempts: int
    max_total_tokens: int

    def public_facts(self) -> dict[str, int]:
        """Return numeric ceilings without host paths or credentials."""
        return {
            "max_slots": self.max_slots,
            "max_copied_bytes": self.max_copied_bytes,
            "max_concurrency": self.max_concurrency,
            "max_attempts": self.max_attempts,
            "max_total_tokens": self.max_total_tokens,
        }


class Settings(BaseModel):
    """Immutable runtime settings.

    Unknown fields are rejected instead of ignored, and every limit must be a
    positive integer.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    database_path: Path = Path("prp_runtime.db")
    max_request_bytes: int = Field(default=1_048_576, gt=0)
    max_input_chars: int = Field(default=100_000, gt=0)
    log_level: LogLevel = "INFO"
    allow_host_yolo: bool = False
    service_token: SecretStr | None = None
    service_principal: PrincipalId = "prn_default"
    workspace_roots: WorkspaceRootsValue = Field(
        default_factory=WorkspaceRootMapping,
        exclude=True,
        repr=False,
    )
    leader_profile: ProfileValue | None = None
    worker_profile: ProfileValue | None = None
    analyzer_profile: ProfileValue | None = None
    verifier_profile: ProfileValue | None = None
    cascade_profiles: CascadeProfilesValue = Field(default_factory=tuple)
    isolation_max_slots: int = Field(
        default=DEFAULT_ISOLATION_MAX_SLOTS,
        ge=1,
        le=MAX_ISOLATION_MAX_SLOTS,
    )
    isolation_max_bytes: int = Field(
        default=DEFAULT_ISOLATION_MAX_BYTES,
        ge=1,
        le=MAX_ISOLATION_MAX_BYTES,
    )
    process_max_concurrency: int = Field(
        default=DEFAULT_PROCESS_MAX_CONCURRENCY,
        ge=1,
        le=MAX_PROCESS_MAX_CONCURRENCY,
    )
    process_max_attempts: int = Field(
        default=DEFAULT_PROCESS_MAX_ATTEMPTS,
        ge=1,
        le=MAX_PROCESS_MAX_ATTEMPTS,
    )
    process_max_total_tokens: int = Field(
        default=DEFAULT_PROCESS_MAX_TOTAL_TOKENS,
        ge=1,
        le=MAX_PROCESS_MAX_TOTAL_TOKENS,
    )
    local_wait_seconds: float = Field(
        default=DEFAULT_LOCAL_WAIT_SECONDS,
        gt=0,
        le=MAX_LOCAL_WAIT_SECONDS,
    )
    export_max_files: int = Field(
        default=DEFAULT_EXPORT_MAX_FILES,
        ge=1,
        le=MAX_EXPORT_MAX_FILES,
    )
    export_max_bytes: int = Field(
        default=DEFAULT_EXPORT_MAX_BYTES,
        ge=1,
        le=MAX_EXPORT_MAX_BYTES,
    )
    export_max_nesting: int = Field(
        default=DEFAULT_EXPORT_MAX_NESTING,
        ge=1,
        le=MAX_EXPORT_MAX_NESTING,
    )
    export_max_seconds: float = Field(
        default=DEFAULT_EXPORT_MAX_SECONDS,
        gt=0,
        le=MAX_EXPORT_MAX_SECONDS,
    )

    @model_validator(mode="after")
    def _profiles_are_consistent(self) -> "Settings":
        if self.service_token is not None and not self.service_token.get_secret_value().strip():
            raise ValueError("service_token must not be blank")
        if self.leader_profile is not None and self.leader_profile.role is not ModelRole.PLANNER:
            raise ValueError("the leader profile must declare the PLANNER role")
        if self.worker_profile is not None and self.worker_profile.role is not ModelRole.WORKER:
            raise ValueError("the worker profile must declare the WORKER role")
        if self.analyzer_profile is not None and self.analyzer_profile.role is not ModelRole.ANALYZER:
            raise ValueError("the analyzer profile must declare the ANALYZER role")
        if self.verifier_profile is not None and self.verifier_profile.role is not ModelRole.VERIFIER:
            raise ValueError("the verifier profile must declare the VERIFIER role")
        if any(profile.role is not ModelRole.WORKER for profile in self.cascade_profiles):
            raise ValueError("cascade profiles must declare the WORKER role")

        aliases = [profile.alias for profile in self.profiles]
        if len(aliases) != len(set(aliases)):
            raise ValueError("all configured profiles must use different aliases")
        return self

    @property
    def resource_envelope(self) -> ProcessResourceEnvelope:
        """Return the process-local resource envelope owned by settings."""
        return ProcessResourceEnvelope(
            max_slots=self.isolation_max_slots,
            max_copied_bytes=self.isolation_max_bytes,
            max_concurrency=self.process_max_concurrency,
            max_attempts=self.process_max_attempts,
            max_total_tokens=self.process_max_total_tokens,
        )

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        """Every configured model profile."""
        primary_profiles = tuple(
            profile
            for profile in (
                self.leader_profile,
                self.worker_profile,
                self.analyzer_profile,
                self.verifier_profile,
            )
            if profile is not None
        )
        return primary_profiles + self.cascade_profiles

    def profile_by_alias(self, alias: str) -> ModelProfile:
        """Look up a configured profile, or fail with a structured error."""
        for profile in self.profiles:
            if profile.alias == alias:
                return profile
        raise ProviderError(
            f"model alias {alias!r} is not configured on this server",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
        )

    def profile_for_role(self, role: ModelRole) -> ModelProfile | None:
        """Return the profile configured for a role, or None when unused.

        Analyzer and Verifier may be omitted: deterministic implementations do
        not require a provider. Missing profiles are never silently replaced by
        the Worker profile.
        """
        mapping = {
            ModelRole.PLANNER: self.leader_profile,
            ModelRole.WORKER: self.worker_profile,
            ModelRole.ANALYZER: self.analyzer_profile,
            ModelRole.VERIFIER: self.verifier_profile,
        }
        if role not in mapping:
            raise ProviderError(
                f"no model profile mapping exists for the {role.value} role",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return mapping[role]

    def require_profile(self, role: ModelRole) -> ModelProfile:
        """Return the profile configured for a role, or fail with a structured error."""
        profile = self.profile_for_role(role)
        if profile is None:
            raise ProviderError(
                f"no model profile is configured for the {role.value} role",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        return profile

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        """Build settings from environment variables.

        Only the documented ``PRP_`` variables are accepted. Any other
        ``PRP_``-prefixed variable is an error, so a typo can never be
        silently ignored.
        """
        source = os.environ if env is None else env
        present = {key: value for key, value in source.items() if key.startswith(ENV_PREFIX)}
        unknown = sorted(set(present) - set(_ENV_FIELDS))
        if unknown:
            raise ValueError(
                "unknown settings environment variable(s): " + ", ".join(unknown)
            )
        values = {_ENV_FIELDS[key]: value for key, value in present.items()}
        # Environment values are always strings; pydantic coerces each one to its
        # declared field type. ``model_validate`` says that, where ``cls(**values)``
        # would claim every field is already a ``str``.
        return cls.model_validate(values)


def load_credential_profiles(credential_file: Path) -> dict[str, ModelProfile]:
    """Load model profiles from a credential file.

    The credential file should be a JSON file with a top-level object where
    each key is a profile name and each value is a ModelProfile object.

    Args:
        credential_file: Path to the credential JSON file

    Returns:
        Dictionary mapping profile names to ModelProfile instances

    Raises:
        FileNotFoundError: If credential file does not exist
        ValueError: If credential file is not valid JSON or schema mismatch
    """
    if not credential_file.exists():
        raise FileNotFoundError(f"Credential file not found: {credential_file}")

    content = credential_file.read_text(encoding="utf-8")
    raw_data = strict_json_loads(content)

    if not isinstance(raw_data, dict):
        raise ValueError("Credential file must contain a JSON object")

    profiles: dict[str, ModelProfile] = {}
    for name, profile_data in raw_data.items():
        if not isinstance(profile_data, dict):
            raise ValueError(f"Profile '{name}' must be a JSON object")
        profiles[name] = ModelProfile.model_validate(profile_data)

    return profiles
