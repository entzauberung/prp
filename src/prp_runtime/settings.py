"""Runtime settings.

Settings are plain data validated by Pydantic. Reading settings has no side
effect: no network access, no database connection, no logging reconfiguration.

Model endpoints and credentials are part of this server-side configuration.
A request never carries a provider URL or an API key.
"""

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.json_support import strict_json_loads
from prp_runtime.providers.base import ModelProfile

__all__ = ["ENV_PREFIX", "LogLevel", "Settings"]

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

ENV_PREFIX = "PRP_"

_ENV_FIELDS: dict[str, str] = {
    "PRP_DATABASE_PATH": "database_path",
    "PRP_MAX_REQUEST_BYTES": "max_request_bytes",
    "PRP_MAX_INPUT_CHARS": "max_input_chars",
    "PRP_LOG_LEVEL": "log_level",
    "PRP_LEADER_PROFILE": "leader_profile",
    "PRP_WORKER_PROFILE": "worker_profile",
    "PRP_CASCADE_PROFILES": "cascade_profiles",
}


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


ProfileValue = Annotated[ModelProfile, BeforeValidator(_profile_from_json)]
CascadeProfilesValue = Annotated[
    tuple[ModelProfile, ...],
    BeforeValidator(_cascade_profiles_from_json),
]


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
    leader_profile: ProfileValue | None = None
    worker_profile: ProfileValue | None = None
    cascade_profiles: CascadeProfilesValue = Field(default_factory=tuple)

    @model_validator(mode="after")
    def _profiles_are_consistent(self) -> "Settings":
        if self.leader_profile is not None and self.leader_profile.role is not ModelRole.PLANNER:
            raise ValueError("the leader profile must declare the PLANNER role")
        if self.worker_profile is not None and self.worker_profile.role is not ModelRole.WORKER:
            raise ValueError("the worker profile must declare the WORKER role")
        if any(profile.role is not ModelRole.WORKER for profile in self.cascade_profiles):
            raise ValueError("cascade profiles must declare the WORKER role")

        aliases = [profile.alias for profile in self.profiles]
        if len(aliases) != len(set(aliases)):
            raise ValueError("all configured profiles must use different aliases")
        return self

    @property
    def profiles(self) -> tuple[ModelProfile, ...]:
        """Every configured model profile."""
        primary_profiles = tuple(
            profile for profile in (self.leader_profile, self.worker_profile) if profile is not None
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

    def require_profile(self, role: ModelRole) -> ModelProfile:
        """Return the profile configured for a role, or fail with a structured error."""
        profile = (
            self.leader_profile if role is ModelRole.PLANNER else self.worker_profile
        )
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
