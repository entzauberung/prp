"""Fail-closed configuration helpers for explicit external validation."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import shutil
import tempfile
from collections.abc import Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx
from pydantic import SecretStr

DEFAULT_MATRIX_PATH = Path(__file__).with_name("model_matrix.json")
_ENV_NAME = re.compile(r"^PRP_EXTERNAL_[A-Z0-9_]+$")
_PROFILE_SELECTION_ENV = "PRP_EXTERNAL_PROFILE_ALIASES"


class ExternalGateError(RuntimeError):
    """A safe, non-secret configuration failure."""


@dataclass(frozen=True, slots=True)
class ExternalProfile:
    """Runtime profile with secret and URL values excluded from representations."""

    alias: str
    vendor: str
    model_id: str
    protocol: str
    base_url: str = field(repr=False)
    api_key: SecretStr = field(repr=False)

    def __repr__(self) -> str:
        return (
            "ExternalProfile("
            f"alias={self.alias!r}, vendor={self.vendor!r}, model_id={self.model_id!r}, "
            f"protocol={self.protocol!r}, base_url_host={_url_host(self.base_url)!r}, "
            "api_key='**********')"
        )

    def redacted(self) -> dict[str, str]:
        """Return the only form that may be placed in diagnostics or JSON."""

        return {
            "alias": self.alias,
            "vendor": self.vendor,
            "model_id": self.model_id,
            "protocol": self.protocol,
            "base_url_host": _url_host(self.base_url),
            "api_key": "**********",
        }


@dataclass(frozen=True, slots=True)
class _ProfileSpec:
    """Validated non-secret matrix metadata for one external profile."""

    alias: str
    vendor: str
    protocol: str
    base_url_env: str
    model_env: str
    api_key_env: str
    allowed_host_env: str | None = None
    optional: bool = False


@dataclass(frozen=True, slots=True)
class OptionalProfileStatus:
    """Classified optional profile state without any secret or full URL."""

    alias: str
    status: str
    reason: str

    def redacted(self) -> dict[str, str]:
        return {"alias": self.alias, "status": self.status, "reason": self.reason}


@dataclass(frozen=True, slots=True)
class ExternalConfig:
    """Loaded external profiles and the host names supplied by the matrix."""

    allowed_hosts: tuple[str, ...]
    profiles: tuple[ExternalProfile, ...]
    optional_statuses: tuple[OptionalProfileStatus, ...] = ()

    def __repr__(self) -> str:
        aliases = tuple(profile.alias for profile in self.profiles)
        optional = tuple(status.alias for status in self.optional_statuses)
        return (
            f"ExternalConfig(allowed_hosts={self.allowed_hosts!r}, aliases={aliases!r}, "
            f"optional={optional!r})"
        )


def require_external_opt_in(environ: Mapping[str, str] | None = None) -> None:
    """Require the exact opt-in value without exposing any environment values."""

    source = os.environ if environ is None else environ
    if source.get("PRP_EXTERNAL_TESTS") != "1":
        raise ExternalGateError(
            "external validation is disabled; set PRP_EXTERNAL_TESTS=1 in the parent process"
        )


def require_external_profile_selection(
    environ: Mapping[str, str] | None = None,
    known_aliases: Collection[str] | None = None,
) -> tuple[str, ...]:
    """Parse and validate the explicitly selected profile aliases."""

    source = os.environ if environ is None else environ
    raw_selection = source.get(_PROFILE_SELECTION_ENV)
    if raw_selection is None:
        raise ExternalGateError(
            f"missing required environment variable {_PROFILE_SELECTION_ENV}"
        )
    if not raw_selection.strip():
        raise ExternalGateError(f"empty profile alias in {_PROFILE_SELECTION_ENV}")

    aliases = tuple(part.strip() for part in raw_selection.split(","))
    if any(not alias for alias in aliases):
        raise ExternalGateError(f"empty profile alias in {_PROFILE_SELECTION_ENV}")
    if len(set(aliases)) != len(aliases):
        raise ExternalGateError(f"duplicate profile alias in {_PROFILE_SELECTION_ENV}")

    if known_aliases is not None:
        known = set(known_aliases)
        for alias in aliases:
            if alias not in known:
                raise ExternalGateError(
                    f"unknown profile alias in {_PROFILE_SELECTION_ENV}: {alias}"
                )
    return aliases


def load_external_config(
    environ: Mapping[str, str] | None = None,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> ExternalConfig:
    """Load only named external variables after the explicit opt-in gate."""

    source = os.environ if environ is None else environ
    require_external_opt_in(source)
    matrix = _read_matrix(matrix_path)
    allowed_hosts = _read_allowed_hosts(matrix)
    active_specs = _validate_profile_specs(_read_profiles(matrix))
    optional_specs = _validate_profile_specs(
        _read_optional_profiles(matrix), optional=True, existing_specs=active_specs
    )
    profile_specs = active_specs + optional_specs
    selected_aliases = require_external_profile_selection(
        source, tuple(spec.alias for spec in profile_specs)
    )
    selected_alias_set = set(selected_aliases)

    profiles: list[ExternalProfile] = []
    selected_hosts: list[str] = []
    optional_statuses: list[OptionalProfileStatus] = []
    for spec in profile_specs:
        if spec.alias not in selected_alias_set:
            continue

        if spec.optional:
            optional_result = _resolve_optional_spec(source, spec, allowed_hosts)
            if optional_result is None:
                optional_statuses.append(
                    OptionalProfileStatus(
                        alias=spec.alias,
                        status="TERRA_NOT_CONFIGURED",
                        reason="complete optional metadata is not available",
                    )
                )
                continue
            if isinstance(optional_result, OptionalProfileStatus):
                optional_statuses.append(optional_result)
                continue
            base_url, model_id, api_key = optional_result
        else:
            base_url = _required_env(source, spec.base_url_env, spec.alias)
            model_id = _required_env(source, spec.model_env, spec.alias)
            api_key = _required_env(source, spec.api_key_env, spec.alias)

        _validate_base_url(base_url, spec.base_url_env, spec.alias)
        base_url_host = _url_host(base_url)
        allowed_host_by_lower = {host.lower(): host for host in allowed_hosts}
        selected_host = allowed_host_by_lower.get(base_url_host.lower())
        if selected_host is None:
            raise ExternalGateError(
                f"base URL host in {spec.base_url_env} for {spec.alias} is outside "
                "the model matrix allowlist"
            )
        if selected_host not in selected_hosts:
            selected_hosts.append(selected_host)
        profiles.append(
            ExternalProfile(
                alias=spec.alias,
                vendor=spec.vendor,
                model_id=model_id,
                protocol=spec.protocol,
                base_url=base_url,
                api_key=SecretStr(api_key),
            )
        )

    if not profiles and not optional_statuses:
        raise ExternalGateError("model matrix contains no in-scope profiles")
    return ExternalConfig(tuple(selected_hosts), tuple(profiles), tuple(optional_statuses))


def load_external_profiles(
    environ: Mapping[str, str] | None = None,
    matrix_path: Path = DEFAULT_MATRIX_PATH,
) -> tuple[ExternalProfile, ...]:
    """Convenience wrapper for callers that only need the loaded profiles."""

    return load_external_config(environ, matrix_path).profiles


def _read_matrix(path: Path) -> dict[str, Any]:
    try:
        raw: Any = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExternalGateError("model matrix is unreadable or malformed") from exc
    if not isinstance(raw, dict):
        raise ExternalGateError("model matrix must contain a JSON object")
    return raw


def _read_allowed_hosts(matrix: Mapping[str, Any]) -> list[str]:
    raw_hosts = matrix.get("allowed_hosts")
    if not isinstance(raw_hosts, list) or not raw_hosts or not all(
        isinstance(host, str) and _is_host(host) for host in raw_hosts
    ):
        raise ExternalGateError("model matrix allowed_hosts is malformed")
    return list(dict.fromkeys(raw_hosts))


def _read_profiles(matrix: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_profiles = matrix.get("profiles")
    if not isinstance(raw_profiles, list) or not all(
        isinstance(profile, dict) for profile in raw_profiles
    ):
        raise ExternalGateError("model matrix profiles is malformed")
    return raw_profiles


def _read_optional_profiles(matrix: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw_profiles = matrix.get("optional_profiles", [])
    if not isinstance(raw_profiles, list) or not all(
        isinstance(profile, dict) for profile in raw_profiles
    ):
        raise ExternalGateError("model matrix optional_profiles is malformed")
    return raw_profiles


def _validate_profile_specs(
    profile_specs: list[Mapping[str, Any]],
    *,
    optional: bool = False,
    existing_specs: tuple[_ProfileSpec, ...] = (),
) -> tuple[_ProfileSpec, ...]:
    """Validate all matrix metadata without reading any profile values."""

    used_env_names: set[str] = {
        "PRP_EXTERNAL_TESTS",
        _PROFILE_SELECTION_ENV,
        *(name for spec in existing_specs for name in _spec_env_names(spec)),
    }
    seen_aliases: set[str] = {spec.alias for spec in existing_specs}
    validated: list[_ProfileSpec] = []
    for spec in profile_specs:
        alias = _required_text(spec, "alias", "matrix")
        if alias in seen_aliases:
            raise ExternalGateError(f"duplicate profile alias in model matrix: {alias}")
        seen_aliases.add(alias)
        vendor = _required_text(spec, "vendor", alias)
        protocol = _required_text(spec, "protocol", alias)
        allowed_host_env = None
        if optional:
            allowed_host_env = _env_name(spec, "allowed_host_env", alias, used_env_names)
        validated.append(
            _ProfileSpec(
                alias=alias,
                vendor=vendor,
                protocol=protocol,
                base_url_env=_env_name(spec, "base_url_env", alias, used_env_names),
                model_env=_env_name(spec, "model_env", alias, used_env_names),
                api_key_env=_env_name(spec, "api_key_env", alias, used_env_names),
                allowed_host_env=allowed_host_env,
                optional=optional,
            )
        )
    return tuple(validated)


def _spec_env_names(spec: _ProfileSpec) -> tuple[str, ...]:
    return tuple(
        name
        for name in (spec.base_url_env, spec.model_env, spec.api_key_env, spec.allowed_host_env)
        if name is not None
    )


def _resolve_optional_spec(
    environ: Mapping[str, str],
    spec: _ProfileSpec,
    allowed_hosts: Collection[str],
) -> tuple[str, str, str] | OptionalProfileStatus | None:
    assert spec.optional and spec.allowed_host_env is not None
    values = {
        "base URL": environ.get(spec.base_url_env, "").strip(),
        "model": environ.get(spec.model_env, "").strip(),
        "API key": environ.get(spec.api_key_env, "").strip(),
        "allowed host": environ.get(spec.allowed_host_env, "").strip(),
    }
    if not all(values.values()):
        return None
    if not _is_host(values["allowed host"]):
        return OptionalProfileStatus(
            alias=spec.alias,
            status="TERRA_NOT_CONFIGURED",
            reason="optional allowed host is malformed",
        )
    try:
        _validate_base_url(values["base URL"], spec.base_url_env, spec.alias)
        base_host = _url_host(values["base URL"])
    except ExternalGateError:
        return OptionalProfileStatus(
            alias=spec.alias,
            status="TERRA_NOT_CONFIGURED",
            reason="optional base URL is not a valid HTTPS URL",
        )
    if base_host != values["allowed host"]:
        return OptionalProfileStatus(
            alias=spec.alias,
            status="TERRA_NOT_CONFIGURED",
            reason="optional base URL host does not match its explicit host",
        )
    if values["allowed host"].lower() not in {host.lower() for host in allowed_hosts}:
        return OptionalProfileStatus(
            alias=spec.alias,
            status="TERRA_NOT_CONFIGURED",
            reason="optional host is not explicitly admitted to the active allowlist",
        )
    return values["base URL"], values["model"], values["API key"]


def _required_text(spec: Mapping[str, Any], field_name: str, context: str) -> str:
    value = spec.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ExternalGateError(f"missing or malformed {field_name} for {context}")
    return value.strip()


def _env_name(
    spec: Mapping[str, Any],
    field_name: str,
    alias: str,
    used_names: set[str],
) -> str:
    value = _required_text(spec, field_name, alias)
    if not _ENV_NAME.fullmatch(value) or value in used_names:
        raise ExternalGateError(f"malformed or duplicate {field_name} for {alias}")
    used_names.add(value)
    return value


def _required_env(environ: Mapping[str, str], env_name: str, alias: str) -> str:
    value = environ.get(env_name)
    if value is None or not value.strip():
        raise ExternalGateError(f"missing required environment variable {env_name} for {alias}")
    return value


def _validate_base_url(value: str, env_name: str, alias: str) -> None:
    try:
        parsed = urlsplit(value)
        valid = (
            parsed.scheme == "https"
            and parsed.hostname is not None
            and parsed.username is None
            and parsed.password is None
            and not parsed.query
            and not parsed.fragment
        )
    except ValueError:
        valid = False
    if not valid:
        raise ExternalGateError(f"malformed HTTPS base URL in {env_name} for {alias}")


def _is_host(value: str) -> bool:
    return value == value.strip() and bool(value) and "/" not in value and ":" not in value


def _url_host(value: str) -> str:
    try:
        return urlsplit(value).hostname or "unknown"
    except ValueError:
        return "unknown"


def validate_external_url(url: str | httpx.URL, allowed_hosts: Collection[str]) -> None:
    """Reject unsafe URLs before an HTTP transport can open a socket."""

    try:
        parsed = urlsplit(str(url))
        host = parsed.hostname
        is_ip = False
        if host is not None:
            try:
                ipaddress.ip_address(host)
                is_ip = True
            except ValueError:
                pass
        valid = (
            parsed.scheme == "https"
            and host is not None
            and parsed.username is None
            and parsed.password is None
            and not is_ip
            and host.lower() in {item.lower() for item in allowed_hosts}
        )
    except ValueError:
        valid = False
    if not valid:
        raise ExternalGateError("request URL is not HTTPS or is outside the host allowlist")


class AllowlistedAsyncTransport(httpx.AsyncBaseTransport):
    """Validate every request and redirect target around an httpx transport."""

    def __init__(self, inner: httpx.AsyncBaseTransport, allowed_hosts: Collection[str]) -> None:
        self._inner = inner
        self._allowed_hosts = tuple(allowed_hosts)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        validate_external_url(request.url, self._allowed_hosts)
        response = await self._inner.handle_async_request(request)
        if 300 <= response.status_code < 400:
            location = response.headers.get("location")
            if location:
                redirect_url = urljoin(str(request.url), location)
                try:
                    validate_external_url(redirect_url, self._allowed_hosts)
                except ExternalGateError:
                    await response.aclose()
                    raise ExternalGateError(
                        "redirect target is outside the host allowlist"
                    ) from None
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


def create_external_http_client(
    allowed_hosts: Collection[str],
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Create an HTTPS-only client with no ambient proxy or redirect behavior."""

    inner = transport or httpx.AsyncHTTPTransport(retries=0, verify=True, trust_env=False)
    return httpx.AsyncClient(
        transport=AllowlistedAsyncTransport(inner, allowed_hosts),
        follow_redirects=False,
        trust_env=False,
        verify=True,
    )


@dataclass(frozen=True, slots=True)
class TemporaryResources:
    """Temporary database/workspace paths with idempotent cleanup."""

    root: Path
    database_path: Path
    workspace_path: Path

    def cleanup(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


@contextmanager
def temporary_external_resources(root: Path | None = None) -> Iterator[TemporaryResources]:
    """Create isolated paths outside the project root and clean them up safely."""

    parent = Path(tempfile.mkdtemp(prefix="prp-external-parent-")) if root is None else root
    _reject_project_overlap(parent)
    resource_root = Path(tempfile.mkdtemp(prefix="validation-", dir=str(parent)))
    resources = TemporaryResources(
        root=resource_root,
        database_path=resource_root / "validation.sqlite3",
        workspace_path=resource_root / "workspace",
    )
    try:
        resources.database_path.touch()
        resources.workspace_path.mkdir()
        yield resources
    finally:
        resources.cleanup()
        if root is None:
            shutil.rmtree(parent, ignore_errors=True)


def _reject_project_overlap(root: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]
    resolved_root = root.resolve()
    if (
        resolved_root == project_root
        or resolved_root.is_relative_to(project_root)
        or project_root.is_relative_to(resolved_root)
    ):
        raise ExternalGateError("temporary resource root overlaps the project root")


@dataclass(frozen=True, slots=True)
class CallReservation:
    """An opaque reservation that represents one request attempt."""

    alias: str
    attempt_number: int
    output_tokens: int


class BudgetCounter:
    """Thread-safe request-attempt and output-token reservation counter."""

    def __init__(
        self,
        *,
        max_provider_calls: int,
        max_attempts_per_alias: int,
        max_successful_calls_per_alias: int,
        max_output_tokens: int,
        max_total_output_tokens: int | None = None,
    ) -> None:
        limits = (
            max_provider_calls,
            max_attempts_per_alias,
            max_successful_calls_per_alias,
            max_output_tokens,
        )
        if any(limit <= 0 for limit in limits):
            raise ValueError("budget limits must be positive")
        if max_total_output_tokens is not None and max_total_output_tokens <= 0:
            raise ValueError("max_total_output_tokens must be positive")
        self._max_provider_calls = max_provider_calls
        self._max_attempts_per_alias = max_attempts_per_alias
        self._max_successful_calls_per_alias = max_successful_calls_per_alias
        self._max_output_tokens = max_output_tokens
        self._max_total_output_tokens = max_total_output_tokens
        self._lock = RLock()
        self._attempts = 0
        self._successful_calls: dict[str, int] = {}
        self._alias_attempts: dict[str, int] = {}
        self._reserved_tokens = 0
        self._observed_output_tokens = 0
        self._reservations: set[CallReservation] = set()

    @property
    def attempts(self) -> int:
        with self._lock:
            return self._attempts

    @property
    def observed_output_tokens(self) -> int:
        with self._lock:
            return self._observed_output_tokens

    def successful_calls(self, alias: str) -> int:
        with self._lock:
            return self._successful_calls.get(alias, 0)

    def reserve(self, alias: str, output_tokens: int) -> CallReservation:
        """Atomically reserve a request before handing it to an HTTP client."""

        if not alias.strip():
            raise ValueError("alias must not be empty")
        if output_tokens <= 0 or output_tokens > self._max_output_tokens:
            raise ExternalGateError("requested output token limit exceeds the budget")
        with self._lock:
            if self._attempts >= self._max_provider_calls:
                raise ExternalGateError("global provider call limit reached")
            alias_attempts = self._alias_attempts.get(alias, 0)
            if alias_attempts >= self._max_attempts_per_alias:
                raise ExternalGateError(f"alias attempt limit reached for {alias}")
            if self._successful_calls.get(alias, 0) >= self._max_successful_calls_per_alias:
                raise ExternalGateError(f"alias success limit reached for {alias}")
            if (
                self._max_total_output_tokens is not None
                and self._reserved_tokens + output_tokens > self._max_total_output_tokens
            ):
                raise ExternalGateError("output token budget reached")

            self._attempts += 1
            self._alias_attempts[alias] = alias_attempts + 1
            self._reserved_tokens += output_tokens
            reservation = CallReservation(alias, alias_attempts + 1, output_tokens)
            self._reservations.add(reservation)
            return reservation

    def settle(
        self,
        reservation: CallReservation,
        *,
        success: bool,
        observed_output_tokens: int | None = None,
    ) -> None:
        """Commit evidence for a reservation; failures still consume the attempt."""

        if observed_output_tokens is not None and observed_output_tokens < 0:
            raise ValueError("observed_output_tokens must not be negative")
        with self._lock:
            if reservation not in self._reservations:
                raise ExternalGateError("unknown or already settled call reservation")
            self._reservations.remove(reservation)
            self._reserved_tokens -= reservation.output_tokens
            if success:
                successful = self._successful_calls.get(reservation.alias, 0) + 1
                if successful > self._max_successful_calls_per_alias:
                    raise ExternalGateError("alias success limit reached")
                self._successful_calls[reservation.alias] = successful
            if observed_output_tokens is not None:
                self._observed_output_tokens += observed_output_tokens
