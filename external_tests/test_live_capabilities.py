"""Real capability probes for profiles that already passed provider smoke."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from external_tests.capability_ledger import CapabilityEntry, CapabilityStore
from external_tests.credential_loader import PROFILE_CONTRACTS
from external_tests.result_ledger import LedgerStore
from external_tests.support import (
    ExternalConfig,
    ExternalGateError,
    ExternalProfile,
    create_external_http_client,
    validate_external_url,
)
from external_tests.test_live_deepseek import _create_adapter, _profile_for_runtime
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import ProviderToolDescriptor
from prp_runtime.providers.base import ProviderRequest, ProviderResponse

CAPABILITY_ALIASES = tuple(PROFILE_CONTRACTS)
CAPABILITY_SCHEMA = (
    '{"type":"object","properties":{"ready":{"type":"string"}},'
    '"required":["ready"],"additionalProperties":false}'
)


def _capability_path() -> str:
    value = os.environ.get("PRP_LIVE_CAPABILITY_FILE")
    if value is None or not value.strip():
        raise ExternalGateError("PRP_LIVE_CAPABILITY_FILE is required for capability evidence")
    return value


def _profile_by_alias(config: ExternalConfig, alias: str) -> ExternalProfile:
    matches = [profile for profile in config.profiles if profile.alias == alias]
    if len(matches) != 1:
        raise ExternalGateError(f"capability matrix must contain exactly one {alias}")
    return matches[0]


def _entry(
    profile: ExternalProfile,
    capability: str,
    status: str,
    attempt_id: str,
    response: ProviderResponse | None = None,
    error_code: str | None = None,
    diagnostic: str | None = None,
) -> CapabilityEntry:
    usage = response.usage if response is not None else None
    text = response.text if response is not None else None
    return CapabilityEntry(
        # Capability probes are repeatable actual attempts; retain each fact
        # instead of overwriting an earlier probe under a fixed scenario key.
        scenario_id=(
            f"wo-002-st-003-{profile.alias.lower()}-{capability}-{attempt_id}"
        ),
        alias=profile.alias,
        model_id=profile.model_id,
        protocol=profile.protocol,
        endpoint_host=urlsplit(profile.base_url).hostname or "unknown",
        capability=capability,
        status=status,
        actual_or_simulated="ACTUAL",
        attempt_id=attempt_id,
        provider_request_id=(
            response.provider_request_id if response is not None else None
        ),
        finish_reason=(response.finish_reason.value if response is not None else None),
        input_tokens=usage.input_tokens if usage is not None else None,
        output_tokens=usage.output_tokens if usage is not None else None,
        latency_ms=usage.elapsed_ms if usage is not None else None,
        error_code=error_code,
        output_sha256=(
            hashlib.sha256(text.encode("utf-8")).hexdigest()
            if isinstance(text, str)
            else None
        ),
        recorded_at=datetime.now(UTC).isoformat(),
        diagnostic=diagnostic,
    )


def _tool_response_classification(
    response: ProviderResponse,
) -> tuple[str, str | None, str]:
    """Classify a strict tool probe without treating text fallback as a parser defect."""
    if response.tool_calls:
        return "PASS", None, "valid_function_call"
    if response.text is not None:
        return (
            "UPSTREAM_UNSUPPORTED",
            "TOOL_CALL_TEXT_FALLBACK",
            "provider_returned_text_without_function_call",
        )
    return "UPSTREAM_ERROR", "EMPTY_TOOL_RESPONSE", "provider_returned_no_output"


def _tool_error_diagnostic(error: ProviderError) -> str:
    """Keep parser/provider distinctions bounded and free of raw response bodies."""
    message = error.detail.message.lower()
    if error.code is ErrorCode.PROVIDER_INVALID_RESPONSE:
        if "function_call" in message or "tool call" in message:
            return "MALFORMED_FUNCTION_CALL_RESPONSE"
        return "PARSER_RESPONSE_REJECTION"
    return f"UPSTREAM_{error.code.value.upper()}"


async def _probe(profile: ExternalProfile) -> tuple[CapabilityEntry, CapabilityEntry]:
    runtime_profile = _profile_for_runtime(profile).model_copy(
        update={"supports_structured_output": True}
    )
    client = create_external_http_client((urlsplit(profile.base_url).hostname or "",))
    adapter = _create_adapter(runtime_profile, client)
    descriptor = ProviderToolDescriptor(
        name="report_status",
        description="Report a ready status.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    )
    try:
        structured_id = f"probe_{uuid.uuid4().hex}"
        try:
            structured = await adapter.complete(
                ProviderRequest.for_profile(
                    runtime_profile,
                    input='Return JSON with ready set to "ok".',
                    json_schema=CAPABILITY_SCHEMA,
                    max_output_tokens=64,
                )
            )
            parsed = json.loads(structured.text or "")
            structured_status = (
                "PASS"
                if isinstance(parsed, dict) and parsed.get("ready")
                else "UPSTREAM_ERROR"
            )
            structured_error = (
                None if structured_status == "PASS" else "INVALID_STRUCTURED_RESPONSE"
            )
            structured_entry = _entry(
                profile,
                "structured_output",
                structured_status,
                structured_id,
                structured,
                structured_error,
            )
        except ProviderError as error:
            structured_entry = _entry(
                profile,
                "structured_output",
                "UPSTREAM_ERROR",
                structured_id,
                error_code=error.code.value,
            )
        except (ValueError, TypeError) as error:
            structured_entry = _entry(
                profile,
                "structured_output",
                "UPSTREAM_ERROR",
                structured_id,
                error_code=type(error).__name__,
            )

        tool_id = f"probe_{uuid.uuid4().hex}"
        try:
            tool_response = await adapter.complete(
                ProviderRequest.for_profile(
                    runtime_profile,
                    input="Call report_status now.",
                    tools=(descriptor,),
                    max_output_tokens=64,
                )
            )
            tool_status, tool_error, diagnostic = _tool_response_classification(tool_response)
            tool_entry = _entry(
                profile,
                "tool_call",
                tool_status,
                tool_id,
                tool_response,
                error_code=tool_error,
                diagnostic=diagnostic,
            )
        except ProviderError as error:
            tool_entry = _entry(
                profile,
                "tool_call",
                "UPSTREAM_ERROR",
                tool_id,
                error_code=error.code.value,
                diagnostic=_tool_error_diagnostic(error),
            )
        return structured_entry, tool_entry
    finally:
        await client.aclose()


@pytest.mark.live_provider
@pytest.mark.parametrize("alias", CAPABILITY_ALIASES)
def test_real_capability_probe(
    alias: str,
    external_config: ExternalConfig,
) -> None:
    profile = _profile_by_alias(external_config, alias)
    provider_passes = {
        entry.alias
        for entry in LedgerStore(Path(os.environ["PRP_LIVE_RESULT_FILE"])).read()
        if entry.protocol == profile.protocol
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }
    capability_passes = {
        entry.alias
        for entry in CapabilityStore(Path(_capability_path())).read()
        if entry.model_id == profile.model_id
        and entry.protocol == profile.protocol
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }
    if alias not in provider_passes | capability_passes:
        pytest.skip("capability probe is limited to profiles with a real smoke PASS")
    validate_external_url(profile.base_url, external_config.allowed_hosts)
    entries = asyncio.run(_probe(profile))
    CapabilityStore(Path(_capability_path())).merge(entries)
