"""Fail-closed local dispatch planning for an authorized Bridge claim.

This module does not execute a handler. It turns a server claim plus a local
registry into a bounded plan for the later local executor. A claim can select a
registered name and declared effect, but it can never supply executable code,
the local root or a handler.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from prp_runtime.domain.enums import BridgeClaimStatus, ToolCallStatus, ToolEffect
from prp_runtime.domain.models import (
    BRIDGE_PROTOCOL_VERSION,
    MAX_BRIDGE_OUTPUT_BYTES,
    MAX_BRIDGE_LEASE_TOTAL_SECONDS,
    ClientCapabilityDescriptor,
    ClientHandshakeRequest,
    ErrorCategory,
    ErrorInfo,
    fingerprint_client_capabilities,
)
from prp_runtime.domain.values import new_work_unit_id, utc_now
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES, ToolCall, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolRegistry

__all__ = [
    "BridgeDispatchError",
    "BridgeDispatchPlan",
    "BridgeExecutor",
]

_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_PATH_ARGUMENT_KEYS = frozenset(
    {
        "path",
        "paths",
        "root",
        "file",
        "files",
        "directory",
        "directories",
        "target",
        "targets",
    }
)
_UNIFIED_DIFF_ARGUMENT_KEY = "unified_diff"
_CLAIM_KEYS = frozenset(
    {
        "claim_id",
        "call_id",
        "run_id",
        "work_unit_id",
        "session_id",
        "workspace_id",
        "snapshot_id",
        "owner_id",
        "client_id",
        "claimant_id",
        "idempotency_key",
        "fingerprint",
        "status",
        "claimed_at",
        "requested_at",
        "expires_at",
        "closed_at",
        "tool_name",
        "effect",
        "arguments",
        "scope",
    }
)


class BridgeDispatchError(ValueError):
    """A claim cannot be mapped to a safe local registry operation."""


@dataclass(frozen=True)
class BridgeDispatchPlan:
    """Bounded local facts needed by a later handler executor."""

    call_id: str
    run_id: str
    workspace_id: str
    work_unit_id: str
    tool_name: str
    effect: ToolEffect
    arguments: BaseModel
    scope_paths: tuple[str, ...]
    workspace_root: Path
    definition: ToolDefinition
    requested_at: datetime
    expires_at: datetime
    snapshot_id: str | None = None


class BridgeExecutor:
    """Validate claims against a server-independent local ToolRegistry."""

    def __init__(
        self,
        registry: ToolRegistry,
        workspace_root: str | Path,
        *,
        timeout_seconds: float = 30.0,
    ) -> None:
        if not isinstance(registry, ToolRegistry):
            raise TypeError("BridgeExecutor requires a ToolRegistry")
        if timeout_seconds <= 0 or timeout_seconds > 60:
            raise ValueError("Bridge executor timeout must be between 0 and 60 seconds")
        root = Path(workspace_root).expanduser()
        if not root.is_absolute():
            raise BridgeDispatchError("Bridge workspace root must be absolute local configuration")
        normalized = Path(os.path.normpath(os.fspath(root)))
        if normalized == Path("/"):
            raise BridgeDispatchError("Bridge workspace root must not be the filesystem root")
        if normalized.is_symlink():
            raise BridgeDispatchError("Bridge workspace root must not be a symlink")
        if not normalized.exists():
            raise BridgeDispatchError("Bridge workspace root must exist")
        if not normalized.is_dir():
            raise BridgeDispatchError("Bridge workspace root must be a directory")
        self._registry = registry
        self._workspace_root = normalized
        self._timeout_seconds = timeout_seconds

    @property
    def workspace_root(self) -> Path:
        """Return the configured local root, which never comes from a claim."""
        return self._workspace_root

    def capability_descriptor(self) -> ClientCapabilityDescriptor:
        """Advertise sorted registry tools and effects without the local root."""
        tools = tuple(sorted(self._registry.names))
        effects = tuple(
            sorted(
                {definition.effect for definition in self._registry.definitions},
                key=lambda item: item.value,
            )
        )
        max_output = max(
            (definition.max_output_bytes for definition in self._registry.definitions),
            default=MAX_BRIDGE_OUTPUT_BYTES,
        )
        return ClientCapabilityDescriptor(
            tools=tools,
            effects=effects,
            max_output_bytes=min(max_output, MAX_BRIDGE_OUTPUT_BYTES),
            max_runtime_ms=max(1, int(self._timeout_seconds * 1000)),
        )

    def handshake_request(
        self,
        client_id: str,
        *,
        workspace_id: str | None = None,
    ) -> ClientHandshakeRequest:
        """Build a model-free handshake from the local registry only."""
        capabilities = self.capability_descriptor()
        return ClientHandshakeRequest(
            client_id=client_id,
            protocol_version=BRIDGE_PROTOCOL_VERSION,
            capabilities=capabilities,
            fingerprint=fingerprint_client_capabilities(capabilities),
            workspace_id=workspace_id,
        )

    def verify_advertised_capabilities(
        self, capabilities: ClientCapabilityDescriptor
    ) -> None:
        """Reject capability claims that do not match the local registry."""
        expected = self.capability_descriptor()
        if capabilities != expected:
            raise BridgeDispatchError(
                "advertised capabilities do not match the local registry"
            )

    def plan(
        self,
        claim: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> BridgeDispatchPlan:
        """Validate one JSON claim without touching the filesystem or transport."""
        if not isinstance(claim, Mapping):
            raise BridgeDispatchError("Bridge claim must be an object")
        if any(key in claim for key in ("now", "observed_at", "clock")):
            raise BridgeDispatchError("Bridge claim contains unsupported fields")
        unknown = set(claim) - _CLAIM_KEYS
        if unknown:
            raise BridgeDispatchError("Bridge claim contains unsupported fields")
        if claim.get("status") != BridgeClaimStatus.ACTIVE.value:
            raise BridgeDispatchError("Bridge claim is not active")
        if claim.get("closed_at") is not None:
            raise BridgeDispatchError("active Bridge claim must not be closed")

        call_id = _required_identifier(claim, "call_id", prefix="tc_")
        run_id = _required_identifier(claim, "run_id", prefix="run_")
        workspace_id = _required_identifier(claim, "workspace_id", prefix="ws_")
        work_unit_value = claim.get("work_unit_id")
        work_unit_id = (
            new_work_unit_id()
            if work_unit_value is None
            else _required_identifier_value(work_unit_value, "work_unit_id", prefix="wu_")
        )
        snapshot_id = claim.get("snapshot_id")
        if snapshot_id is not None:
            snapshot_id = _required_identifier_value(snapshot_id, "snapshot_id", prefix="snap_")
        tool_name = _required_tool_name(claim.get("tool_name"))
        try:
            effect = ToolEffect(str(claim.get("effect")))
        except ValueError as error:
            raise BridgeDispatchError("Bridge claim effect is invalid") from error
        if effect is ToolEffect.NETWORK:
            raise BridgeDispatchError("Bridge claim names an unsupported network effect")
        claimed_at = _aware_timestamp(claim.get("claimed_at"), "claimed_at")
        expires_at = _aware_timestamp(claim.get("expires_at"), "expires_at")
        if expires_at <= claimed_at:
            raise BridgeDispatchError("Bridge claim lease is invalid")
        lease_ceiling = claimed_at + timedelta(seconds=MAX_BRIDGE_LEASE_TOTAL_SECONDS)
        if expires_at > lease_ceiling:
            expires_at = lease_ceiling
        requested_at = _aware_timestamp(
            claim.get("requested_at", claimed_at.isoformat()), "requested_at"
        )
        observed_at = now or datetime.now(tz=UTC)
        if observed_at.tzinfo is None:
            raise BridgeDispatchError("Bridge dispatch time must be timezone-aware")
        if expires_at <= observed_at.astimezone(UTC):
            raise BridgeDispatchError("Bridge claim lease has expired")

        arguments = claim.get("arguments")
        if not isinstance(arguments, Mapping):
            raise BridgeDispatchError("Bridge claim arguments must be an object")
        scope_paths = _scope_paths(claim.get("scope"))
        self._assert_argument_paths(arguments, scope_paths)
        try:
            definition = self._registry.get(tool_name)
        except KeyError as error:
            raise BridgeDispatchError("Bridge claim names an unknown local tool") from error
        if definition.effect is not effect:
            raise BridgeDispatchError("Bridge claim effect does not match local registry")
        try:
            validated_arguments = definition.validate_arguments(arguments)
        except Exception as error:
            raise BridgeDispatchError("Bridge claim arguments do not match local tool") from error
        return BridgeDispatchPlan(
            call_id=call_id,
            run_id=run_id,
            workspace_id=workspace_id,
            work_unit_id=work_unit_id,
            tool_name=tool_name,
            effect=effect,
            arguments=validated_arguments,
            scope_paths=scope_paths,
            workspace_root=self._workspace_root,
            definition=definition,
            requested_at=requested_at.astimezone(UTC),
            expires_at=expires_at.astimezone(UTC),
            snapshot_id=snapshot_id,
        )

    async def execute(
        self,
        claim_or_plan: Mapping[str, Any] | BridgeDispatchPlan,
        *,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        """Execute one planned local handler and return a server-safe payload."""
        plan = (
            claim_or_plan
            if isinstance(claim_or_plan, BridgeDispatchPlan)
            else self.plan(claim_or_plan, now=now)
        )
        observed_at = now or datetime.now(tz=UTC)
        if observed_at.tzinfo is None:
            raise BridgeDispatchError("Bridge dispatch time must be timezone-aware")
        if plan.expires_at <= observed_at.astimezone(UTC):
            raise BridgeDispatchError("Bridge claim lease expired before local execution")
        call = _tool_call(plan)
        context = ExecutionContext(
            call=call,
            arguments=plan.arguments,
            workspace_id=plan.workspace_id,
            resolved_paths=plan.scope_paths,
        )
        try:
            raw_result = await asyncio.wait_for(
                plan.definition.handler(context), timeout=self._timeout_seconds
            )
            result = self._coerce_result(call, raw_result, plan.definition.max_output_bytes)
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            result = _failed_result(
                call,
                ErrorCategory.TIMEOUT,
                "local tool handler timed out",
            )
        except BridgeDispatchError:
            raise
        except Exception:
            result = _failed_result(
                call,
                ErrorCategory.INTERNAL,
                "local tool handler failed",
            )
        return _submission_payload(result, plan.workspace_root)

    @classmethod
    def _coerce_result(
        cls,
        call: ToolCall,
        raw_result: Any,
        output_limit: int,
    ) -> ToolResult:
        limit = min(output_limit, MAX_TOOL_OUTPUT_BYTES)
        if isinstance(raw_result, ToolResult):
            if raw_result.call_id != call.call_id:
                raise BridgeDispatchError("local tool result call_id does not match the claim")
            output, output_truncated = _bound_text(raw_result.output, limit)
            return ToolResult(
                call_id=call.call_id,
                status=raw_result.status,
                result=None if output_truncated else raw_result.result,
                output=output,
                truncated=raw_result.truncated or output_truncated,
                changed_paths=raw_result.changed_paths,
                exit_code=raw_result.exit_code,
                error=raw_result.error,
                completed_at=raw_result.completed_at,
            )
        if raw_result is None:
            return ToolResult(
                call_id=call.call_id,
                status=ToolCallStatus.SUCCEEDED,
                completed_at=utc_now(),
            )
        if isinstance(raw_result, str):
            payload = None
            output = raw_result
        elif isinstance(raw_result, Mapping):
            payload = dict(raw_result)
            output = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        elif isinstance(raw_result, BaseModel):
            payload = raw_result.model_dump(mode="json")
            output = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
        else:
            raise BridgeDispatchError("local tool returned an unsupported result")
        bounded, truncated = _bound_text(output, limit)
        return ToolResult(
            call_id=call.call_id,
            status=ToolCallStatus.SUCCEEDED,
            result=None if truncated else payload,
            output=bounded,
            truncated=truncated,
            completed_at=utc_now(),
        )

    def _assert_argument_paths(
        self, arguments: Mapping[str, Any], scope_paths: tuple[str, ...]
    ) -> None:
        for key, value in arguments.items():
            if key == _UNIFIED_DIFF_ARGUMENT_KEY:
                self._assert_unified_diff_paths(value, scope_paths)
            elif key in _PATH_ARGUMENT_KEYS:
                values = value if isinstance(value, (list, tuple)) else (value,)
                for candidate in values:
                    if not isinstance(candidate, str):
                        raise BridgeDispatchError("Bridge path arguments must be strings")
                    relative = _relative_path(candidate, allow_empty=key in {"root", "path"})
                    if not _scope_covers(relative, scope_paths):
                        raise BridgeDispatchError("Bridge path argument is outside claim scope")
            elif isinstance(value, Mapping):
                self._assert_argument_paths(value, scope_paths)
            elif isinstance(value, (list, tuple)):
                for nested in value:
                    if isinstance(nested, Mapping):
                        self._assert_argument_paths(nested, scope_paths)

    @staticmethod
    def _assert_unified_diff_paths(
        value: Any, scope_paths: tuple[str, ...]
    ) -> None:
        if not isinstance(value, str) or not value:
            raise BridgeDispatchError("Bridge unified_diff must be a non-empty string")
        found_path = False
        for line in value.splitlines():
            if not (line.startswith("--- ") or line.startswith("+++ ")):
                continue
            raw_path = line[4:].split("\t", 1)[0]
            if " " in raw_path or not raw_path:
                raise BridgeDispatchError("Bridge unified_diff contains an invalid path")
            if raw_path == "/dev/null":
                continue
            if raw_path.startswith(("a/", "b/")):
                raw_path = raw_path[2:]
            relative = _relative_path(raw_path)
            if not _scope_covers(relative, scope_paths):
                raise BridgeDispatchError("Bridge unified_diff path is outside claim scope")
            found_path = True
        if not found_path:
            raise BridgeDispatchError("Bridge unified_diff is missing file headers")


def _required_identifier(claim: Mapping[str, Any], key: str, *, prefix: str) -> str:
    return _required_identifier_value(claim.get(key), key, prefix=prefix)


def _required_identifier_value(value: Any, key: str, *, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or not _ID_RE.fullmatch(value[len(prefix) :])
    ):
        raise BridgeDispatchError(f"Bridge claim {key} is invalid")
    return value


def _required_tool_name(value: Any) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", value):
        raise BridgeDispatchError("Bridge claim tool_name is invalid")
    return value


def _aware_timestamp(value: Any, key: str) -> datetime:
    if not isinstance(value, str):
        raise BridgeDispatchError(f"Bridge claim {key} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise BridgeDispatchError(f"Bridge claim {key} is invalid") from error
    if parsed.tzinfo is None:
        raise BridgeDispatchError(f"Bridge claim {key} must be timezone-aware")
    return parsed


def _scope_paths(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Mapping) or set(value) != {"paths"}:
        raise BridgeDispatchError("Bridge claim scope must contain only relative paths")
    paths = value.get("paths")
    if not isinstance(paths, Sequence) or isinstance(paths, (str, bytes)) or not paths:
        raise BridgeDispatchError("Bridge claim scope paths must be non-empty")
    normalized = tuple(_scope_path(path) for path in paths)
    if len(set(normalized)) != len(normalized):
        raise BridgeDispatchError("Bridge claim scope paths must be unique")
    return normalized


def _scope_path(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise BridgeDispatchError("Bridge scope path must be a non-empty string")
    if value == "**":
        return value
    if value.endswith("/**"):
        base = value[:-3]
        if not base:
            raise BridgeDispatchError("Bridge scope path must be relative")
    else:
        base = value
    _relative_path(base)
    if "*" in base:
        raise BridgeDispatchError("Bridge scope wildcards are only allowed as trailing /**")
    return value


def _relative_path(value: str, *, allow_empty: bool = False) -> str:
    if not value and allow_empty:
        return value
    if (
        not value
        or value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise BridgeDispatchError("Bridge path must use relative POSIX syntax")
    return value


def _scope_covers(path: str, scopes: tuple[str, ...]) -> bool:
    if path == "":
        return any(scope == "**" for scope in scopes)
    for scope in scopes:
        if scope == "**" or path == scope:
            return True
        if scope.endswith("/**") and path.startswith(scope[:-3] + "/"):
            return True
    return False


def _tool_call(plan: BridgeDispatchPlan) -> ToolCall:
    try:
        arguments = plan.arguments.model_dump(mode="json")
        return ToolCall(
            call_id=plan.call_id,
            run_id=plan.run_id,
            work_unit_id=plan.work_unit_id,
            tool_name=plan.tool_name,
            effect=plan.effect,
            arguments=arguments,
            status=ToolCallStatus.RUNNING,
            snapshot_id=plan.snapshot_id,
            requested_at=plan.requested_at,
        )
    except Exception as error:
        raise BridgeDispatchError("Bridge claim cannot form a local tool call") from error


def _failed_result(call: ToolCall, category: ErrorCategory, message: str) -> ToolResult:
    return ToolResult.from_call(
        call,
        status=ToolCallStatus.FAILED,
        error=ErrorInfo(category=category, message=message),
        completed_at=utc_now(),
    )


def _bound_text(value: str, limit: int) -> tuple[str, bool]:
    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return value, False
    return encoded[:limit].decode("utf-8", errors="ignore"), True


def _submission_payload(result: ToolResult, workspace_root: Path) -> dict[str, Any]:
    redacted_result = _redact_local_paths(result.result, workspace_root)
    redacted_output = _redact_local_paths(result.output, workspace_root)
    output, output_truncated = _bound_text(redacted_output, MAX_TOOL_OUTPUT_BYTES)
    payload: dict[str, Any] = {
        "status": result.status.value,
        "result": redacted_result,
        "output": output,
        "truncated": result.truncated or output_truncated,
        "changed_paths": list(result.changed_paths),
    }
    if result.exit_code is not None:
        payload["exit_code"] = result.exit_code
    if result.error is not None:
        payload["error"] = result.error.model_dump(mode="json")
    try:
        encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as error:
        raise BridgeDispatchError("local tool result is not JSON-safe") from error
    if len(encoded.encode("utf-8")) > MAX_TOOL_OUTPUT_BYTES:
        payload["result"] = None
        payload["output"], truncated = _bound_text(
            payload["output"], MAX_TOOL_OUTPUT_BYTES
        )
        payload["truncated"] = True or truncated
    return payload


def _redact_local_paths(value: Any, workspace_root: Path) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _redact_local_paths(nested, workspace_root)
            for key, nested in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact_local_paths(nested, workspace_root) for nested in value]
    if isinstance(value, str):
        root = os.fspath(workspace_root)
        text = value.replace(root, "<local-root>")
        if text.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:[\\/]", text):
            return "<local-path>"
        return re.sub(
            r"(?<![A-Za-z0-9_.-])/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+",
            "<local-path>",
            text,
        )
    return value
