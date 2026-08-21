"""Bounded, redacted, and atomically persisted external-validation results."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

MAX_LEDGER_BYTES = 256 * 1024
LedgerMode = Literal["ACTUAL", "SIMULATED"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COST = re.compile(r"^(?:unknown|[0-9]+(?:\.[0-9]+)?)$")
_PATH_VALUE = re.compile(r"^(?:[A-Za-z]:[\\/]|/)")
_SECRET_PATTERN = re.compile(r"(?i)\b(?:bearer|basic)\s+[a-z0-9._~+/=-]{8,}")
_FORBIDDEN_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "body",
    "credential",
    "headers",
    "input",
    "output",
    "prompt",
    "raw_output",
    "request_body",
    "response",
    "response_body",
    "secret",
}
_PATH_LOCKS: dict[Path, RLock] = {}
_PATH_LOCKS_GUARD = RLock()


class LedgerError(RuntimeError):
    """A safe ledger validation or persistence failure."""


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """The complete redacted record allowed in LIVE-VALIDATION.json."""

    scenario_id: str
    alias: str
    model_id: str
    protocol: str
    endpoint_host: str
    run_id: str
    attempt_id: str
    status: str
    actual_or_simulated: LedgerMode
    input_tokens: int | None
    output_tokens: int | None
    known_cost: str
    latency_ms: int | None
    error_code: str | None
    output_sha256: str | None
    recorded_at: str

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "scenario_id": self.scenario_id,
            "alias": self.alias,
            "model_id": self.model_id,
            "protocol": self.protocol,
            "endpoint_host": self.endpoint_host,
            "run_id": self.run_id,
            "attempt_id": self.attempt_id,
            "status": self.status,
            "actual_or_simulated": self.actual_or_simulated,
            "usage": {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
            },
            "known_cost": self.known_cost,
            "latency_ms": self.latency_ms,
            "error_code": self.error_code,
            "output_sha256": self.output_sha256,
            "recorded_at": self.recorded_at,
        }
        scan_safe_payload(value)
        return value

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> LedgerEntry:
        scan_safe_payload(raw)
        required = {
            "scenario_id",
            "alias",
            "model_id",
            "protocol",
            "endpoint_host",
            "run_id",
            "attempt_id",
            "status",
            "actual_or_simulated",
            "usage",
            "known_cost",
            "latency_ms",
            "error_code",
            "output_sha256",
            "recorded_at",
        }
        if set(raw) != required:
            raise LedgerError("ledger entry contains an unsupported or missing field")
        usage = raw["usage"]
        if not isinstance(usage, dict) or set(usage) != {"input_tokens", "output_tokens"}:
            raise LedgerError("ledger usage is malformed")
        input_tokens = _optional_nonnegative_int(usage["input_tokens"], "input_tokens")
        output_tokens = _optional_nonnegative_int(usage["output_tokens"], "output_tokens")
        actual_or_simulated = raw["actual_or_simulated"]
        if actual_or_simulated not in {"ACTUAL", "SIMULATED"}:
            raise LedgerError("ledger execution mode is malformed")
        known_cost = raw["known_cost"]
        if not isinstance(known_cost, str) or not _COST.fullmatch(known_cost):
            raise LedgerError("ledger known_cost must be a decimal string or unknown")
        output_sha256 = raw["output_sha256"]
        if output_sha256 is not None and (
            not isinstance(output_sha256, str) or not _SHA256.fullmatch(output_sha256)
        ):
            raise LedgerError("ledger output hash is malformed")
        latency_ms = _optional_nonnegative_int(raw["latency_ms"], "latency_ms")
        error_code = raw["error_code"]
        if error_code is not None and (
            not isinstance(error_code, str) or not error_code.strip()
        ):
            raise LedgerError("ledger error code is malformed")
        text_fields = (
            "scenario_id",
            "alias",
            "model_id",
            "protocol",
            "endpoint_host",
            "run_id",
            "attempt_id",
            "status",
            "recorded_at",
        )
        for field_name in text_fields:
            if not isinstance(raw[field_name], str) or not raw[field_name].strip():
                raise LedgerError(f"ledger {field_name} is malformed")
        endpoint_host = cast(str, raw["endpoint_host"])
        if "/" in endpoint_host or ":" in endpoint_host or "?" in endpoint_host:
            raise LedgerError("ledger endpoint_host must contain a host only")
        return cls(
            scenario_id=cast(str, raw["scenario_id"]),
            alias=cast(str, raw["alias"]),
            model_id=cast(str, raw["model_id"]),
            protocol=cast(str, raw["protocol"]),
            endpoint_host=endpoint_host,
            run_id=cast(str, raw["run_id"]),
            attempt_id=cast(str, raw["attempt_id"]),
            status=cast(str, raw["status"]),
            actual_or_simulated=cast(LedgerMode, actual_or_simulated),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            known_cost=known_cost,
            latency_ms=latency_ms,
            error_code=error_code,
            output_sha256=output_sha256,
            recorded_at=cast(str, raw["recorded_at"]),
        )


class LedgerStore:
    """Round-trip and atomically merge a bounded ledger file."""

    def __init__(self, path: Path, *, max_bytes: int = MAX_LEDGER_BYTES) -> None:
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self._path = path
        self._max_bytes = max_bytes
        self._lock = _lock_for(path)

    def read(self) -> tuple[LedgerEntry, ...]:
        with self._lock:
            if not self._path.exists():
                return ()
            try:
                encoded = self._path.read_bytes()
                if len(encoded) > self._max_bytes:
                    raise LedgerError("ledger exceeds size limit")
                raw: Any = json.loads(encoded.decode("utf-8"))
            except LedgerError:
                raise
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                raise LedgerError("ledger file is unreadable or malformed") from exc
            return _decode_document(raw)

    def merge(
        self,
        entries: Iterable[LedgerEntry | Mapping[str, Any]],
        *,
        secret_values: Sequence[str] = (),
    ) -> tuple[LedgerEntry, ...]:
        """Merge by scenario id, allowing exact duplicates but rejecting conflicts."""

        incoming = tuple(_coerce_entry(entry) for entry in entries)
        for entry in incoming:
            scan_safe_payload(entry.to_dict(), secret_values=secret_values)
        with self._lock:
            existing = {entry.scenario_id: entry for entry in self.read()}
            for entry in incoming:
                prior = existing.get(entry.scenario_id)
                if prior is not None and prior != entry:
                    raise LedgerError("scenario id has conflicting ledger evidence")
                existing[entry.scenario_id] = entry
            merged = tuple(existing[key] for key in sorted(existing))
            self._atomic_write(merged, secret_values=secret_values)
            return merged

    def _atomic_write(
        self,
        entries: Iterable[LedgerEntry],
        *,
        secret_values: Sequence[str],
    ) -> None:
        document = {
            "schema_version": "0.0.2",
            "entries": [entry.to_dict() for entry in entries],
        }
        scan_safe_payload(document, secret_values=secret_values)
        encoded = json.dumps(
            document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        if len(encoded) > self._max_bytes:
            raise LedgerError("ledger exceeds size limit")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent
        )
        try:
            with os.fdopen(fd, "wb") as temporary:
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, self._path)
        except OSError as exc:
            try:
                os.unlink(temporary_name)
            except OSError:
                pass
            raise LedgerError("ledger atomic write failed") from exc


def scan_safe_payload(
    value: Any,
    *,
    secret_values: Sequence[str] = (),
) -> None:
    """Reject sensitive keys, secret values, raw content, and absolute paths."""

    secrets = tuple(secret for secret in secret_values if secret)

    def visit(item: Any) -> None:
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise LedgerError("ledger keys must be strings")
                normalized = key.lower().replace("-", "_")
                if normalized in _FORBIDDEN_KEYS or normalized.endswith("_body"):
                    raise LedgerError("ledger contains a forbidden sensitive field")
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
        elif isinstance(item, str):
            if any(secret in item for secret in secrets) or _SECRET_PATTERN.search(item):
                raise LedgerError("ledger contains a secret-like value")
            if _PATH_VALUE.match(item):
                raise LedgerError("ledger contains an absolute path")

    visit(value)


def _coerce_entry(entry: LedgerEntry | Mapping[str, Any]) -> LedgerEntry:
    if isinstance(entry, LedgerEntry):
        return entry
    if isinstance(entry, Mapping):
        return LedgerEntry.from_dict(entry)
    raise LedgerError("ledger entry has an unsupported type")


def _decode_document(raw: Any) -> tuple[LedgerEntry, ...]:
    if not isinstance(raw, dict) or set(raw) != {"schema_version", "entries"}:
        raise LedgerError("ledger document schema is malformed")
    if raw["schema_version"] != "0.0.2" or not isinstance(raw["entries"], list):
        raise LedgerError("ledger document schema is malformed")
    entries = tuple(LedgerEntry.from_dict(item) for item in raw["entries"])
    scenario_ids = [entry.scenario_id for entry in entries]
    if len(scenario_ids) != len(set(scenario_ids)):
        raise LedgerError("ledger contains duplicate scenario ids")
    return entries


def _optional_nonnegative_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LedgerError(f"ledger {field_name} must be a non-negative integer or null")
    return value


def _lock_for(path: Path) -> RLock:
    resolved = path.resolve()
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(resolved)
        if lock is None:
            lock = RLock()
            _PATH_LOCKS[resolved] = lock
        return lock
