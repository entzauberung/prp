"""Small redacted ledger for real Provider capability probes."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Any

MAX_CAPABILITY_LEDGER_BYTES = 128 * 1024
_LOCKS: dict[Path, RLock] = {}
_LOCKS_GUARD = RLock()
_STATUSES = frozenset({"PASS", "UPSTREAM_UNSUPPORTED", "UPSTREAM_ERROR"})


@dataclass(frozen=True, slots=True)
class CapabilityEntry:
    scenario_id: str
    alias: str
    model_id: str
    protocol: str
    endpoint_host: str
    capability: str
    status: str
    actual_or_simulated: str
    attempt_id: str
    provider_request_id: str | None
    finish_reason: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_ms: int | None
    error_code: str | None
    output_sha256: str | None
    recorded_at: str
    diagnostic: str | None = None

    def to_dict(self) -> dict[str, Any]:
        if self.status not in _STATUSES:
            raise ValueError("unsupported capability status")
        return asdict(self)


class CapabilityStore:
    """Atomically merge capability facts by scenario id."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = _lock_for(path)

    def read(self) -> tuple[CapabilityEntry, ...]:
        with self._lock:
            if not self._path.exists():
                return ()
            try:
                raw = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as error:
                raise ValueError("capability ledger is unreadable or malformed") from error
            if (
                not isinstance(raw, dict)
                or raw.get("schema_version") != "0.0.2"
                or not isinstance(raw.get("entries"), list)
            ):
                raise ValueError("capability ledger schema is malformed")
            return tuple(CapabilityEntry(**item) for item in raw["entries"])

    def merge(self, entries: tuple[CapabilityEntry, ...] = ()) -> tuple[CapabilityEntry, ...]:
        with self._lock:
            existing = {entry.scenario_id: entry for entry in self.read()}
            for entry in entries:
                prior = existing.get(entry.scenario_id)
                if prior is not None and prior != entry:
                    raise ValueError("capability scenario has conflicting evidence")
                existing[entry.scenario_id] = entry
            merged = tuple(existing[key] for key in sorted(existing))
            document = {
                "schema_version": "0.0.2",
                "entries": [entry.to_dict() for entry in merged],
            }
            encoded = json.dumps(
                document, ensure_ascii=True, separators=(",", ":"), sort_keys=True
            ).encode("utf-8")
            if len(encoded) > MAX_CAPABILITY_LEDGER_BYTES:
                raise ValueError("capability ledger exceeds size limit")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{self._path.name}.", suffix=".tmp", dir=self._path.parent
            )
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temporary_name, self._path)
            except OSError as error:
                try:
                    os.unlink(temporary_name)
                except OSError:
                    pass
                raise ValueError("capability ledger write failed") from error
            return merged


def _lock_for(path: Path) -> RLock:
    resolved = path.resolve()
    with _LOCKS_GUARD:
        lock = _LOCKS.get(resolved)
        if lock is None:
            lock = RLock()
            _LOCKS[resolved] = lock
        return lock
