"""Offline tests for the bounded and redacted external result ledger."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

_EXTERNAL_TESTS_DIR = Path(__file__).resolve().parents[3] / "external_tests"
if str(_EXTERNAL_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(_EXTERNAL_TESTS_DIR))

from result_ledger import LedgerEntry, LedgerError, LedgerStore  # noqa: E402


def _entry(scenario_id: str = "scenario-1") -> LedgerEntry:
    return LedgerEntry(
        scenario_id=scenario_id,
        alias="TEST_ALIAS",
        model_id="test-model",
        protocol="OPENAI_CHAT",
        endpoint_host="models.example",
        run_id=f"run-{scenario_id}",
        attempt_id=f"attempt-{scenario_id}",
        status="SUCCESS",
        actual_or_simulated="SIMULATED",
        input_tokens=3,
        output_tokens=5,
        known_cost="unknown",
        latency_ms=12,
        error_code=None,
        output_sha256="a" * 64,
        recorded_at="2026-08-18T00:00:00Z",
    )


def test_ledger_round_trip_and_duplicate_scenario_is_idempotent(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.json")
    entry = _entry()
    assert store.merge([entry]) == (entry,)
    assert store.merge([entry]) == (entry,)
    assert store.read() == (entry,)


def test_ledger_rejects_conflicting_scenario_evidence(tmp_path: Path) -> None:
    store = LedgerStore(tmp_path / "ledger.json")
    store.merge([_entry()])
    conflicting = _entry()
    object.__setattr__(conflicting, "status", "FAIL")
    with pytest.raises(LedgerError, match="conflicting"):
        store.merge([conflicting])


def test_ledger_merges_concurrently_with_atomic_replacement(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"

    def write(index: int) -> None:
        LedgerStore(path).merge([_entry(f"scenario-{index}")])

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(write, range(8)))
    entries = LedgerStore(path).read()
    assert len(entries) == 8
    assert [entry.scenario_id for entry in entries] == sorted(
        entry.scenario_id for entry in entries
    )


def test_ledger_rejects_corrupt_file_without_echoing_contents(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    path.write_text("{ raw-output", encoding="utf-8")
    with pytest.raises(LedgerError, match="malformed") as excinfo:
        LedgerStore(path).read()
    assert "raw-output" not in str(excinfo.value)


def test_ledger_rejects_size_overrun(tmp_path: Path) -> None:
    with pytest.raises(LedgerError, match="size"):
        LedgerStore(tmp_path / "ledger.json", max_bytes=32).merge([_entry()])


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output", "raw response text"),
        ("prompt", "private input"),
        ("headers", {"x-api-key": "secret-value"}),
        ("model_id", "/home/bruce/private-model"),
    ],
)
def test_ledger_rejects_raw_output_secret_fields_and_absolute_paths(
    tmp_path: Path, field: str, value: object
) -> None:
    record = _entry().to_dict()
    record[field] = value
    with pytest.raises(LedgerError):
        LedgerStore(tmp_path / "ledger.json").merge([record])


def test_ledger_rejects_explicit_secret_value_before_write(tmp_path: Path) -> None:
    record = _entry().to_dict()
    secret_value = "-".join(("fixture", "secret", "value"))
    record["error_code"] = secret_value
    with pytest.raises(LedgerError, match="secret"):
        LedgerStore(tmp_path / "ledger.json").merge(
            [record], secret_values=[secret_value]
        )


def test_ledger_document_is_small_ascii_and_has_no_raw_fields(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    LedgerStore(path).merge([_entry()])
    raw = path.read_bytes()
    assert len(raw) < 256 * 1024
    document = json.loads(raw)
    assert "output" not in document["entries"][0]
    assert "prompt" not in document["entries"][0]
