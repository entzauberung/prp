"""Bounded, classified logging for relaxed grouped test campaigns."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Final

MAX_LOG_FILE_BYTES: Final = 2 * 1024 * 1024  # 2 MiB
MAX_LOG_DIR_BYTES: Final = 16 * 1024 * 1024  # 16 MiB

RESULT_CLASSIFICATIONS: Final = frozenset(
    {
        "PASS",
        "PRODUCT_DEFECT",
        "UPSTREAM_UNSUPPORTED",
        "UPSTREAM_AUTH_OR_BALANCE",
        "UPSTREAM_TRANSIENT",
        "ENVIRONMENT_LIMITATION",
        "NOT_APPLICABLE",
        "BUDGET_NOT_RUN",
        "LOG_TRUNCATED",
    }
)


@dataclass(frozen=True)
class ScenarioResult:
    """Structured result for one test scenario."""

    scenario: str
    classification: str
    exit_code: int
    duration_seconds: float
    error_summary: str = ""

    def __post_init__(self) -> None:
        if self.classification not in RESULT_CLASSIFICATIONS:
            raise ValueError(f"invalid classification: {self.classification}")


class GroupLogger:
    """Bounded logger for one test group with command and structured outputs."""

    def __init__(self, log_dir: Path, group_name: str) -> None:
        self.log_dir = Path(log_dir)
        self.group_name = group_name
        self.command_log_path = self.log_dir / f"{group_name}.log"
        self.structured_log_path = self.log_dir / f"{group_name}.jsonl"
        self._command_bytes = 0
        self._structured_bytes = 0
        self._truncated = False

    def write_command_output(self, text: str) -> None:
        """Append command output to .log file with size limit."""
        if self._truncated:
            return

        text_bytes = text.encode("utf-8")
        if self._command_bytes + len(text_bytes) > MAX_LOG_FILE_BYTES:
            remaining = MAX_LOG_FILE_BYTES - self._command_bytes
            if remaining > 100:
                truncation_marker = "\n... [LOG TRUNCATED] ...\n"
                write_bytes = text_bytes[:remaining - len(truncation_marker)]
                with open(self.command_log_path, "ab") as f:
                    f.write(write_bytes)
                    f.write(truncation_marker.encode("utf-8"))
            self._truncated = True
            return

        with open(self.command_log_path, "ab") as f:
            f.write(text_bytes)
        self._command_bytes += len(text_bytes)

    def write_scenario_result(self, result: ScenarioResult) -> None:
        """Append structured result to .jsonl file with size limit."""
        if self._truncated:
            return

        line = json.dumps(asdict(result), ensure_ascii=False) + "\n"
        line_bytes = line.encode("utf-8")

        if self._structured_bytes + len(line_bytes) > MAX_LOG_FILE_BYTES:
            self._truncated = True
            truncation_result = ScenarioResult(
                scenario="LOG_TRUNCATED",
                classification="LOG_TRUNCATED",
                exit_code=0,
                duration_seconds=0.0,
                error_summary="Structured log exceeded size limit",
            )
            truncation_line = json.dumps(asdict(truncation_result), ensure_ascii=False) + "\n"
            with open(self.structured_log_path, "ab") as f:
                f.write(truncation_line.encode("utf-8"))
            return

        with open(self.structured_log_path, "ab") as f:
            f.write(line_bytes)
        self._structured_bytes += len(line_bytes)

    def get_total_bytes(self) -> int:
        """Return total bytes written to both log files."""
        return self._command_bytes + self._structured_bytes

    def is_truncated(self) -> bool:
        """Return True if logging was truncated due to size limits."""
        return self._truncated


def ensure_log_directory(log_dir: Path) -> None:
    """Create log directory if it does not exist."""
    log_dir.mkdir(parents=True, exist_ok=True)
    if not log_dir.is_dir():
        raise OSError(f"log directory is not accessible: {log_dir}")


def check_directory_size(log_dir: Path) -> int:
    """Return total size of all files in log directory."""
    total = 0
    for entry in os.scandir(log_dir):
        if entry.is_file(follow_symlinks=False):
            total += entry.stat().st_size
    return total


__all__ = [
    "GroupLogger",
    "ScenarioResult",
    "RESULT_CLASSIFICATIONS",
    "MAX_LOG_FILE_BYTES",
    "MAX_LOG_DIR_BYTES",
    "ensure_log_directory",
    "check_directory_size",
]
