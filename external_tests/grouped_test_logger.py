"""Grouped test logger for relaxed campaign runner."""

import json
from pathlib import Path
from datetime import datetime
from typing import Any


class GroupLogger:
    """Logger for test group execution results."""

    def __init__(self, log_dir: Path, group_name: str):
        self.log_dir = log_dir
        self.group_name = group_name
        self.log_file = log_dir / f"{group_name}.jsonl"
        self.summary_file = log_dir / f"{group_name}_summary.txt"

        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Clear previous logs
        if self.log_file.exists():
            self.log_file.unlink()
        if self.summary_file.exists():
            self.summary_file.unlink()

    def log_event(self, event_type: str, **data: Any) -> None:
        """Log a structured event."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            **data
        }
        with self.log_file.open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_scenario_start(self, scenario_id: str, description: str) -> None:
        """Log start of a test scenario."""
        self.log_event("scenario_start", scenario_id=scenario_id, description=description)

    def log_scenario_end(
        self,
        scenario_id: str,
        status: str,
        duration: float,
        stdout: str = "",
        stderr: str = ""
    ) -> None:
        """Log end of a test scenario."""
        self.log_event(
            "scenario_end",
            scenario_id=scenario_id,
            status=status,
            duration=duration,
            stdout_preview=stdout[:500] if stdout else "",
            stderr_preview=stderr[:500] if stderr else ""
        )

    def log_repair_attempt(self, scenario_id: str, reason: str) -> None:
        """Log a repair attempt."""
        self.log_event("repair_attempt", scenario_id=scenario_id, reason=reason)

    def log_repair_result(self, scenario_id: str, success: bool, details: str) -> None:
        """Log repair result."""
        self.log_event(
            "repair_result",
            scenario_id=scenario_id,
            success=success,
            details=details
        )

    def log_structured(self, data: dict[str, Any]) -> None:
        """Log structured scenario result data."""
        self.log_event("scenario_result", **data)

    def write_summary(self, summary: str) -> None:
        """Write human-readable summary to separate file."""
        self.summary_file.write_text(summary)

    def close(self) -> None:
        """Close logger (no-op, but provided for compatibility)."""
        pass
