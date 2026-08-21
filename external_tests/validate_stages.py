"""Validate that STAGE_REGISTRY aligns with actual pytest markers and test files."""

from __future__ import annotations

import sys
from pathlib import Path

# Import the registry
from unattended_runner import STAGE_REGISTRY

# Expected structure
EXPECTED_STAGES = {
    "preflight": {"is_collection": True, "marker": None},
    "provider": {"is_collection": False, "marker": "live_provider"},
    "integration": {"is_collection": False, "marker": "live_integration"},
    "strategy": {"is_collection": False, "marker": "live_strategy"},
    "agent": {"is_collection": False, "marker": "live_agent"},
    "regression": {"is_collection": False, "marker": "live_regression"},
}


def validate_registry() -> list[str]:
    """Check that STAGE_REGISTRY matches expectations."""
    issues = []

    # Check all expected stages exist
    for stage_name, expected in EXPECTED_STAGES.items():
        if stage_name not in STAGE_REGISTRY:
            issues.append(f"Missing stage: {stage_name}")
            continue

        spec = STAGE_REGISTRY[stage_name]

        # Validate pytest args contain the marker (except preflight)
        if expected["marker"] is not None:
            if "-m" not in spec.pytest_args:
                issues.append(f"{stage_name}: missing -m flag")
            elif expected["marker"] not in spec.pytest_args:
                issues.append(f"{stage_name}: missing marker {expected['marker']}")

    # Check for unexpected stages
    unexpected = set(STAGE_REGISTRY) - set(EXPECTED_STAGES)
    if unexpected:
        issues.append(f"Unexpected stages: {unexpected}")

    return issues


def main() -> int:
    """Run validation and report."""
    issues = validate_registry()

    if issues:
        print("❌ Stage registry validation failed:")
        for issue in issues:
            print(f"  • {issue}")
        return 1

    print("✅ Stage registry validation passed")
    print(f"   Registered stages: {', '.join(sorted(STAGE_REGISTRY.keys()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
