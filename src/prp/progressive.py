"""Small public entry point for the Progressive protocol.

This module intentionally contains no runtime, provider, storage, or transport
logic.  It gives implementations a stable name for the one protocol decision
PRP defines: whether a new graph version may be opened.
"""

from prp.revision import (
    ComparisonOutcome,
    RevisionDecision,
    decide_revision,
)

ProgressiveDecision = RevisionDecision


def decide_progressive(**kwargs: object) -> ProgressiveDecision:
    """Return the deterministic Progressive revision decision.

    Keyword arguments are the same public inputs accepted by
    :func:`prp.revision.decide_revision`.  Keeping this wrapper keyword-only
    prevents callers from depending on argument order while making the
    protocol's intent explicit at integration points.
    """

    return decide_revision(**kwargs)  # type: ignore[arg-type]


__all__ = ["ComparisonOutcome", "ProgressiveDecision", "decide_progressive"]
