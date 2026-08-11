"""CASCADE execution profile chain.

The cascade chain is an ordered, deduplicated list of WORKER-role model profiles
that the CASCADE strategy escalates through. The first profile is always the
base worker; subsequent profiles are escalation targets ordered from weakest to
strongest.

All functions here are pure: they accept profiles and return a chain or raise.
No IO, no adapter calls, no side effects.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, unique

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import ErrorCode, ProviderError
from prp_runtime.domain.models import ErrorCategory, ErrorInfo, VerificationResult
from prp_runtime.providers.base import ModelProfile

__all__ = [
    "CascadeChain",
    "CascadeDecision",
    "CascadeDisposition",
    "build_cascade_chain",
    "decide_cascade",
    "provider_failure_is_retryable",
]

CascadeChain = tuple[ModelProfile, ...]

_RETRYABLE_PROVIDER_CATEGORIES = frozenset(
    {
        ErrorCategory.TIMEOUT,
        ErrorCategory.RATE_LIMIT,
        ErrorCategory.NETWORK,
    }
)


@unique
class CascadeDisposition(StrEnum):
    """The three possible outcomes of one CASCADE attempt."""

    ACCEPT = "ACCEPT"
    ESCALATE = "ESCALATE"
    STOP = "STOP"


@dataclass(frozen=True, slots=True)
class CascadeDecision:
    """A pure CASCADE policy result with a stable audit rationale."""

    disposition: CascadeDisposition
    rationale: str


def provider_failure_is_retryable(error: ErrorInfo | None) -> bool:
    """Recover provider retryability from the Worker's stable error category."""
    return error is not None and error.category in _RETRYABLE_PROVIDER_CATEGORIES


def decide_cascade(
    *,
    has_next_profile: bool,
    verification_result: VerificationResult | None = None,
    provider_retryable: bool | None = None,
) -> CascadeDecision:
    """Decide whether one attempt is accepted, escalated, or terminal.

    Exactly one input signal is required. Provider errors reach the controller as
    recorded attempt facts, so their stable retryability is passed explicitly.
    No enum relies on truthiness: FAIL and INCONCLUSIVE are separate matrix rows.
    """
    if (verification_result is None) == (provider_retryable is None):
        raise ValueError(
            "exactly one of verification_result or provider_retryable is required"
        )

    if verification_result is VerificationResult.PASS:
        return CascadeDecision(
            CascadeDisposition.ACCEPT,
            "deterministic verification passed",
        )
    if verification_result is not None:
        if has_next_profile:
            return CascadeDecision(
                CascadeDisposition.ESCALATE,
                f"verification {verification_result.value} permits the next profile",
            )
        return CascadeDecision(
            CascadeDisposition.STOP,
            f"cascade chain exhausted after verification {verification_result.value}",
        )

    if provider_retryable and has_next_profile:
        return CascadeDecision(
            CascadeDisposition.ESCALATE,
            "retryable provider failure permits the next profile",
        )
    if provider_retryable:
        return CascadeDecision(
            CascadeDisposition.STOP,
            "cascade chain exhausted after a retryable provider failure",
        )
    return CascadeDecision(
        CascadeDisposition.STOP,
        "non-retryable provider failure stops cascade",
    )


def build_cascade_chain(
    base_worker: ModelProfile,
    escalation_profiles: tuple[ModelProfile, ...] = (),
) -> CascadeChain:
    """Build an ordered, deduplicated CASCADE chain starting from ``base_worker``.

    Rules:
    - ``base_worker`` must have role WORKER.
    - Every ``escalation_profiles`` entry must also have role WORKER.
    - No two entries may share the same alias.
    - An empty escalation list is allowed; the chain is then a single-entry tuple.

    Raises ``ProviderError(PROVIDER_NOT_CONFIGURED)`` for any violation so callers
    receive a structured error rather than a Python exception.
    """
    if base_worker.role is not ModelRole.WORKER:
        raise ProviderError(
            f"cascade base profile {base_worker.alias!r} must have role WORKER, "
            f"got {base_worker.role.value}",
            code=ErrorCode.PROVIDER_NOT_CONFIGURED,
        )

    chain: list[ModelProfile] = [base_worker]
    seen_aliases: set[str] = {base_worker.alias}

    for profile in escalation_profiles:
        if profile.role is not ModelRole.WORKER:
            raise ProviderError(
                f"cascade escalation profile {profile.alias!r} must have role WORKER, "
                f"got {profile.role.value}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        if profile.alias in seen_aliases:
            raise ProviderError(
                f"cascade chain contains duplicate alias {profile.alias!r}",
                code=ErrorCode.PROVIDER_NOT_CONFIGURED,
            )
        chain.append(profile)
        seen_aliases.add(profile.alias)

    return tuple(chain)
