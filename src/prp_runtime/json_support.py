"""Strict JSON parsing.

``strict_json_loads`` is the runtime's only entry point for turning JSON text
into Python values. No caller reaches for ``json.loads`` on its own, so what
counts as JSON here is stated once and cannot drift between the domain models
and the verifier.

The standard library is deliberately more permissive than the JSON grammar. It
accepts three constants the grammar does not define -- ``NaN``, ``Infinity`` and
``-Infinity`` -- and it silently overflows a literal such as ``1e999`` to
``float('inf')``. None of those values can be written back out as standard JSON,
so accepting one here would push the failure onto whatever reads the value next:
an artifact, a declared schema, or a provider payload. This module refuses them
at the door instead.
"""

import json
import math
from typing import Any, Final

__all__ = [
    "NON_STANDARD_JSON_CONSTANTS",
    "StrictJsonError",
    "canonical_json_dumps",
    "strict_json_loads",
]

#: The constants ``json`` accepts but the JSON grammar does not define.
NON_STANDARD_JSON_CONSTANTS: Final[frozenset[str]] = frozenset(
    {"NaN", "Infinity", "-Infinity"}
)


class StrictJsonError(ValueError):
    """The text is not standard JSON.

    A ``ValueError`` on purpose: a pydantic validator can let it surface as a
    field error without translating it. ``reason`` names what was wrong and
    ``token`` names the offending literal when there is one. Neither carries a
    stack trace, a path, or the full document.
    """

    def __init__(self, reason: str, *, token: str | None = None) -> None:
        self.reason = reason
        self.token = token
        super().__init__(reason)


def _reject_constant(name: str) -> Any:
    """Refuse ``NaN``, ``Infinity`` and ``-Infinity``."""
    raise StrictJsonError(f"{name} is not standard JSON and is not accepted", token=name)


def _reject_non_finite_float(text: str) -> float:
    """Refuse a numeric literal that would become a non finite float."""
    value = float(text)
    if not math.isfinite(value):
        raise StrictJsonError(
            f"the number {text} is not finite and is not accepted", token=text
        )
    return value


def strict_json_loads(text: str) -> Any:
    """Parse ``text`` as standard JSON.

    Everything the grammar defines -- ``null``, booleans, finite numbers,
    strings, arrays and objects -- is returned unchanged, with JSON integers
    still arriving as ``int`` and JSON fractions as ``float``.

    Raises ``StrictJsonError`` for malformed text, for the three non standard
    constants, and for any numeric literal that would overflow to infinity.
    """
    try:
        return json.loads(
            text,
            parse_constant=_reject_constant,
            parse_float=_reject_non_finite_float,
        )
    except StrictJsonError:
        raise
    except json.JSONDecodeError as error:
        raise StrictJsonError(f"invalid JSON: {error.msg}") from error


def canonical_json_dumps(value: Any) -> str:
    """Render JSON-compatible public facts deterministically."""
    return json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
