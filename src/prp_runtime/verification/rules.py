"""Deterministic verification rules.

Every rule here decides by inspecting the artifact itself. A rule that cannot
decide returns ``INCONCLUSIVE``; it never reports a pass it cannot prove. There
is no general purpose model-as-judge in this module.

JSON Schema constraints are decided by ``jsonschema``, not by hand. A hand
written checker has to be told about every keyword, and the keyword it has not
been told about is silently skipped -- which reads as a pass for a constraint
that was never applied. The keyword allowlist below still bounds what this
runtime claims to decide, but within that boundary the library does the work and
the Draft 2020-12 meta-schema decides whether the declared schema is even legal.

A schema outside the allowlist is ``INCONCLUSIVE``. A schema inside it that is
not a legal JSON Schema is a configuration error, not a verdict about the
artifact: blaming the output for a broken declaration would be a false failure
in the same way a skipped keyword is a false pass.
"""

from collections.abc import Sequence
from enum import StrEnum, unique
from functools import lru_cache
from typing import Final

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError
from pydantic import Field, model_validator

from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    ArtifactKind,
    DomainModel,
    Label,
    OutputRequirement,
    VerificationResult,
)
from prp_runtime.json_support import StrictJsonError, strict_json_loads

__all__ = [
    "SUPPORTED_SCHEMA_KEYWORDS",
    "VerificationCheck",
    "VerificationPlan",
    "VerificationRule",
    "check_json_schema",
    "compile_schema",
    "plan_for_output",
]


@unique
class VerificationRule(StrEnum):
    """The rules this version can decide deterministically."""

    NON_EMPTY_OUTPUT = "NON_EMPTY_OUTPUT"
    OUTPUT_KIND_MATCHES = "OUTPUT_KIND_MATCHES"
    VALID_JSON = "VALID_JSON"
    MATCHES_JSON_SCHEMA = "MATCHES_JSON_SCHEMA"
    REQUIRED_REFERENCES = "REQUIRED_REFERENCES"
    WITHIN_LENGTH_LIMIT = "WITHIN_LENGTH_LIMIT"


#: JSON Schema keywords the built-in checker understands. A schema using any
#: other keyword makes the check INCONCLUSIVE rather than falsely passing.
SUPPORTED_SCHEMA_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "enum",
        "const",
        "minItems",
        "maxItems",
        "minLength",
        "maxLength",
        "minimum",
        "maximum",
        "title",
        "description",
        "$schema",
        "$comment",
        "default",
        "examples",
    }
)

#: How many distinct declared schemas stay compiled. A run declares one schema per
#: output requirement, so a small cache covers every check a run repeats across
#: attempts and work units without growing without bound.
_SCHEMA_CACHE_SIZE: Final[int] = 128

#: The one parameter each rule may carry. A check that also carries a parameter
#: belonging to a different rule is an ambiguous declaration: the reader cannot
#: tell which constraint was meant to apply. Such a check is rejected rather than
#: accepted with the extra parameter silently ignored.
_RULE_PARAMETER: dict[VerificationRule, str | None] = {
    VerificationRule.NON_EMPTY_OUTPUT: None,
    VerificationRule.OUTPUT_KIND_MATCHES: "expected_kind",
    VerificationRule.VALID_JSON: None,
    VerificationRule.MATCHES_JSON_SCHEMA: "json_schema",
    VerificationRule.REQUIRED_REFERENCES: "required_references",
    VerificationRule.WITHIN_LENGTH_LIMIT: "max_characters",
}

#: Every parameter field, so the validator can name the ones that must be unset.
_PARAMETER_FIELDS: tuple[str, ...] = (
    "expected_kind",
    "json_schema",
    "required_references",
    "max_characters",
)


class VerificationCheck(DomainModel):
    """One rule with the parameters it needs."""

    rule: VerificationRule
    expected_kind: ArtifactKind | None = None
    json_schema: str | None = None
    required_references: tuple[str, ...] = ()
    max_characters: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _parameters_match_the_rule(self) -> "VerificationCheck":
        if self.rule is VerificationRule.OUTPUT_KIND_MATCHES and self.expected_kind is None:
            raise ValueError("OUTPUT_KIND_MATCHES requires expected_kind")
        if self.rule is VerificationRule.MATCHES_JSON_SCHEMA and self.json_schema is None:
            raise ValueError("MATCHES_JSON_SCHEMA requires json_schema")
        if self.rule is VerificationRule.REQUIRED_REFERENCES and not self.required_references:
            raise ValueError("REQUIRED_REFERENCES requires at least one reference")
        if self.rule is VerificationRule.WITHIN_LENGTH_LIMIT and self.max_characters is None:
            raise ValueError("WITHIN_LENGTH_LIMIT requires max_characters")

        owned = _RULE_PARAMETER[self.rule]
        for name in _PARAMETER_FIELDS:
            if name == owned:
                continue
            value = getattr(self, name)
            if value not in (None, ()):
                raise ValueError(f"{self.rule.value} does not use {name}")

        if self.json_schema is not None:
            # Confirming the declaration here means a legal-schema failure is
            # raised where the check is declared, not later where it would look
            # like a verdict about the artifact.
            try:
                compile_schema(self.json_schema)
            except DomainValidationError as error:
                raise ValueError(error.detail.message) from error
        if any(not reference.strip() for reference in self.required_references):
            raise ValueError("a required reference must not be blank")
        return self

    @property
    def label(self) -> Label:
        """The rule name recorded on the evidence row."""
        return self.rule.value


VerificationPlan = tuple[VerificationCheck, ...]


def plan_for_output(
    output: OutputRequirement,
    *,
    required_references: Sequence[str] = (),
    max_characters: int | None = None,
) -> VerificationPlan:
    """Build the deterministic plan implied by a work unit's output requirement.

    The plan is a pure function of the declared requirement, so the same unit
    always gets the same checks.
    """
    checks: list[VerificationCheck] = [
        VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT),
        VerificationCheck(
            rule=VerificationRule.OUTPUT_KIND_MATCHES, expected_kind=output.kind
        ),
    ]
    if output.kind is ArtifactKind.JSON:
        checks.append(VerificationCheck(rule=VerificationRule.VALID_JSON))
        if output.json_schema is not None:
            checks.append(
                VerificationCheck(
                    rule=VerificationRule.MATCHES_JSON_SCHEMA,
                    json_schema=output.json_schema,
                )
            )
    if required_references:
        checks.append(
            VerificationCheck(
                rule=VerificationRule.REQUIRED_REFERENCES,
                required_references=tuple(required_references),
            )
        )
    if max_characters is not None:
        checks.append(
            VerificationCheck(
                rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=max_characters
            )
        )
    return tuple(checks)


def _unsupported_keywords(schema: object, path: str = "$") -> list[str]:
    """Collect schema keywords the built-in checker cannot decide."""
    if not isinstance(schema, dict):
        return [path]
    unsupported = [
        f"{path}.{key}" for key in sorted(schema) if key not in SUPPORTED_SCHEMA_KEYWORDS
    ]
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, subschema in sorted(properties.items()):
            unsupported.extend(_unsupported_keywords(subschema, f"{path}.properties.{name}"))
    items = schema.get("items")
    if isinstance(items, dict):
        unsupported.extend(_unsupported_keywords(items, f"{path}.items"))
    elif isinstance(items, list):
        # Tuple-form "items" is positional validation, which this runtime does not
        # claim to decide. Any other illegal shape is left to the meta-schema, so
        # it is reported as a broken declaration instead of an undecided check.
        unsupported.append(f"{path}.items")
    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        unsupported.extend(_unsupported_keywords(additional, f"{path}.additionalProperties"))
    return unsupported


def _pointer(path: Sequence[object]) -> str:
    """Render an error location as ``$``, ``$.name`` or ``$[0]``."""
    rendered = "$"
    for part in path:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


@lru_cache(maxsize=_SCHEMA_CACHE_SIZE)
def compile_schema(
    json_schema: str,
) -> tuple[Draft202012Validator | None, tuple[str, ...]]:
    """Turn declared schema text into a validator, or say why there is none.

    Memoised on the schema text. ``VerificationCheck`` confirms the declaration at
    construction and ``check_json_schema`` needs the same validator per verify;
    without the cache the second call would re-parse the text, re-walk it for
    unsupported keywords and re-run the meta-schema on every single artifact. A
    compiled validator holds no per-document state, so one instance serves every
    check that declared the same schema. Illegal schemas raise instead of
    returning, and ``lru_cache`` does not memoise exceptions, so a broken
    declaration is re-reported rather than cached.

    Exactly one half of the result is meaningful. When the schema stays inside the
    supported keyword subset a validator is returned and the second half is empty.
    When it does not, the validator is ``None`` and the second half names the
    keywords that put it out of range: no verdict about a document is possible,
    and the meta-schema is deliberately not consulted, because a keyword this
    runtime does not decide should read as undecided rather than as illegal.

    Raises ``DomainValidationError`` with ``INVALID_OUTPUT_REQUIREMENT`` when the
    text is not standard JSON, is not a JSON object, or -- inside the supported
    subset -- is not a legal Draft 2020-12 schema. A miswritten keyword such as
    ``minLength: "5"`` or ``properties: []`` is caught here, so it can never reach
    the document comparison and be reported as a verdict about the artifact.

    The dialect is always Draft 2020-12 whatever ``$schema`` says, so it is a
    property of this runtime rather than of the declaration. Across the supported
    subset the earlier drafts agree with 2020-12, except for tuple-form ``items``,
    which is reported as undecidable.
    """
    try:
        schema = strict_json_loads(json_schema)
    except StrictJsonError as error:
        raise DomainValidationError(
            f"the declared json_schema is not valid JSON: {error.reason}",
            code=ErrorCode.INVALID_OUTPUT_REQUIREMENT,
            field="json_schema",
        ) from error
    if not isinstance(schema, dict):
        raise DomainValidationError(
            "the declared json_schema must be a JSON object",
            code=ErrorCode.INVALID_OUTPUT_REQUIREMENT,
            field="json_schema",
        )

    unsupported = _unsupported_keywords(schema)
    if unsupported:
        return None, tuple(unsupported)

    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as error:
        raise DomainValidationError(
            f"the declared json_schema is not a legal JSON Schema: {error.message}",
            code=ErrorCode.INVALID_OUTPUT_REQUIREMENT,
            field="json_schema",
        ) from error
    return Draft202012Validator(schema), ()


def check_json_schema(content: str, json_schema: str) -> tuple[VerificationResult, str]:
    """Check one document against one declared schema.

    ``FAIL`` when the output is not standard JSON, or when it contradicts a legal
    schema. ``INCONCLUSIVE`` when the schema uses a keyword outside the supported
    subset, so a constraint that was never applied is not reported as satisfied.
    An illegal declaration raises instead of returning a verdict.

    The output is parsed before the undecidable case is reported: content that is
    not JSON at all provably fails any schema, so that much is decided even when
    the schema itself is out of range.
    """
    validator, undecidable = compile_schema(json_schema)

    try:
        document = strict_json_loads(content)
    except StrictJsonError as error:
        return VerificationResult.FAIL, f"the output is not valid JSON: {error.reason}"

    if validator is None:
        return (
            VerificationResult.INCONCLUSIVE,
            "the schema uses keywords this checker cannot decide: "
            + ", ".join(undecidable[:5]),
        )

    # Sorted by rendered location so the reported reasons are stable between runs.
    errors = sorted(
        validator.iter_errors(document),
        key=lambda error: (_pointer(error.absolute_path), error.message),
    )
    if errors:
        return VerificationResult.FAIL, "; ".join(
            f"{_pointer(error.absolute_path)} {error.message}" for error in errors[:5]
        )
    return VerificationResult.PASS, "the output satisfies the declared schema"
