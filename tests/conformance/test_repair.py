"""Conformance tests for the verification integrity repair.

These are the invariants the repair exists to establish, written against the
public surface rather than against any one module's internals. A unit test proves
a function behaves; these prove the runtime cannot be talked into claiming a
constraint held when it did not.

Each section states the defect it closes, so a reviewer can check the claim
against the behaviour instead of against a report.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime import json_support as json_support_module
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    Evidence,
    EvidenceKind,
    OutputRequirement,
    VerificationResult,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import new_attempt_id, new_run_id, new_work_unit_id
from prp_runtime.json_support import StrictJsonError, strict_json_loads
from prp_runtime.storage.sqlite import SCHEMA_VERSION
from prp_runtime.verification.rules import (
    VerificationCheck,
    VerificationRule,
    check_json_schema,
    compile_schema,
)
from prp_runtime.verification.verifier import RuleVerifier

RUN_ID = new_run_id()
WORK_UNIT_ID = new_work_unit_id()
ATTEMPT_ID = new_attempt_id()

NON_STANDARD_CONSTANTS = ["NaN", "Infinity", "-Infinity"]


def artifact(content: str, kind: ArtifactKind = ArtifactKind.TEXT) -> Artifact:
    return Artifact(
        artifact_id=new_artifact_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        attempt_id=ATTEMPT_ID,
        name="answer",
        kind=kind,
        content=content,
    )


def verdict(content: str, check: VerificationCheck) -> VerificationResult:
    report = RuleVerifier().verify(artifact(content), (check,))
    assert len(report.outcomes) == 1
    return report.outcomes[0].result


def schema_check(schema: object) -> VerificationCheck:
    return VerificationCheck(
        rule=VerificationRule.MATCHES_JSON_SCHEMA, json_schema=json.dumps(schema)
    )


# --- defect 1: non standard JSON was accepted -----------------------------------


@pytest.mark.parametrize("constant", NON_STANDARD_CONSTANTS)
def test_a_non_standard_constant_cannot_become_a_json_artifact(constant: str) -> None:
    with pytest.raises(ValidationError):
        artifact(f'{{"v": {constant}}}', ArtifactKind.JSON)


@pytest.mark.parametrize("constant", NON_STANDARD_CONSTANTS)
def test_a_non_standard_constant_cannot_pass_valid_json(constant: str) -> None:
    check = VerificationCheck(rule=VerificationRule.VALID_JSON)
    assert verdict(f"[{constant}]", check) is VerificationResult.FAIL


@pytest.mark.parametrize("constant", NON_STANDARD_CONSTANTS)
def test_a_non_standard_constant_cannot_be_declared_in_a_schema(constant: str) -> None:
    with pytest.raises(ValidationError):
        OutputRequirement(kind=ArtifactKind.JSON, json_schema=f'{{"const": {constant}}}')


def test_a_literal_that_would_overflow_to_infinity_is_refused() -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads("1e999")
    with pytest.raises(ValidationError):
        artifact('{"v": 1e999}', ArtifactKind.JSON)


def test_there_is_one_json_parser_and_it_is_the_strict_one() -> None:
    assert strict_json_loads('{"a": [1, 2.5, null, true, "s"]}') == {
        "a": [1, 2.5, None, True, "s"]
    }


# --- defect 2: an illegal schema parameter could report a pass ------------------

#: Keywords inside the supported subset carrying the wrong kind of value. A
#: checker that applies a keyword only when its value has the expected shape skips
#: these, and a skipped constraint is indistinguishable from a satisfied one.
ILLEGAL_SCHEMA_PARAMETERS: list[dict[str, object]] = [
    {"minLength": "5"},
    {"maxLength": 1.5},
    {"minimum": "0"},
    {"maxItems": -1},
    {"enum": "x"},
    {"required": "title"},
    {"properties": []},
    {"additionalProperties": 5},
    {"type": "objekt"},
    {"items": 5},
]


@pytest.mark.parametrize("schema", ILLEGAL_SCHEMA_PARAMETERS)
def test_an_illegal_schema_parameter_is_a_configuration_error(
    schema: dict[str, object],
) -> None:
    with pytest.raises(DomainValidationError) as raised:
        check_json_schema('{"title": "t"}', json.dumps(schema))
    assert raised.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT


@pytest.mark.parametrize("schema", ILLEGAL_SCHEMA_PARAMETERS)
def test_an_illegal_schema_parameter_cannot_even_be_declared(
    schema: dict[str, object],
) -> None:
    # No verify path exists that could reach a verdict about an unapplicable schema.
    with pytest.raises(ValidationError):
        schema_check(schema)


# --- defect 3: an unapplied keyword read as satisfied ---------------------------


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "oneOf": [{"type": "object"}]},
        {"type": "object", "allOf": [{"required": ["a"]}]},
        {"type": "object", "properties": {"k": {"$ref": "#/$defs/x"}}},
        {"type": "object", "properties": {"k": {"pattern": "^a"}}},
        {"type": "object", "patternProperties": {"^a": {"type": "string"}}},
        {"type": "array", "items": [{"type": "string"}]},
    ],
)
def test_a_keyword_outside_the_subset_is_undecided_never_a_pass(
    schema: dict[str, object],
) -> None:
    assert verdict('{"k": "v"}', schema_check(schema)) is VerificationResult.INCONCLUSIVE


def test_the_three_verdicts_stay_distinguishable() -> None:
    document = '{"k": "v"}'
    assert (
        verdict(document, schema_check({"type": "object", "required": ["k"]}))
        is VerificationResult.PASS
    )
    assert (
        verdict(document, schema_check({"type": "object", "required": ["absent"]}))
        is VerificationResult.FAIL
    )
    assert (
        verdict(document, schema_check({"type": "object", "oneOf": [{"type": "object"}]}))
        is VerificationResult.INCONCLUSIVE
    )


# --- defect 4: python equality stood in for json equality ----------------------


@pytest.mark.parametrize(
    ("schema", "document"),
    [
        ({"enum": [1, 2]}, "true"),
        ({"enum": [0]}, "false"),
        ({"const": 1}, "true"),
        ({"const": 0}, "false"),
        ({"const": True}, "1"),
        ({"enum": [True]}, "1"),
    ],
)
def test_a_boolean_is_never_a_number(schema: dict[str, object], document: str) -> None:
    assert verdict(document, schema_check(schema)) is VerificationResult.FAIL


@pytest.mark.parametrize(
    ("schema", "document"),
    [
        ({"const": True}, "true"),
        ({"const": 1}, "1.0"),
        ({"enum": [2]}, "2.0"),
        ({"enum": [1, 2]}, "1"),
    ],
)
def test_what_json_calls_equal_still_passes(
    schema: dict[str, object], document: str
) -> None:
    assert verdict(document, schema_check(schema)) is VerificationResult.PASS


# --- defect 5: evidence could record a verdict it never reached -----------------


def _evidence(**overrides: object) -> Evidence:
    data: dict[str, object] = {
        "evidence_id": new_evidence_id(),
        "run_id": RUN_ID,
        "work_unit_id": WORK_UNIT_ID,
        "artifact_id": new_artifact_id(),
        "kind": EvidenceKind.DETERMINISTIC_CHECK,
        "rule": VerificationRule.VALID_JSON.value,
        "result": VerificationResult.PASS,
        "detail": "a detail",
    }
    data.update(overrides)
    return Evidence(**data)  # type: ignore[arg-type]


def test_the_boolean_is_derived_and_cannot_be_supplied() -> None:
    assert "passed" not in Evidence.model_fields
    with pytest.raises(ValidationError):
        _evidence(passed=True)


@pytest.mark.parametrize("result", list(VerificationResult))
def test_the_boolean_always_follows_the_verdict(result: VerificationResult) -> None:
    assert _evidence(result=result).passed is result.is_pass


def test_an_undecided_verdict_is_not_a_proven_failure() -> None:
    undecided = _evidence(result=VerificationResult.INCONCLUSIVE)
    failed = _evidence(result=VerificationResult.FAIL)
    assert undecided.passed == failed.passed
    assert undecided.result is not failed.result


def test_a_verdict_is_mandatory() -> None:
    with pytest.raises(ValidationError):
        _evidence(result=None)


def test_a_deterministic_check_must_name_its_rule() -> None:
    with pytest.raises(ValidationError):
        _evidence(rule=None)
    assert _evidence(kind=EvidenceKind.MODEL_REVIEW, rule=None).rule is None


def test_every_row_the_verifier_writes_is_a_complete_verdict() -> None:
    report = RuleVerifier().verify(
        artifact('{"k": "v"}', ArtifactKind.JSON),
        (
            VerificationCheck(rule=VerificationRule.VALID_JSON),
            VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=1),
            schema_check({"type": "object", "oneOf": [{"type": "object"}]}),
        ),
    )
    rows = report.to_evidence()
    assert {row.result for row in rows} == set(VerificationResult)
    for row in rows:
        assert row.rule is not None
        assert row.passed is row.result.is_pass


# --- defect 6: the store kept a boolean beside the verdict ----------------------


def test_the_schema_version_is_the_repaired_one() -> None:
    assert SCHEMA_VERSION == 3


def test_the_stored_shape_has_no_boolean_column() -> None:
    from prp_runtime.storage.sqlite import SCHEMA_PATH

    schema_sql = SCHEMA_PATH.read_text(encoding="utf-8")
    evidence_table = schema_sql.split("CREATE TABLE IF NOT EXISTS evidence")[1]
    evidence_table = evidence_table.split(");")[0]
    assert "passed" not in evidence_table
    assert "result       TEXT    NOT NULL" in evidence_table


# --- defect 7: the schema was compiled again on every verify --------------------


def test_the_same_schema_text_compiles_once() -> None:
    text = json.dumps({"type": "object", "required": ["k"]})
    first, _ = compile_schema(text)
    second, _ = compile_schema(text)
    # One compiled validator holds no per-document state, so every check that
    # declared this schema can share it.
    assert first is second


def test_declaring_a_check_then_verifying_does_not_recompile() -> None:
    """Compilation belongs to declaring a check, not to judging each artifact."""
    text = json.dumps({"type": "object", "required": ["only_here"]})
    compile_schema.cache_clear()

    check = VerificationCheck(
        rule=VerificationRule.MATCHES_JSON_SCHEMA, json_schema=text
    )
    assert compile_schema.cache_info().misses == 1

    for _ in range(3):
        RuleVerifier().verify(
            artifact('{"only_here": 1}', ArtifactKind.JSON), (check,)
        )

    info = compile_schema.cache_info()
    assert info.misses == 1
    assert info.hits >= 3


# --- defect 8: the strict parser was not actually the only entry point ---------


def test_json_loads_appears_only_inside_the_strict_helper() -> None:
    """The single-entry claim is checked against the source, not asserted."""
    package = Path(json_support_module.__file__).parent
    offenders = sorted(
        path.relative_to(package).as_posix()
        for path in package.rglob("*.py")
        if path.name != "json_support.py"
        and "json.loads" in path.read_text(encoding="utf-8")
    )
    assert offenders == []


def test_the_ledger_refuses_to_write_a_non_finite_number() -> None:
    # Symmetry: the write refuses what the strict read would reject, so a stored
    # payload can never be one the reader has to fail on.
    with pytest.raises(ValueError):
        json.dumps({"v": float("inf")}, allow_nan=False)


def test_the_repair_added_no_migration_path() -> None:
    from prp_runtime.storage.sqlite import SCHEMA_PATH

    lines = SCHEMA_PATH.read_text(encoding="utf-8").splitlines()
    # The comments are where the file explains that there is no migration path, so
    # the statements are what has to be checked.
    statements = " ".join(
        line for line in lines if not line.strip().startswith("--")
    ).lower()
    assert "alter table" not in statements
    assert "drop table" not in statements
    assert "migrat" not in statements
