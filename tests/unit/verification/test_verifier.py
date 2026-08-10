"""Targeted tests for the deterministic verifier."""

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic import ValidationError

from prp_runtime.domain.enums import ModelRole
from prp_runtime.domain.errors import DomainValidationError, ErrorCode
from prp_runtime.domain.models import (
    Artifact,
    ArtifactKind,
    Attempt,
    Evidence,
    EvidenceKind,
    NativeRunRequest,
    OutputRequirement,
    Run,
    VerificationResult,
    WorkUnit,
    new_artifact_id,
    new_evidence_id,
)
from prp_runtime.domain.values import (
    ModelRef,
    new_attempt_id,
    new_run_id,
    new_work_unit_id,
)
from prp_runtime.storage.sqlite import SqliteStore
from prp_runtime.verification.rules import (
    SUPPORTED_SCHEMA_KEYWORDS,
    VerificationCheck,
    VerificationRule,
    check_json_schema,
    plan_for_output,
)
from prp_runtime.verification.verifier import (
    CheckOutcome,
    RuleVerifier,
    VerificationReport,
    aggregate,
)

RUN_ID = new_run_id()
WORK_UNIT_ID = new_work_unit_id()
ATTEMPT_ID = new_attempt_id()

PASS = VerificationResult.PASS
FAIL = VerificationResult.FAIL
UNDECIDED = VerificationResult.INCONCLUSIVE


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


def json_artifact(document: object) -> Artifact:
    return artifact(json.dumps(document), ArtifactKind.JSON)


def verify(art: Artifact, *checks: VerificationCheck) -> VerificationReport:
    return RuleVerifier().verify(art, tuple(checks))


def only(art: Artifact, check: VerificationCheck) -> CheckOutcome:
    report = verify(art, check)
    assert len(report.outcomes) == 1
    return report.outcomes[0]


# --- rule surface ---------------------------------------------------------------


def test_the_supported_rule_set_is_closed() -> None:
    assert {rule.value for rule in VerificationRule} == {
        "NON_EMPTY_OUTPUT",
        "OUTPUT_KIND_MATCHES",
        "VALID_JSON",
        "MATCHES_JSON_SCHEMA",
        "REQUIRED_REFERENCES",
        "WITHIN_LENGTH_LIMIT",
    }


def test_every_rule_has_a_deterministic_implementation() -> None:
    verifier = RuleVerifier()
    sample = json_artifact({"name": "x"})
    parameters: dict[VerificationRule, dict[str, object]] = {
        VerificationRule.NON_EMPTY_OUTPUT: {},
        VerificationRule.OUTPUT_KIND_MATCHES: {"expected_kind": ArtifactKind.JSON},
        VerificationRule.VALID_JSON: {},
        VerificationRule.MATCHES_JSON_SCHEMA: {"json_schema": '{"type": "object"}'},
        VerificationRule.REQUIRED_REFERENCES: {"required_references": ("name",)},
        VerificationRule.WITHIN_LENGTH_LIMIT: {"max_characters": 100},
    }
    assert set(parameters) == set(VerificationRule)
    for rule, extra in parameters.items():
        outcome = verifier.verify(sample, (VerificationCheck(rule=rule, **extra),)).outcomes[0]
        assert outcome.rule is rule
        assert outcome.result in tuple(VerificationResult)


def test_an_unknown_rule_name_is_rejected() -> None:
    with pytest.raises(ValidationError):
        VerificationCheck(rule="LOOKS_GOOD_TO_ME")
    with pytest.raises(ValueError):
        VerificationRule("SEEMS_FINE")


@pytest.mark.parametrize(
    ("rule", "missing"),
    [
        (VerificationRule.OUTPUT_KIND_MATCHES, "expected_kind"),
        (VerificationRule.MATCHES_JSON_SCHEMA, "json_schema"),
        (VerificationRule.REQUIRED_REFERENCES, "at least one reference"),
        (VerificationRule.WITHIN_LENGTH_LIMIT, "max_characters"),
    ],
)
def test_a_rule_without_its_parameter_is_rejected(
    rule: VerificationRule, missing: str
) -> None:
    with pytest.raises(ValidationError, match=missing):
        VerificationCheck(rule=rule)


def test_check_parameters_are_validated() -> None:
    with pytest.raises(ValidationError):
        VerificationCheck(rule=VerificationRule.MATCHES_JSON_SCHEMA, json_schema="{not json")
    with pytest.raises(ValidationError):
        VerificationCheck(
            rule=VerificationRule.REQUIRED_REFERENCES, required_references=("  ",)
        )
    with pytest.raises(ValidationError):
        VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=0)


# A check carrying a parameter its rule never reads is ambiguous: a reader cannot
# tell which constraint was meant. Accepting it and ignoring the extra would hide
# the mistake, so the declaration is refused.
@pytest.mark.parametrize(
    ("rule", "extra"),
    [
        (VerificationRule.NON_EMPTY_OUTPUT, {"max_characters": 10}),
        (VerificationRule.NON_EMPTY_OUTPUT, {"expected_kind": ArtifactKind.JSON}),
        (VerificationRule.VALID_JSON, {"json_schema": '{"type": "object"}'}),
        (VerificationRule.VALID_JSON, {"required_references": ("a",)}),
        (VerificationRule.WITHIN_LENGTH_LIMIT, {"expected_kind": ArtifactKind.TEXT}),
        (VerificationRule.OUTPUT_KIND_MATCHES, {"required_references": ("a",)}),
        (VerificationRule.MATCHES_JSON_SCHEMA, {"max_characters": 5}),
        (VerificationRule.REQUIRED_REFERENCES, {"json_schema": '{"type": "object"}'}),
    ],
)
def test_a_check_rejects_a_parameter_its_rule_does_not_use(
    rule: VerificationRule, extra: dict[str, object]
) -> None:
    required: dict[VerificationRule, dict[str, object]] = {
        VerificationRule.OUTPUT_KIND_MATCHES: {"expected_kind": ArtifactKind.JSON},
        VerificationRule.MATCHES_JSON_SCHEMA: {"json_schema": '{"type": "object"}'},
        VerificationRule.REQUIRED_REFERENCES: {"required_references": ("report.md",)},
        VerificationRule.WITHIN_LENGTH_LIMIT: {"max_characters": 5},
    }
    with pytest.raises(ValidationError, match="does not use"):
        VerificationCheck(rule=rule, **required.get(rule, {}), **extra)


def test_a_check_that_carries_only_its_own_parameter_is_accepted() -> None:
    assert VerificationCheck(rule=VerificationRule.VALID_JSON).json_schema is None
    assert (
        VerificationCheck(
            rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=5
        ).max_characters
        == 5
    )


# --- individual rules -----------------------------------------------------------


def test_non_empty_output_fails_on_blank_content() -> None:
    # Artifact already forbids blank content, so the check is proved against a
    # model_construct instance: the rule must not depend on that outer guard.
    blank = Artifact.model_construct(
        artifact_id=new_artifact_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        attempt_id=ATTEMPT_ID,
        name="answer",
        kind=ArtifactKind.TEXT,
        content="   \n\t ",
    )
    outcome = only(blank, VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT))
    assert outcome.result is FAIL
    assert "empty" in outcome.detail

    check = VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT)
    assert only(artifact("an answer"), check).result is PASS


def test_output_kind_must_match_the_requirement() -> None:
    check = VerificationCheck(
        rule=VerificationRule.OUTPUT_KIND_MATCHES, expected_kind=ArtifactKind.JSON
    )
    assert only(json_artifact({"ok": True}), check).result is PASS
    prose = only(artifact("just prose"), check)
    assert prose.result is FAIL
    assert "TEXT" in prose.detail and "JSON" in prose.detail


def test_valid_json_decides_both_ways() -> None:
    check = VerificationCheck(rule=VerificationRule.VALID_JSON)
    assert only(json_artifact([1, 2]), check).result is PASS
    # A TEXT artifact is not parsed by the domain, so malformed JSON reaches here.
    broken = only(artifact("{'almost': json}"), check)
    assert broken.result is FAIL
    assert "not valid JSON" in broken.detail


def test_content_that_is_not_json_fails_a_schema_rather_than_going_undecided() -> None:
    # Non-JSON content provably fails an object schema, so the parse verdict is
    # reported even when the schema itself uses undecidable keywords.
    outcome = only(
        artifact("just prose"), schema_check(json.dumps({"type": "object", "oneOf": []}))
    )
    assert outcome.result is FAIL
    assert "not valid JSON" in outcome.detail


def test_required_references_must_all_appear() -> None:
    check = VerificationCheck(
        rule=VerificationRule.REQUIRED_REFERENCES,
        required_references=("report.md", "table-3"),
    )
    assert only(artifact("see report.md and table-3"), check).result is PASS
    missing = only(artifact("see report.md only"), check)
    assert missing.result is FAIL
    assert "table-3" in missing.detail


def test_length_limit_is_inclusive() -> None:
    check = VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=5)
    assert only(artifact("12345"), check).result is PASS
    over = only(artifact("123456"), check)
    assert over.result is FAIL
    assert "over the 5 character limit" in over.detail


# --- json schema subset ---------------------------------------------------------

OBJECT_SCHEMA = json.dumps(
    {
        "type": "object",
        "required": ["title", "score"],
        "properties": {
            "title": {"type": "string", "minLength": 1},
            "score": {"type": "integer", "minimum": 0, "maximum": 10},
            "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3},
        },
        "additionalProperties": False,
    }
)


def schema_check(schema: str = OBJECT_SCHEMA) -> VerificationCheck:
    return VerificationCheck(rule=VerificationRule.MATCHES_JSON_SCHEMA, json_schema=schema)


def test_a_conforming_document_passes() -> None:
    document = {"title": "a title", "score": 7, "tags": ["a", "b"]}
    assert only(json_artifact(document), schema_check()).result is PASS


@pytest.mark.parametrize(
    ("document", "expected_path", "expected_in_detail"),
    [
        ({"score": 7}, "$", "'title' is a required property"),
        ({"title": "t", "score": "high"}, "$.score", "is not of type 'integer'"),
        ({"title": "t", "score": 99}, "$.score", "greater than the maximum of 10"),
        ({"title": "t", "score": -1}, "$.score", "less than the minimum of 0"),
        ({"title": "", "score": 1}, "$.title", "should be non-empty"),
        ({"title": "t", "score": 1, "extra": 1}, "$", "'extra' was unexpected"),
        ({"title": "t", "score": 1, "tags": ["a", "b", "c", "d"]}, "$.tags", "is too long"),
        ({"title": "t", "score": 1, "tags": [1]}, "$.tags[0]", "is not of type 'string'"),
        ([], "$", "is not of type 'object'"),
    ],
)
def test_a_non_conforming_document_fails_with_a_reason(
    document: object, expected_path: str, expected_in_detail: str
) -> None:
    outcome = only(json_artifact(document), schema_check())
    assert outcome.result is FAIL
    assert expected_path in outcome.detail
    assert expected_in_detail in outcome.detail


def test_a_failure_names_the_location_it_was_found_at() -> None:
    outcome = only(json_artifact({"title": "t", "score": "high"}), schema_check())
    assert outcome.detail.startswith("$.score ")


def test_multiple_failures_are_reported_in_a_stable_order() -> None:
    document = {"title": "", "score": 99}
    first = only(json_artifact(document), schema_check()).detail
    assert first == only(json_artifact(document), schema_check()).detail
    assert first.index("$.score") < first.index("$.title")


def test_booleans_are_not_accepted_as_numbers() -> None:
    schema = json.dumps({"type": "object", "properties": {"score": {"type": "integer"}}})
    outcome = only(json_artifact({"score": True}), schema_check(schema))
    assert outcome.result is FAIL


def test_enum_and_const_are_decided() -> None:
    enum_schema = json.dumps({"type": "object", "properties": {"k": {"enum": ["a", "b"]}}})
    assert only(json_artifact({"k": "a"}), schema_check(enum_schema)).result is PASS
    assert only(json_artifact({"k": "z"}), schema_check(enum_schema)).result is FAIL

    const_schema = json.dumps({"type": "object", "properties": {"k": {"const": 1}}})
    assert only(json_artifact({"k": 1}), schema_check(const_schema)).result is PASS
    assert only(json_artifact({"k": 2}), schema_check(const_schema)).result is FAIL


# In Python ``True == 1`` and ``False == 0``. JSON Schema says a boolean is never a
# number, so equality here has to follow JSON and not Python, or ``true`` would be
# accepted for ``enum: [1]``.
@pytest.mark.parametrize(
    ("schema", "document"),
    [
        ({"enum": [1, 2]}, True),
        ({"enum": [0]}, False),
        ({"const": 1}, True),
        ({"const": 0}, False),
        ({"enum": [True]}, 1),
        ({"const": True}, 1),
        ({"const": False}, 0),
    ],
)
def test_a_boolean_is_never_equal_to_a_number(
    schema: dict[str, object], document: object
) -> None:
    outcome = only(json_artifact(document), schema_check(json.dumps(schema)))
    assert outcome.result is FAIL


@pytest.mark.parametrize(
    ("schema", "document"),
    [
        ({"const": True}, True),
        ({"const": False}, False),
        ({"enum": [1, 2]}, 1),
        ({"enum": [True, False]}, False),
        # 1 and 1.0 are the same JSON number.
        ({"const": 1}, 1.0),
        ({"enum": [2]}, 2.0),
    ],
)
def test_equality_that_json_calls_equal_still_passes(
    schema: dict[str, object], document: object
) -> None:
    outcome = only(json_artifact(document), schema_check(json.dumps(schema)))
    assert outcome.result is PASS


def test_nested_objects_are_validated() -> None:
    schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "inner": {
                    "type": "object",
                    "required": ["deep"],
                    "properties": {"deep": {"type": "string"}},
                }
            },
        }
    )
    assert only(json_artifact({"inner": {"deep": "v"}}), schema_check(schema)).result is PASS
    outcome = only(json_artifact({"inner": {}}), schema_check(schema))
    assert outcome.result is FAIL
    assert "$.inner" in outcome.detail
    assert "'deep' is a required property" in outcome.detail


@pytest.mark.parametrize(
    "unsupported",
    [
        {"type": "object", "oneOf": [{"type": "object"}]},
        {"type": "object", "properties": {"k": {"$ref": "#/definitions/x"}}},
        {"type": "object", "patternProperties": {"^a": {"type": "string"}}},
        {"type": "object", "properties": {"k": {"type": "string", "pattern": "^a"}}},
        {"type": "array", "items": [{"type": "string"}]},
        {"type": "object", "allOf": [{"required": ["a"]}]},
    ],
)
def test_an_undecidable_schema_is_inconclusive_never_a_pass(
    unsupported: dict[str, object],
) -> None:
    outcome = only(json_artifact({"k": "v"}), schema_check(json.dumps(unsupported)))
    assert outcome.result is UNDECIDED
    assert "cannot decide" in outcome.detail


def test_the_supported_keyword_set_is_explicit() -> None:
    assert "type" in SUPPORTED_SCHEMA_KEYWORDS
    assert "oneOf" not in SUPPORTED_SCHEMA_KEYWORDS
    assert "$ref" not in SUPPORTED_SCHEMA_KEYWORDS
    assert "pattern" not in SUPPORTED_SCHEMA_KEYWORDS


def test_malformed_json_against_a_schema_fails_rather_than_erroring() -> None:
    result, detail = check_json_schema("{not json", '{"type": "object"}')
    assert result is FAIL
    assert "not valid JSON" in detail


def test_a_malformed_declared_schema_is_a_configuration_error() -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        check_json_schema('{"ok": 1}', "{not json")
    assert excinfo.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT
    with pytest.raises(DomainValidationError):
        check_json_schema('{"ok": 1}', '["not", "an", "object"]')


# A supported keyword given the wrong kind of value is a broken declaration, not
# evidence about the artifact. Reporting FAIL would blame the output for it, and
# reporting PASS would claim a constraint was applied when it could not be.
@pytest.mark.parametrize(
    "illegal",
    [
        {"type": "object", "properties": {"k": {"minLength": "5"}}},
        {"type": "object", "properties": {"k": {"maxLength": 1.5}}},
        {"type": "object", "properties": {"k": {"minimum": "0"}}},
        {"enum": "x"},
        {"properties": []},
        {"required": "title"},
        {"required": ["title", "title"]},
        {"maxItems": -1},
        {"minItems": "2"},
        {"additionalProperties": 5},
        {"type": "objekt"},
        {"type": []},
        {"items": 5},
    ],
)
def test_an_illegal_schema_is_a_configuration_error_not_a_verdict(
    illegal: dict[str, object],
) -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        check_json_schema('{"k": "v"}', json.dumps(illegal))
    assert excinfo.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT
    assert excinfo.value.detail.field == "json_schema"


@pytest.mark.parametrize(
    "illegal",
    [{"minLength": "5"}, {"enum": "x"}, {"properties": []}, {"required": "title"}],
)
def test_an_illegal_schema_is_refused_where_the_check_is_declared(
    illegal: dict[str, object],
) -> None:
    with pytest.raises(ValidationError, match="not a legal JSON Schema"):
        schema_check(json.dumps(illegal))


def test_an_empty_enum_is_legal_but_satisfied_by_nothing() -> None:
    # Draft 2020-12 puts no minimum on "enum", so this is a legal schema that no
    # document can match. That is a provable FAIL, not a broken declaration.
    outcome = only(json_artifact("anything"), schema_check(json.dumps({"enum": []})))
    assert outcome.result is FAIL


def test_an_out_of_range_keyword_is_undecided_even_when_also_illegal() -> None:
    # "oneOf" is outside the supported subset, and an empty "oneOf" also breaks the
    # meta-schema. Out of range wins: this runtime does not rule on a dialect it
    # does not apply, so the check is undecided rather than called a broken
    # declaration.
    outcome = only(json_artifact({"k": "v"}), schema_check(json.dumps({"oneOf": []})))
    assert outcome.result is UNDECIDED


def test_an_undecidable_schema_is_still_declarable() -> None:
    check = schema_check(json.dumps({"type": "object", "oneOf": [{"type": "object"}]}))
    assert check.json_schema is not None


# --- strict json at the verification boundary -----------------------------------


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_non_standard_json_is_not_valid_json(constant: str) -> None:
    # A TEXT artifact is not parsed by the domain, so the content reaches the rule.
    outcome = only(
        artifact('{"v": ' + constant + "}"),
        VerificationCheck(rule=VerificationRule.VALID_JSON),
    )
    assert outcome.result is FAIL
    assert "not valid JSON" in outcome.detail


def test_non_standard_json_fails_a_schema_check() -> None:
    outcome = only(
        artifact('{"score": NaN}'), schema_check(json.dumps({"type": "object"}))
    )
    assert outcome.result is FAIL
    assert "not valid JSON" in outcome.detail


def test_a_schema_declaring_a_non_standard_number_is_a_configuration_error() -> None:
    with pytest.raises(DomainValidationError) as excinfo:
        check_json_schema('{"k": 1}', '{"maximum": Infinity}')
    assert excinfo.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT


# --- audit matrix: nothing unprovable may report a pass -------------------------

#: Schemas whose keywords are all inside the supported subset but whose values are
#: the wrong kind. A checker that only knows how to apply a keyword when its value
#: has the expected shape will skip these, and a skipped constraint reads as a
#: pass. Every one of them must be refused instead.
ILLEGAL_SCHEMA_PARAMETERS: list[dict[str, object]] = [
    {"minLength": "5"},
    {"maxLength": 1.5},
    {"minLength": -1},
    {"minimum": "0"},
    {"maximum": None},
    {"minItems": "2"},
    {"maxItems": -1},
    {"enum": "x"},
    {"enum": {"a": 1}},
    {"required": "title"},
    {"required": ["title", "title"]},
    {"required": [1]},
    {"properties": []},
    {"properties": "title"},
    {"additionalProperties": 5},
    {"type": "objekt"},
    {"type": []},
    {"type": 1},
    {"items": 5},
]


@pytest.mark.parametrize("schema", ILLEGAL_SCHEMA_PARAMETERS)
def test_an_illegal_schema_parameter_can_never_report_a_pass(
    schema: dict[str, object],
) -> None:
    """Refused at both layers, by stable code rather than by message text."""
    text = json.dumps(schema)
    with pytest.raises(DomainValidationError) as raised:
        check_json_schema('{"title": "t", "k": "v"}', text)
    assert raised.value.code is ErrorCode.INVALID_OUTPUT_REQUIREMENT
    assert raised.value.detail.field == "json_schema"
    # The same schema cannot be declared as a check either, so no verify path
    # exists that could reach a verdict about it.
    with pytest.raises(ValidationError):
        schema_check(text)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize(
    "template", ["{}", '{{"v": {}}}', "[{}]", '{{"a": [{{"b": {}}}]}}']
)
def test_no_json_rule_passes_a_non_standard_constant(constant: str, template: str) -> None:
    content = template.format(constant)
    for check in (
        VerificationCheck(rule=VerificationRule.VALID_JSON),
        schema_check(json.dumps({"type": "object"})),
        schema_check(json.dumps({"type": "array"})),
    ):
        outcome = only(artifact(content), check)
        assert outcome.result is FAIL
        assert "not valid JSON" in outcome.detail


def test_the_three_verdicts_are_distinguishable_for_one_document() -> None:
    """The same document against three schemas must give three different answers."""
    document = json_artifact({"k": "v"})
    conforming = schema_check(json.dumps({"type": "object", "required": ["k"]}))
    contradicted = schema_check(json.dumps({"type": "object", "required": ["missing"]}))
    out_of_range = schema_check(json.dumps({"type": "object", "oneOf": [{"type": "object"}]}))

    assert only(document, conforming).result is PASS
    assert only(document, contradicted).result is FAIL
    assert only(document, out_of_range).result is UNDECIDED


# --- aggregation ----------------------------------------------------------------


@pytest.mark.parametrize(
    ("results", "expected"),
    [
        ((), UNDECIDED),
        ((PASS,), PASS),
        ((PASS, PASS), PASS),
        ((PASS, FAIL), FAIL),
        ((PASS, UNDECIDED), UNDECIDED),
        ((FAIL, UNDECIDED), FAIL),
        ((UNDECIDED, UNDECIDED), UNDECIDED),
    ],
)
def test_aggregation_never_upgrades_a_doubt_to_a_pass(
    results: tuple[VerificationResult, ...], expected: VerificationResult
) -> None:
    assert aggregate(results) is expected


def test_an_empty_plan_proves_nothing() -> None:
    report = verify(artifact("anything"))
    assert report.outcomes == ()
    assert report.result is UNDECIDED
    assert report.passed is False


def test_report_exposes_failures_and_undecided_checks() -> None:
    report = verify(
        json_artifact({"k": "v"}),
        VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT),
        VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=2),
        schema_check(json.dumps({"type": "object", "oneOf": []})),
    )
    assert report.result is FAIL
    assert [outcome.rule for outcome in report.failures] == [
        VerificationRule.WITHIN_LENGTH_LIMIT
    ]
    assert [outcome.rule for outcome in report.undecided] == [
        VerificationRule.MATCHES_JSON_SCHEMA
    ]
    assert report.summary() == "FAIL: 1 passed, 1 failed, 1 undecided"


# --- plan construction ----------------------------------------------------------


def test_a_text_requirement_plans_the_text_checks() -> None:
    plan = plan_for_output(OutputRequirement())
    assert [check.rule for check in plan] == [
        VerificationRule.NON_EMPTY_OUTPUT,
        VerificationRule.OUTPUT_KIND_MATCHES,
    ]
    assert plan[1].expected_kind is ArtifactKind.TEXT


def test_a_json_requirement_plans_the_json_checks() -> None:
    plan = plan_for_output(
        OutputRequirement(kind=ArtifactKind.JSON, json_schema=OBJECT_SCHEMA)
    )
    assert [check.rule for check in plan] == [
        VerificationRule.NON_EMPTY_OUTPUT,
        VerificationRule.OUTPUT_KIND_MATCHES,
        VerificationRule.VALID_JSON,
        VerificationRule.MATCHES_JSON_SCHEMA,
    ]


def test_json_without_a_schema_does_not_plan_a_schema_check() -> None:
    plan = plan_for_output(OutputRequirement(kind=ArtifactKind.JSON))
    assert VerificationRule.MATCHES_JSON_SCHEMA not in [check.rule for check in plan]


def test_optional_checks_are_planned_only_when_declared() -> None:
    plan = plan_for_output(
        OutputRequirement(), required_references=("report.md",), max_characters=500
    )
    assert [check.rule for check in plan][-2:] == [
        VerificationRule.REQUIRED_REFERENCES,
        VerificationRule.WITHIN_LENGTH_LIMIT,
    ]


def test_planning_is_deterministic() -> None:
    requirement = OutputRequirement(kind=ArtifactKind.JSON, json_schema=OBJECT_SCHEMA)
    assert plan_for_output(requirement) == plan_for_output(requirement)


def test_an_end_to_end_plan_passes_a_good_artifact() -> None:
    requirement = OutputRequirement(kind=ArtifactKind.JSON, json_schema=OBJECT_SCHEMA)
    report = RuleVerifier().verify(
        json_artifact({"title": "t", "score": 3}), plan_for_output(requirement)
    )
    assert report.result is PASS
    assert len(report.outcomes) == 4


# --- evidence -------------------------------------------------------------------


def test_evidence_records_rule_result_and_references() -> None:
    report = verify(
        artifact("an answer"),
        VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT),
        VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=2),
    )
    rows = report.to_evidence()
    assert len(rows) == 2
    for row, outcome in zip(rows, report.outcomes, strict=True):
        assert row.rule == outcome.rule.value
        assert row.result is outcome.result
        assert row.passed is outcome.result.is_pass
        assert row.kind is EvidenceKind.DETERMINISTIC_CHECK
        assert row.artifact_id == report.artifact_id
        assert row.work_unit_id == report.work_unit_id
        assert row.run_id == report.run_id
        assert row.detail == outcome.detail


def test_inconclusive_evidence_is_not_stored_as_a_proven_failure() -> None:
    report = verify(
        json_artifact({"k": "v"}),
        schema_check(json.dumps({"type": "object", "oneOf": []})),
    )
    row = report.to_evidence()[0]
    assert row.result is UNDECIDED
    assert row.passed is False
    # The distinction survives: a reader can tell "not proven" from "proven bad".
    assert row.result is not FAIL


def test_evidence_cannot_carry_a_boolean_that_disagrees_with_the_verdict() -> None:
    # There is nothing left to contradict: ``passed`` is derived, so supplying it
    # is refused outright rather than checked for agreement.
    with pytest.raises(ValidationError):
        Evidence(
            evidence_id=new_evidence_id(),
            run_id=RUN_ID,
            work_unit_id=WORK_UNIT_ID,
            artifact_id=new_artifact_id(),
            kind=EvidenceKind.DETERMINISTIC_CHECK,
            rule=VerificationRule.VALID_JSON.value,
            result=VerificationResult.FAIL,
            passed=True,
            detail="inconsistent",
        )


def test_a_model_review_may_omit_the_rule() -> None:
    row = Evidence(
        evidence_id=new_evidence_id(),
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        artifact_id=new_artifact_id(),
        kind=EvidenceKind.MODEL_REVIEW,
        result=VerificationResult.PASS,
        detail="a reviewer accepted it",
    )
    assert row.rule is None
    assert row.passed is True


def test_a_deterministic_check_cannot_omit_its_rule() -> None:
    with pytest.raises(ValidationError, match="DETERMINISTIC_CHECK requires rule"):
        Evidence(
            evidence_id=new_evidence_id(),
            run_id=RUN_ID,
            work_unit_id=WORK_UNIT_ID,
            artifact_id=new_artifact_id(),
            kind=EvidenceKind.DETERMINISTIC_CHECK,
            result=VerificationResult.PASS,
            detail="a rule that does not say which rule",
        )


def test_every_evidence_row_the_verifier_writes_names_a_rule_and_a_verdict() -> None:
    report = verify(
        json_artifact({"k": "v"}),
        VerificationCheck(rule=VerificationRule.NON_EMPTY_OUTPUT),
        VerificationCheck(rule=VerificationRule.VALID_JSON),
        schema_check(json.dumps({"type": "object", "oneOf": []})),
        VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=1),
    )
    rows = report.to_evidence()
    assert len(rows) == 4
    assert {row.result for row in rows} == {PASS, FAIL, UNDECIDED}
    for row in rows:
        assert row.kind is EvidenceKind.DETERMINISTIC_CHECK
        assert row.rule in {rule.value for rule in VerificationRule}
        assert row.passed is row.result.is_pass


# --- persistence ----------------------------------------------------------------


@pytest_asyncio.fixture
async def store(tmp_path: Path) -> AsyncIterator[SqliteStore]:
    async with SqliteStore(tmp_path / "verification.db") as opened:
        yield opened


async def seed(store: SqliteStore) -> Artifact:
    run = Run(run_id=RUN_ID, request=NativeRunRequest(input="verify me"))
    await store.create_run(run)
    unit = WorkUnit(
        work_unit_id=WORK_UNIT_ID, run_id=RUN_ID, name="direct", instruction="do it"
    )
    await store.create_work_unit(unit)
    attempt = Attempt(
        attempt_id=ATTEMPT_ID,
        run_id=RUN_ID,
        work_unit_id=WORK_UNIT_ID,
        role=ModelRole.WORKER,
        model=ModelRef(provider="openai_compatible", model="weak-model"),
    )
    await store.create_attempt(attempt)
    produced = json_artifact({"title": "t", "score": 3})
    await store.add_artifact(produced)
    return produced


@pytest.mark.asyncio
async def test_evidence_round_trips_through_the_store(store: SqliteStore) -> None:
    produced = await seed(store)
    requirement = OutputRequirement(kind=ArtifactKind.JSON, json_schema=OBJECT_SCHEMA)

    report = await RuleVerifier().verify_and_record(
        store, produced, plan_for_output(requirement)
    )

    assert report.result is PASS
    stored = await store.list_evidence(WORK_UNIT_ID)
    assert len(stored) == 4
    assert {row.rule for row in stored} == {
        VerificationRule.NON_EMPTY_OUTPUT.value,
        VerificationRule.OUTPUT_KIND_MATCHES.value,
        VerificationRule.VALID_JSON.value,
        VerificationRule.MATCHES_JSON_SCHEMA.value,
    }
    assert all(row.result is PASS for row in stored)
    assert all(row.passed for row in stored)


@pytest.mark.asyncio
async def test_an_inconclusive_verdict_survives_persistence(store: SqliteStore) -> None:
    produced = await seed(store)

    await RuleVerifier().verify_and_record(
        store, produced, (schema_check(json.dumps({"type": "object", "oneOf": []})),)
    )

    stored = await store.list_evidence(WORK_UNIT_ID)
    assert len(stored) == 1
    assert stored[0].result is UNDECIDED
    assert stored[0].passed is False
    assert stored[0].rule == VerificationRule.MATCHES_JSON_SCHEMA.value


@pytest.mark.asyncio
async def test_verification_never_touches_run_or_work_unit_state(
    store: SqliteStore,
) -> None:
    produced = await seed(store)
    before_run = await store.get_run(RUN_ID)
    before_unit = await store.get_work_unit(WORK_UNIT_ID)
    before_attempt = await store.get_attempt(ATTEMPT_ID)
    before_events = await store.list_events(RUN_ID)

    report = await RuleVerifier().verify_and_record(
        store,
        produced,
        (VerificationCheck(rule=VerificationRule.WITHIN_LENGTH_LIMIT, max_characters=1),),
    )

    assert report.result is FAIL
    assert await store.get_run(RUN_ID) == before_run
    assert await store.get_work_unit(WORK_UNIT_ID) == before_unit
    assert await store.get_attempt(ATTEMPT_ID) == before_attempt
    # Emitting the event is the controller's decision, not the verifier's.
    assert await store.list_events(RUN_ID) == before_events


@pytest.mark.asyncio
async def test_an_empty_plan_writes_no_evidence(store: SqliteStore) -> None:
    produced = await seed(store)
    report = await RuleVerifier().verify_and_record(store, produced, ())
    assert report.result is UNDECIDED
    assert await store.list_evidence(WORK_UNIT_ID) == ()
