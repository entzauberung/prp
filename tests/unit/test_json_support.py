"""Targeted tests for the runtime's only JSON parser."""

import json
import math

import pytest

from prp_runtime.json_support import (
    NON_STANDARD_JSON_CONSTANTS,
    StrictJsonError,
    strict_json_loads,
)

# --- what the grammar defines still works ---------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("null", None),
        ("true", True),
        ("false", False),
        ("0", 0),
        ("-17", -17),
        ("1.5", 1.5),
        ("-0.25", -0.25),
        ("1e3", 1000.0),
        ('"text"', "text"),
        ('""', ""),
        ("[]", []),
        ("{}", {}),
        ("[1, 2, 3]", [1, 2, 3]),
        ('{"a": 1}', {"a": 1}),
    ],
)
def test_standard_json_is_returned_unchanged(text: str, expected: object) -> None:
    assert strict_json_loads(text) == expected


def test_nested_structures_keep_their_shape() -> None:
    text = '{"a": [1, {"b": null}], "c": {"d": [true, false]}}'
    assert strict_json_loads(text) == json.loads(text)


def test_integers_stay_int_and_fractions_stay_float() -> None:
    assert isinstance(strict_json_loads("7"), int)
    assert isinstance(strict_json_loads("7.0"), float)
    assert strict_json_loads("7") == 7


def test_zero_and_boolean_are_not_confused() -> None:
    assert strict_json_loads("0") is not False
    assert strict_json_loads("false") is False


# --- the three non standard constants -------------------------------------------


def test_the_rejected_constant_set_is_exactly_the_three_extras() -> None:
    assert NON_STANDARD_JSON_CONSTANTS == frozenset({"NaN", "Infinity", "-Infinity"})


@pytest.mark.parametrize("constant", sorted(NON_STANDARD_JSON_CONSTANTS))
def test_non_standard_constants_are_rejected_at_the_top_level(constant: str) -> None:
    # The standard library accepts these; this parser is the reason they stop here.
    assert math.isnan(json.loads(constant)) or math.isinf(json.loads(constant))
    with pytest.raises(StrictJsonError) as caught:
        strict_json_loads(constant)
    assert caught.value.token == constant


@pytest.mark.parametrize("constant", sorted(NON_STANDARD_JSON_CONSTANTS))
def test_non_standard_constants_are_rejected_inside_an_object(constant: str) -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads('{"value": ' + constant + "}")


@pytest.mark.parametrize("constant", sorted(NON_STANDARD_JSON_CONSTANTS))
def test_non_standard_constants_are_rejected_inside_an_array(constant: str) -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads("[1, " + constant + "]")


@pytest.mark.parametrize("constant", sorted(NON_STANDARD_JSON_CONSTANTS))
def test_non_standard_constants_are_rejected_when_deeply_nested(constant: str) -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads('{"a": {"b": [{"c": ' + constant + "}]}}")


def test_a_string_that_merely_spells_a_constant_is_still_a_string() -> None:
    assert strict_json_loads('"NaN"') == "NaN"
    assert strict_json_loads('{"Infinity": "-Infinity"}') == {"Infinity": "-Infinity"}


# --- literals that would overflow to infinity -----------------------------------


@pytest.mark.parametrize("text", ["1e999", "-1e999", "1E400", "1.7976931348623159e309"])
def test_numeric_overflow_to_infinity_is_rejected(text: str) -> None:
    # json.loads returns inf here without complaint; strict parsing must not.
    assert math.isinf(json.loads(text))
    with pytest.raises(StrictJsonError):
        strict_json_loads(text)


def test_large_but_finite_numbers_are_accepted() -> None:
    assert strict_json_loads("1e308") == 1e308
    assert math.isfinite(strict_json_loads("-1.5e300"))


def test_a_huge_integer_stays_an_exact_int() -> None:
    assert strict_json_loads("1" + "0" * 400) == 10**400


# --- malformed text -------------------------------------------------------------


@pytest.mark.parametrize(
    "text", ["", "{not json", "[1, 2", "'single'", "undefined", "{,}", "01", "+1", "nan"]
)
def test_malformed_text_is_rejected(text: str) -> None:
    with pytest.raises(StrictJsonError):
        strict_json_loads(text)


# --- error shape ----------------------------------------------------------------


def test_the_error_is_a_value_error_so_validators_can_surface_it() -> None:
    assert issubclass(StrictJsonError, ValueError)


def test_the_error_states_a_reason_without_leaking_a_traceback() -> None:
    with pytest.raises(StrictJsonError) as caught:
        strict_json_loads("[NaN]")
    error = caught.value
    assert error.reason == str(error)
    assert error.reason.strip() != ""
    assert "Traceback" not in error.reason
    assert "json_support" not in error.reason


def test_a_malformed_document_reports_no_token() -> None:
    with pytest.raises(StrictJsonError) as caught:
        strict_json_loads("{not json")
    assert caught.value.token is None
    assert "invalid JSON" in caught.value.reason
