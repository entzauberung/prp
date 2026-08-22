"""Unit tests for group_logging module."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_LOGGING_PATH = Path(__file__).parents[3] / "external_tests" / "group_logging.py"
_LOGGING_SPEC = importlib.util.spec_from_file_location("prp_external_group_logging", _LOGGING_PATH)
assert _LOGGING_SPEC is not None and _LOGGING_SPEC.loader is not None
_LOGGING = importlib.util.module_from_spec(_LOGGING_SPEC)
sys.modules["prp_external_group_logging"] = _LOGGING
_LOGGING_SPEC.loader.exec_module(_LOGGING)

GroupLogger = _LOGGING.GroupLogger
ScenarioResult = _LOGGING.ScenarioResult
RESULT_CLASSIFICATIONS = _LOGGING.RESULT_CLASSIFICATIONS
MAX_LOG_FILE_BYTES = _LOGGING.MAX_LOG_FILE_BYTES
ensure_log_directory = _LOGGING.ensure_log_directory
check_directory_size = _LOGGING.check_directory_size


def test_scenario_result_valid_classification():
    """Valid classification should not raise."""
    result = ScenarioResult(
        scenario="test_example",
        classification="PASS",
        exit_code=0,
        duration_seconds=1.5,
        error_summary="",
    )
    assert result.classification == "PASS"


def test_scenario_result_invalid_classification():
    """Invalid classification should raise ValueError."""
    with pytest.raises(ValueError, match="invalid classification"):
        ScenarioResult(
            scenario="test_example",
            classification="INVALID",
            exit_code=0,
            duration_seconds=1.0,
        )


def test_all_classifications_in_result_set():
    """All documented classifications should be valid."""
    expected = {
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
    assert RESULT_CLASSIFICATIONS == expected


def test_group_logger_writes_command_output(tmp_path):
    """GroupLogger should write command output to .log file."""
    logger = GroupLogger(tmp_path, "10-provider")
    logger.write_command_output("test output\n")
    logger.write_command_output("more output\n")

    log_file = tmp_path / "10-provider.log"
    assert log_file.exists()
    content = log_file.read_text()
    assert "test output\n" in content
    assert "more output\n" in content


def test_group_logger_writes_scenario_result(tmp_path):
    """GroupLogger should write structured result to .jsonl file."""
    logger = GroupLogger(tmp_path, "20-integration")
    result = ScenarioResult(
        scenario="test_scenario",
        classification="PASS",
        exit_code=0,
        duration_seconds=2.5,
        error_summary="",
    )
    logger.write_scenario_result(result)

    jsonl_file = tmp_path / "20-integration.jsonl"
    assert jsonl_file.exists()
    lines = jsonl_file.read_text().strip().split("\n")
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["scenario"] == "test_scenario"
    assert parsed["classification"] == "PASS"
    assert parsed["exit_code"] == 0


def test_group_logger_truncates_command_log(tmp_path):
    """Command log should truncate when exceeding size limit."""
    logger = GroupLogger(tmp_path, "test-group")

    # Write more than MAX_LOG_FILE_BYTES
    chunk_size = 1024 * 100  # 100 KB chunks
    chunk = "x" * chunk_size
    chunks_to_exceed = (MAX_LOG_FILE_BYTES // chunk_size) + 2

    for _ in range(chunks_to_exceed):
        logger.write_command_output(chunk)

    assert logger.is_truncated()
    log_file = tmp_path / "test-group.log"
    assert log_file.stat().st_size <= MAX_LOG_FILE_BYTES


def test_group_logger_truncates_structured_log(tmp_path):
    """Structured log should truncate when exceeding size limit."""
    logger = GroupLogger(tmp_path, "test-group")

    # Create large results to exceed limit
    large_error = "e" * 10000
    results_needed = (MAX_LOG_FILE_BYTES // len(large_error)) + 2

    for i in range(results_needed):
        result = ScenarioResult(
            scenario=f"scenario_{i}",
            classification="PRODUCT_DEFECT",
            exit_code=1,
            duration_seconds=1.0,
            error_summary=large_error,
        )
        logger.write_scenario_result(result)

    assert logger.is_truncated()
    jsonl_file = tmp_path / "test-group.jsonl"

    # Should contain truncation marker
    lines = jsonl_file.read_text().strip().split("\n")
    last_line = json.loads(lines[-1])
    assert last_line["classification"] == "LOG_TRUNCATED"


def test_ensure_log_directory_creates_directory(tmp_path):
    """ensure_log_directory should create missing directory."""
    log_dir = tmp_path / "new_logs"
    assert not log_dir.exists()

    ensure_log_directory(log_dir)
    assert log_dir.exists()
    assert log_dir.is_dir()


def test_ensure_log_directory_accepts_existing(tmp_path):
    """ensure_log_directory should accept existing directory."""
    log_dir = tmp_path / "existing"
    log_dir.mkdir()

    ensure_log_directory(log_dir)  # Should not raise
    assert log_dir.is_dir()


def test_check_directory_size_empty(tmp_path):
    """check_directory_size should return 0 for empty directory."""
    assert check_directory_size(tmp_path) == 0


def test_check_directory_size_with_files(tmp_path):
    """check_directory_size should sum file sizes."""
    (tmp_path / "file1.txt").write_text("a" * 100)
    (tmp_path / "file2.txt").write_text("b" * 200)

    total = check_directory_size(tmp_path)
    assert total == 300


def test_group_logger_get_total_bytes(tmp_path):
    """get_total_bytes should return sum of both log files."""
    logger = GroupLogger(tmp_path, "test-group")

    logger.write_command_output("test output\n")
    result = ScenarioResult(
        scenario="test",
        classification="PASS",
        exit_code=0,
        duration_seconds=1.0,
    )
    logger.write_scenario_result(result)

    total = logger.get_total_bytes()
    assert total > 0
    assert total == (
        len(b"test output\n") +
        len(json.dumps({
            "scenario": "test",
            "classification": "PASS",
            "exit_code": 0,
            "duration_seconds": 1.0,
            "error_summary": "",
        }) + "\n")
    )
