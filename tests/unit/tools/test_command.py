"""Targeted tests for server-owned command specifications."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.domain.enums import IsolationMode, ToolCallStatus, ToolEffect
from prp_runtime.domain.values import new_run_id, new_tool_call_id, new_work_unit_id
from prp_runtime.policy.models import CommandClass
from prp_runtime.tools.command import (
    DEFAULT_COMMAND_REGISTRY,
    MAX_COMMAND_OUTPUT_BYTES,
    CommandArgumentKind,
    CommandCwd,
    CommandExecutionError,
    CommandInvocation,
    CommandParameter,
    CommandRegistry,
    CommandRunner,
    CommandSpec,
    build_command_plan,
    build_targeted_test_definition,
    expand_argv,
)
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import ToolCall

FIXTURE = Path(__file__).parents[2] / "fixtures" / "command_fixture.py"


def make_spec(**overrides: object) -> CommandSpec:
    values: dict[str, object] = {
        "name": "pytest",
        "executable": "pytest",
        "argv_template": ("-q", "{targets}"),
        "parameters": (
            CommandParameter(
                name="targets",
                kind=CommandArgumentKind.PATH,
                multiple=True,
                max_items=3,
            ),
        ),
        "command_class": CommandClass.TEST,
        "timeout_seconds": 15,
        "max_output_bytes": 32_768,
        "environment_allowlist": ("CI",),
    }
    values.update(overrides)
    return CommandSpec(**values)


def test_expansion_keeps_each_untrusted_value_as_one_argv_token() -> None:
    specification = make_spec()
    invocation = CommandInvocation(
        spec_name="pytest",
        parameters={"targets": ("tests/unit", "tests with spaces/test.py")},
        environment={"CI": "1"},
    )

    assert expand_argv(specification, invocation) == (
        "pytest",
        "-q",
        "tests/unit",
        "tests with spaces/test.py",
    )
    plan = build_command_plan(specification, invocation)
    assert plan.cwd is CommandCwd.WORKSPACE
    assert plan.environment == {"CI": "1"}
    assert plan.network_enabled is False
    assert plan.timeout_seconds == 15


@pytest.mark.parametrize("value", ["../escape.py", "/tmp/test.py", "tests/../test.py", "-k"])
def test_path_parameters_cannot_escape_or_inject_options(value: str) -> None:
    specification = make_spec()
    with pytest.raises(ValueError, match="(workspace-relative|option)"):
        expand_argv(specification, {"targets": (value,)})


@pytest.mark.parametrize("value", ["tests;touch pwned", "$(whoami)", "tests|cat", "tests\nother"])
def test_shell_syntax_is_rejected_before_any_executor_exists(value: str) -> None:
    specification = make_spec()
    with pytest.raises(ValueError, match="(shell syntax|control character)"):
        expand_argv(specification, {"targets": (value,)})


def test_server_owns_executable_and_template_shape() -> None:
    with pytest.raises(ValidationError, match="bare server-owned"):
        make_spec(executable="../pytest")
    with pytest.raises(ValidationError, match="one complete token"):
        make_spec(argv_template=("--target={targets}",))
    with pytest.raises(ValidationError, match="unknown parameter"):
        make_spec(argv_template=("{missing}",))
    with pytest.raises(ValidationError, match="COMMAND effect"):
        make_spec(effect="READ")


def test_parameter_contract_and_environment_allowlist_are_closed() -> None:
    with pytest.raises(ValidationError, match="max_items=1"):
        CommandParameter(name="target", max_items=2)
    specification = make_spec()
    invocation = CommandInvocation(spec_name="pytest", parameters={"targets": ("tests",)})
    denied = invocation.model_copy(update={"environment": {"SECRET": "value"}})
    with pytest.raises(ValueError, match="allowlisted"):
        build_command_plan(specification, denied)

    for reserved in ("LANG", "LC_ALL", "PATH"):
        with pytest.raises(ValidationError, match="reserved"):
            make_spec(environment_allowlist=(reserved,))


def test_default_registry_contains_only_server_defined_safe_tools() -> None:
    assert DEFAULT_COMMAND_REGISTRY.names == ("pytest", "ruff", "mypy")
    assert all(
        not DEFAULT_COMMAND_REGISTRY.get(name).network_enabled
        for name in DEFAULT_COMMAND_REGISTRY.names
    )
    with pytest.raises(ValueError, match="duplicate"):
        CommandRegistry((DEFAULT_COMMAND_REGISTRY.get("pytest"),) * 2)
    with pytest.raises(KeyError, match="unknown command"):
        DEFAULT_COMMAND_REGISTRY.get("shell")


def test_command_result_ceiling_is_bounded() -> None:
    assert MAX_COMMAND_OUTPUT_BYTES == 512 * 1024


def fixture_runner(tmp_path: Path, *, timeout_seconds: float = 2) -> CommandRunner:
    specification = CommandSpec(
        name="fixture",
        executable=sys.executable,
        argv_template=(str(FIXTURE), "{mode}"),
        parameters=(CommandParameter(name="mode"),),
        command_class=CommandClass.TEST,
        timeout_seconds=timeout_seconds,
        max_output_bytes=1024,
        environment_allowlist=("CI",),
    )
    return CommandRunner(
        CommandRegistry((specification,)),
        workspace_cwd=tmp_path,
        test_only=True,
        isolation_mode=IsolationMode.HOST,
    )


def fixture_invocation(mode: str) -> CommandInvocation:
    return CommandInvocation(spec_name="fixture", parameters={"mode": mode})


@pytest.mark.asyncio
async def test_runner_records_success_failure_cwd_and_elapsed_time(tmp_path: Path) -> None:
    runner = fixture_runner(tmp_path)

    success = await runner.run(fixture_invocation("success"))
    assert success.exit_code == 0
    assert success.stdout.strip() == str(tmp_path)
    assert success.duration_ms >= 0
    assert success.timed_out is False

    failure = await runner.run(fixture_invocation("failure"))
    assert failure.exit_code == 3
    assert "stdout failure" in failure.stdout
    assert "stderr failure" in failure.stderr


@pytest.mark.asyncio
async def test_runner_wraps_process_start_resource_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runner = fixture_runner(tmp_path)

    async def fail_start(*args: object, **kwargs: object) -> object:
        raise OSError("too many open files")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_start)
    with pytest.raises(CommandExecutionError, match="could not be started"):
        await runner.run(fixture_invocation("success"))


@pytest.mark.asyncio
async def test_runner_terminates_on_timeout_and_output_limit(tmp_path: Path) -> None:
    timed_out = await fixture_runner(tmp_path, timeout_seconds=0.05).run(
        fixture_invocation("timeout")
    )
    assert timed_out.timed_out is True
    assert timed_out.exit_code is not None

    flooded = await fixture_runner(tmp_path).run(fixture_invocation("flood"))
    assert flooded.truncated is True
    assert len(flooded.stdout.encode("utf-8")) <= 1024


@pytest.mark.asyncio
async def test_runner_passes_only_clean_environment_and_allowlisted_values(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PRP_TEST_SECRET", "must-not-reach-child")
    runner = fixture_runner(tmp_path)
    result = await runner.run(
        CommandInvocation(
            spec_name="fixture",
            parameters={"mode": "env"},
            environment={"CI": "1"},
        )
    )
    names = {line.split("=", 1)[0] for line in result.stdout.splitlines()}
    assert names == {"CI", "LANG", "LC_ALL", "PATH"}
    assert "PRP_TEST_SECRET" not in result.stdout


@pytest.mark.asyncio
async def test_runner_cancellation_kills_the_entire_process_group(tmp_path: Path) -> None:
    runner = fixture_runner(tmp_path)
    task = asyncio.create_task(runner.run(fixture_invocation("children")))
    child_pid_file = tmp_path / "child.pid"
    for _ in range(100):
        if child_pid_file.exists():
            break
        await asyncio.sleep(0.01)
    assert child_pid_file.exists()
    child_pid = int(child_pid_file.read_text(encoding="ascii"))

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(100):
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("cancelled command left a child process running")


@pytest.mark.asyncio
async def test_targeted_test_handler_returns_auditable_tool_result(tmp_path: Path) -> None:
    runner = fixture_runner(tmp_path)
    definition = build_targeted_test_definition(runner)
    call = ToolCall(
        call_id=new_tool_call_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        tool_name="run_targeted_test",
        effect=ToolEffect.COMMAND,
        arguments={"spec_name": "fixture", "parameters": {"mode": "success"}},
        status=ToolCallStatus.RUNNING,
        requested_at="2026-08-14T12:00:00+00:00",
    )
    result = await definition.handler(
        ExecutionContext(
            call=call,
            arguments=fixture_invocation("success"),
            workspace_id="ws-test",
        )
    )
    assert result.status is ToolCallStatus.SUCCEEDED
    assert result.exit_code == 0
    assert result.result is not None
