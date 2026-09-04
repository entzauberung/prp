"""Server-owned, shell-free command specifications and argv expansion."""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import signal
import time
from collections.abc import Collection, Mapping
from enum import StrEnum, unique
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, cast

from pydantic import Field, StrictBool, StringConstraints, field_validator, model_validator

from prp_runtime.domain.enums import IsolationMode, ToolCallStatus, ToolEffect
from prp_runtime.domain.models import DomainModel, ErrorCategory, ErrorInfo
from prp_runtime.domain.values import utc_now
from prp_runtime.policy.models import CommandClass
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES, ToolResult
from prp_runtime.tools.registry import ToolDefinition, ToolHandler
from prp_runtime.workspace.sandbox import (
    BubblewrapBackend,
    SandboxBackend,
    SandboxExecutionError,
    default_runtime_roots,
    require_sandbox,
)

__all__ = [
    "CommandArgumentKind",
    "CommandCwd",
    "CommandInvocation",
    "CommandExecutionError",
    "CommandParameter",
    "CommandPlan",
    "CommandRegistry",
    "CommandResult",
    "CommandRunner",
    "CommandSpec",
    "DEFAULT_COMMAND_REGISTRY",
    "DEFAULT_COMMAND_SPECS",
    "MAX_COMMAND_OUTPUT_BYTES",
    "build_command_argv",
    "build_command_plan",
    "build_targeted_test_definition",
    "expand_argv",
]

MAX_COMMAND_OUTPUT_BYTES = 512 * 1024
MAX_COMMAND_ARGUMENTS = 64
_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_PLACEHOLDER_RE = re.compile(r"^\{([a-z][a-z0-9_-]{0,31})\}$")
_SHELL_SYNTAX = frozenset(";&|$`()<>`")
_RESERVED_ENVIRONMENT_NAMES = frozenset(("LANG", "LC_ALL", "PATH"))

CommandName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[a-z][a-z0-9_.-]{0,63}$",
    ),
]
ParameterName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=32,
        pattern=r"^[a-z][a-z0-9_-]{0,31}$",
    ),
]


@unique
class CommandArgumentKind(StrEnum):
    """Validation policy for one user-provided argv placeholder."""

    TOKEN = "TOKEN"
    PATH = "PATH"


@unique
class CommandCwd(StrEnum):
    """The only cwd currently exposed to a command executor."""

    WORKSPACE = "WORKSPACE"


class CommandParameter(DomainModel):
    """One named, bounded value that may be expanded into argv."""

    name: ParameterName
    kind: CommandArgumentKind = CommandArgumentKind.TOKEN
    required: StrictBool = True
    multiple: StrictBool = False
    max_items: int = Field(default=1, gt=0, le=MAX_COMMAND_ARGUMENTS)

    @model_validator(mode="after")
    def _single_values_have_one_slot(self) -> CommandParameter:
        if not self.multiple and self.max_items != 1:
            raise ValueError("a single command parameter must have max_items=1")
        return self


class CommandSpec(DomainModel):
    """A server-owned command template with no shell interpolation."""

    name: CommandName
    executable: str
    argv_template: tuple[str, ...] = ()
    parameters: tuple[CommandParameter, ...] = ()
    command_class: CommandClass
    effect: ToolEffect = ToolEffect.COMMAND
    cwd: CommandCwd = CommandCwd.WORKSPACE
    environment_allowlist: tuple[str, ...] = ()
    timeout_seconds: float = Field(gt=0, le=60)
    max_output_bytes: int = Field(gt=0, le=MAX_COMMAND_OUTPUT_BYTES)
    network_enabled: StrictBool = False

    @field_validator("executable")
    @classmethod
    def _executable_is_server_owned(cls, value: str) -> str:
        if (
            not value
            or "\\" in value
            or any(char.isspace() for char in value)
            or ("/" in value and not value.startswith("/"))
        ):
            raise ValueError("executable must be a bare server-owned name")
        _validate_token(value, label="executable")
        return value

    @field_validator("environment_allowlist")
    @classmethod
    def _environment_names_are_closed(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(value) != len(set(value)):
            raise ValueError("environment allowlist must not contain duplicates")
        for name in value:
            if _ENV_NAME_RE.fullmatch(name) is None:
                raise ValueError("environment allowlist contains an invalid name")
            if name in _RESERVED_ENVIRONMENT_NAMES:
                raise ValueError("environment allowlist contains a reserved name")
        return value

    @model_validator(mode="after")
    def _template_is_closed(self) -> CommandSpec:
        if self.effect is not ToolEffect.COMMAND:
            raise ValueError("command specifications must declare the COMMAND effect")
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("command parameters must not contain duplicates")
        parameter_names = set(names)
        for token in self.argv_template:
            match = _PLACEHOLDER_RE.fullmatch(token)
            if match is None:
                if "{" in token or "}" in token:
                    raise ValueError("argv placeholders must occupy one complete token")
                _validate_token(token, label="argv template token")
                continue
            if match.group(1) not in parameter_names:
                raise ValueError("argv template references an unknown parameter")
        return self


class CommandInvocation(DomainModel):
    """Untrusted request data naming a registered command and its parameters."""

    spec_name: CommandName
    parameters: dict[str, str | tuple[str, ...]] = Field(default_factory=dict)
    environment: dict[str, str] = Field(default_factory=dict)

    @field_validator("environment")
    @classmethod
    def _environment_values_are_bounded(cls, value: dict[str, str]) -> dict[str, str]:
        for name, content in value.items():
            if _ENV_NAME_RE.fullmatch(name) is None:
                raise ValueError("command environment contains an invalid name")
            _validate_token(content, label="environment value")
        return value


class CommandPlan(DomainModel):
    """Expanded execution facts for a later, isolated command executor."""

    spec_name: CommandName
    argv: tuple[str, ...]
    cwd: CommandCwd
    environment: dict[str, str]
    timeout_seconds: float
    max_output_bytes: int
    network_enabled: StrictBool
    command_class: CommandClass


class CommandResult(DomainModel):
    """Bounded result facts produced by the future command executor."""

    exit_code: int | None
    stdout: str = Field(max_length=MAX_COMMAND_OUTPUT_BYTES)
    stderr: str = Field(max_length=MAX_COMMAND_OUTPUT_BYTES)
    truncated: StrictBool = False
    timed_out: StrictBool = False
    cancelled: StrictBool = False
    duration_ms: int = Field(ge=0)


class CommandExecutionError(RuntimeError):
    """A registered command cannot be started or safely collected."""


class CommandRunner:
    """Execute one registered command below a server-owned workspace cwd."""

    def __init__(
        self,
        registry: CommandRegistry,
        *,
        workspace_cwd: Path,
        test_only: bool = False,
        isolation_mode: IsolationMode = IsolationMode.SANDBOXED,
        sandbox_backend: SandboxBackend | None = None,
        runtime_roots: Collection[Path] | None = None,
    ) -> None:
        try:
            workspace_stat = workspace_cwd.lstat()
        except OSError as error:
            raise CommandExecutionError("command workspace is unavailable") from error
        if (
            not workspace_cwd.is_absolute()
            or workspace_cwd.is_symlink()
            or not workspace_cwd.is_dir()
        ):
            raise CommandExecutionError(
                "command workspace must be an absolute directory"
            )
        del workspace_stat
        self._registry = registry
        self._workspace_cwd = workspace_cwd
        self._test_only = test_only
        self._isolation_mode = isolation_mode
        self._runtime_roots = tuple(runtime_roots or default_runtime_roots())
        self._sandbox_backend = sandbox_backend
        if isolation_mode is IsolationMode.SANDBOXED:
            self._sandbox_backend = sandbox_backend or BubblewrapBackend()
            require_sandbox(self._sandbox_backend.probe())

    async def run(self, invocation: CommandInvocation) -> CommandResult:
        """Run a fully expanded command with bounded output and cancellation."""
        specification = self._registry.get(invocation.spec_name)
        if self._test_only and specification.command_class is not CommandClass.TEST:
            raise CommandExecutionError("targeted test tool requires a TEST command")
        plan = build_command_plan(specification, invocation)
        executable = _resolve_executable(specification.executable)
        argv = (executable, *plan.argv[1:])
        environment = {"LANG": "C", "LC_ALL": "C", "PATH": ""}
        environment.update(plan.environment)
        process_cwd: Path | None = self._workspace_cwd
        process_environment = environment
        if self._isolation_mode is IsolationMode.SANDBOXED:
            if plan.network_enabled:
                raise CommandExecutionError("sandboxed commands cannot enable network access")
            if self._sandbox_backend is None:
                raise CommandExecutionError("sandbox backend is unavailable")
            try:
                argv = self._sandbox_backend.build_argv(
                    argv,
                    self._workspace_cwd,
                    environment=plan.environment,
                    runtime_roots=self._runtime_roots,
                )
            except SandboxExecutionError as error:
                raise CommandExecutionError(str(error)) from error
            process_cwd = None
            process_environment = {"PATH": ""}
        started = time.monotonic()
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=process_cwd,
                env=process_environment,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except (FileNotFoundError, PermissionError) as error:
            raise CommandExecutionError("registered command is unavailable") from error
        except OSError as error:
            raise CommandExecutionError("registered command could not be started") from error

        output_limit_reached = asyncio.Event()
        stdout_task = asyncio.create_task(
            _read_bounded(process.stdout, plan.max_output_bytes, output_limit_reached)
        )
        stderr_task = asyncio.create_task(
            _read_bounded(process.stderr, plan.max_output_bytes, output_limit_reached)
        )
        wait_task = asyncio.create_task(process.wait())
        limit_task = asyncio.create_task(output_limit_reached.wait())
        timed_out = False
        cancelled = False
        try:
            done, _ = await asyncio.wait(
                (wait_task, limit_task),
                timeout=plan.timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                timed_out = True
                await _stop_process_group(process)
            elif limit_task in done and process.returncode is None:
                await _stop_process_group(process)
            await _reap_process(process, wait_task)
        except asyncio.CancelledError:
            cancelled = True
            await _stop_process_group(process)
            raise
        finally:
            if not wait_task.done():
                await _reap_process(process, wait_task)
            limit_task.cancel()
            await asyncio.gather(limit_task, return_exceptions=True)
            await _finish_output_tasks(stdout_task, stderr_task)
        stdout, stdout_truncated = _output_task_result(stdout_task)
        stderr, stderr_truncated = _output_task_result(stderr_task)
        stdout = _redact_local_paths(stdout, plan.cwd)
        stderr = _redact_local_paths(stderr, plan.cwd)
        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            truncated=(
                output_limit_reached.is_set() or stdout_truncated or stderr_truncated
            ),
            timed_out=timed_out,
            cancelled=cancelled,
            duration_ms=max(0, round((time.monotonic() - started) * 1000)),
        )


def build_targeted_test_definition(runner: CommandRunner) -> ToolDefinition:
    """Build the policy-classified handler for registered TEST commands."""

    async def handler(context: object) -> ToolResult:
        if not isinstance(context, ExecutionContext):
            raise TypeError("run_targeted_test requires an execution context")
        if not isinstance(context.arguments, CommandInvocation):
            raise TypeError("run_targeted_test received invalid arguments")
        result = await runner.run(context.arguments)
        output = result.stdout
        if result.stderr:
            output = f"{output}\n{result.stderr}" if output else result.stderr
        output = output[:MAX_TOOL_OUTPUT_BYTES]
        if result.timed_out:
            return ToolResult.from_call(
                context.call,
                status=ToolCallStatus.FAILED,
                error=ErrorInfo(category=ErrorCategory.TIMEOUT, message="command timed out"),
                result=result.model_dump(mode="json"),
                output=output,
                truncated=result.truncated,
                exit_code=result.exit_code,
                completed_at=utc_now(),
            )
        if result.exit_code != 0:
            return ToolResult.from_call(
                context.call,
                status=ToolCallStatus.FAILED,
                error=ErrorInfo(
                    category=ErrorCategory.VERIFICATION_FAILED,
                    message="targeted command failed",
                ),
                result=result.model_dump(mode="json"),
                output=output,
                truncated=result.truncated,
                exit_code=result.exit_code,
                completed_at=utc_now(),
            )
        return ToolResult.from_call(
            context.call,
            status=ToolCallStatus.SUCCEEDED,
            result=result.model_dump(mode="json"),
            output=output,
            truncated=result.truncated,
            exit_code=result.exit_code,
            completed_at=utc_now(),
        )

    return ToolDefinition(
        name="run_targeted_test",
        description="Run one server-owned targeted verification command.",
        effect=ToolEffect.COMMAND,
        argument_model=CommandInvocation,
        handler=cast(ToolHandler, handler),
    )


class CommandRegistry:
    """Immutable server-owned lookup of command specifications."""

    __slots__ = ("_specs",)
    _specs: Mapping[str, CommandSpec]

    def __init__(self, specifications: Collection[CommandSpec] = ()) -> None:
        by_name: dict[str, CommandSpec] = {}
        for specification in specifications:
            if specification.name in by_name:
                raise ValueError(f"duplicate command specification: {specification.name}")
            by_name[specification.name] = specification
        object.__setattr__(self, "_specs", MappingProxyType(by_name))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def get(self, name: str) -> CommandSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"unknown command specification: {name}") from error

    def __contains__(self, name: object) -> bool:
        return name in self._specs


def expand_argv(
    specification: CommandSpec,
    parameters: Mapping[str, str | tuple[str, ...]] | CommandInvocation,
) -> tuple[str, ...]:
    """Expand validated named values into an argv tuple, never a shell string."""
    values: Mapping[str, str | tuple[str, ...]]
    if isinstance(parameters, CommandInvocation):
        if parameters.spec_name != specification.name:
            raise ValueError("command invocation does not name the supplied specification")
        values = parameters.parameters
    else:
        values = parameters
    expected = {parameter.name: parameter for parameter in specification.parameters}
    unknown = set(values) - set(expected)
    if unknown:
        raise ValueError("command invocation contains an unknown parameter")
    missing = {
        name for name, parameter in expected.items() if parameter.required and name not in values
    }
    if missing:
        raise ValueError("command invocation is missing a required parameter")

    argv: list[str] = [specification.executable]
    for token in specification.argv_template:
        match = _PLACEHOLDER_RE.fullmatch(token)
        if match is None:
            argv.append(token)
            continue
        parameter = expected[match.group(1)]
        expanded = _expand_parameter(parameter, values.get(parameter.name))
        argv.extend(expanded)
    if len(argv) > MAX_COMMAND_ARGUMENTS:
        raise ValueError("command invocation expands to too many arguments")
    return tuple(argv)


def build_command_argv(
    specification: CommandSpec,
    parameters: Mapping[str, str | tuple[str, ...]] | CommandInvocation,
) -> tuple[str, ...]:
    """Explicit-name alias for the shell-free argv expansion function."""
    return expand_argv(specification, parameters)


def build_command_plan(
    specification: CommandSpec, invocation: CommandInvocation
) -> CommandPlan:
    """Validate environment scope and return all executor-owned command facts."""
    if invocation.spec_name != specification.name:
        raise ValueError("command invocation does not name the supplied specification")
    unknown_environment = set(invocation.environment) - set(specification.environment_allowlist)
    if unknown_environment:
        raise ValueError("command invocation contains a non-allowlisted environment name")
    return CommandPlan(
        spec_name=specification.name,
        argv=expand_argv(specification, invocation),
        cwd=specification.cwd,
        environment=invocation.environment,
        timeout_seconds=specification.timeout_seconds,
        max_output_bytes=specification.max_output_bytes,
        network_enabled=specification.network_enabled,
        command_class=specification.command_class,
    )


def _expand_parameter(
    parameter: CommandParameter, value: str | tuple[str, ...] | None
) -> tuple[str, ...]:
    if value is None:
        return ()
    values = value if isinstance(value, tuple) else (value,)
    if not parameter.multiple and len(values) != 1:
        raise ValueError("single command parameter received multiple values")
    if len(values) > parameter.max_items:
        raise ValueError("command parameter has too many values")
    for item in values:
        _validate_token(item, label="command parameter")
        if item.startswith("-"):
            raise ValueError("command parameters must not inject option tokens")
        if parameter.kind is CommandArgumentKind.PATH:
            _validate_workspace_path(item)
    return values


def _validate_token(value: str, *, label: str) -> None:
    if not value or "\x00" in value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise ValueError(f"{label} contains an invalid control character")
    if any(char in _SHELL_SYNTAX for char in value):
        raise ValueError(f"{label} contains shell syntax")


def _resolve_executable(value: str) -> str:
    if value.startswith("/"):
        candidate = Path(value)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return value
    else:
        resolved = shutil.which(value)
        if resolved is not None:
            return resolved
    raise CommandExecutionError("registered command is unavailable")


async def _read_bounded(
    stream: asyncio.StreamReader | None,
    limit: int,
    output_limit_reached: asyncio.Event,
) -> tuple[str, bool]:
    if stream is None:
        raise CommandExecutionError("command output pipe is unavailable")
    chunks: list[bytes] = []
    size = 0
    truncated = False
    while True:
        chunk = await stream.read(8_192)
        if not chunk:
            break
        remaining = limit - size
        if remaining <= 0:
            truncated = True
            output_limit_reached.set()
            continue
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            size += remaining
            truncated = True
            output_limit_reached.set()
            continue
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace"), truncated


async def _finish_output_tasks(
    *tasks: asyncio.Task[tuple[str, bool]],
) -> None:
    """Bound pipe draining when a detached descendant keeps stdout open."""
    try:
        await asyncio.wait_for(asyncio.gather(*tasks), timeout=1.0)
    except TimeoutError:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def _output_task_result(task: asyncio.Task[tuple[str, bool]]) -> tuple[str, bool]:
    if task.cancelled():
        return "", True
    return task.result()


async def _stop_process_group(process: asyncio.subprocess.Process) -> None:
    if process.returncode is not None:
        return
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group = None
    if process_group is not None:
        try:
            os.killpg(process_group, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.terminate()
    except ProcessLookupError:
        pass


async def _reap_process(
    process: asyncio.subprocess.Process, wait_task: asyncio.Task[int]
) -> None:
    """Reap a stopped process without leaving an unbounded wait on its pipes."""
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=1.0)
        return
    except TimeoutError:
        pass
    try:
        process_group = os.getpgid(process.pid)
    except ProcessLookupError:
        process_group = None
    if process_group is not None:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
    try:
        process.kill()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(asyncio.shield(wait_task), timeout=1.0)
    except TimeoutError as error:
        raise CommandExecutionError("command process did not terminate") from error


def _validate_workspace_path(value: str) -> None:
    if (
        not value
        or value.startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", value)
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("command path must be workspace-relative")


def _parameter(name: str, *, multiple: bool = False, max_items: int = 16) -> CommandParameter:
    return CommandParameter(
        name=name,
        kind=CommandArgumentKind.PATH,
        multiple=multiple,
        max_items=max_items,
    )


DEFAULT_COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        name="pytest",
        executable="python",
        argv_template=(
            "-B",
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "{targets}",
        ),
        parameters=(_parameter("targets", multiple=True),),
        command_class=CommandClass.TEST,
        timeout_seconds=60,
        max_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    ),
    CommandSpec(
        name="ruff",
        executable="ruff",
        argv_template=("check", "{targets}"),
        parameters=(_parameter("targets", multiple=True),),
        command_class=CommandClass.LINT,
        timeout_seconds=60,
        max_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    ),
    CommandSpec(
        name="mypy",
        executable="mypy",
        argv_template=("{targets}",),
        parameters=(_parameter("targets", multiple=True),),
        command_class=CommandClass.LINT,
        timeout_seconds=60,
        max_output_bytes=MAX_COMMAND_OUTPUT_BYTES,
    ),
)

DEFAULT_COMMAND_REGISTRY = CommandRegistry(DEFAULT_COMMAND_SPECS)


def _redact_local_paths(text: str, root: object) -> str:
    """Strip the authorized workspace cwd from public command observations."""
    if not text or root is None:
        return text
    value = str(root)
    if not value:
        return text
    return text.replace(value, "").replace(value.replace("\\", "/"), "")
