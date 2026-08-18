"""Capability contract and active readiness probe for bubblewrap."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import ConfigDict, Field

from prp_runtime.domain.models import DomainModel
from prp_runtime.json_support import StrictJsonError, strict_json_loads

__all__ = [
    "BubblewrapBackend",
    "SandboxBackend",
    "SandboxCapabilities",
    "SandboxExecutionError",
    "SandboxProbeDiagnostic",
    "SandboxUnavailableError",
    "build_bwrap_argv",
    "default_runtime_roots",
    "parse_bwrap_version",
    "probe_bwrap",
    "require_sandbox",
]

_VERSION_RE = re.compile(r"^bubblewrap\s+([0-9][A-Za-z0-9.+_-]*)$")
_SANDBOX_WORKSPACE = "/workspace"
_SAFE_RUNTIME_ROOTS = (
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/usr/bin"),
    Path("/usr/lib"),
    Path("/usr/lib64"),
)
_PROBE_TIMEOUT_SECONDS = 5
_PROBE_OUTPUT_BYTES = 4096
_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]{0,63}$")
_RESERVED_ENVIRONMENT_NAMES = frozenset(("LANG", "LC_ALL", "PATH"))
_MOUNT_PROBE_COMMAND = "/bin/true"
_PYTHON_PROBE_SCRIPT = "import sys; sys.stdout.write('prp-python-stage\\n')"
_RUNTIME_PROBE_SCRIPT = (
    "import _json, encodings, sys; "
    "sys.stdout.write('prp-runtime-stage\\n')"
)
_ACTIVE_PROBE_SCRIPT = (
    "import json, os\n"
    "from pathlib import Path\n"
    "sentinel = Path('/workspace/sentinel')\n"
    "routes = Path('/proc/net/route').read_text(encoding='ascii').splitlines()\n"
    "print(json.dumps({\n"
    "    'mount': sentinel.is_file() and sentinel.read_text(encoding='ascii') == 'prp-probe\\n' "
    "        and not Path('/etc/passwd').exists(),\n"
    "    'network': len(routes) <= 1,\n"
    "    'pid': os.getpid() == 1,\n"
    "}, separators=(',', ':')))\n"
)

ProbeStage = Literal[
    "lookup",
    "version",
    "mount",
    "elf",
    "python",
    "runtime",
    "script",
    "active",
]
ProbeCategory = Literal[
    "unavailable",
    "namespace",
    "mount",
    "loader",
    "script",
    "timeout",
    "output_limit",
    "invalid_output",
    "isolation",
    "unknown",
]


class SandboxProbeDiagnostic(DomainModel):
    """Bounded, non-sensitive evidence about one failed probe stage."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    stage: ProbeStage
    category: ProbeCategory
    exit_code: int | None = None
    summary: str = Field(min_length=1, max_length=128)


class SandboxCapabilities(DomainModel):
    """Facts needed before CLOUD SANDBOXED execution may be admitted."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    backend: Literal["bubblewrap", "unavailable"]
    available: bool
    version: str | None = None
    supports_mount_isolation: bool = False
    supports_network_isolation: bool = False
    supports_process_isolation: bool = False
    reason: str | None = Field(default=None, max_length=256)
    diagnostic: SandboxProbeDiagnostic | None = None

    @property
    def ready(self) -> bool:
        """Whether the backend can truthfully be called an OS sandbox."""
        return (
            self.available
            and self.backend == "bubblewrap"
            and self.supports_mount_isolation
            and self.supports_network_isolation
            and self.supports_process_isolation
        )


class SandboxUnavailableError(RuntimeError):
    """Raised when a caller requests SANDBOXED execution without bwrap."""


class SandboxExecutionError(RuntimeError):
    """Raised when a command cannot be represented by a closed bwrap plan."""


class SandboxBackend(Protocol):
    """Probe-only contract consumed by readiness and future executors."""

    def probe(self) -> SandboxCapabilities:
        """Return local capability facts without requiring root or network."""

    def build_argv(
        self,
        command: Sequence[str],
        workspace_root: Path,
        *,
        environment: Mapping[str, str],
        runtime_roots: Sequence[Path],
    ) -> tuple[str, ...]:
        """Build a closed, shell-free bubblewrap command line."""


class _ProbeResult(Protocol):
    returncode: int
    stdout: str
    stderr: str


ProbeRunner = Callable[[Sequence[str]], _ProbeResult]


def parse_bwrap_version(output: str) -> str | None:
    """Parse the stable version line emitted by ``bwrap --version``."""
    for line in output.splitlines():
        match = _VERSION_RE.fullmatch(line.strip())
        if match is not None:
            return match.group(1)
    return None


def _run_probe(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(argv),
        check=False,
        capture_output=True,
        text=True,
        timeout=_PROBE_TIMEOUT_SECONDS,
    )


def _diagnostic(
    stage: ProbeStage,
    category: ProbeCategory,
    *,
    exit_code: int | None = None,
    summary: str,
) -> SandboxProbeDiagnostic:
    return SandboxProbeDiagnostic(
        stage=stage,
        category=category,
        exit_code=exit_code,
        summary=summary,
    )


def _classify_failure(text: str) -> ProbeCategory:
    """Classify known failure families without retaining process output."""
    lowered = text.casefold()
    if any(
        marker in lowered
        for marker in (
            "namespace",
            "unshare",
            "clone_new",
            "userns",
        )
    ):
        return "namespace"
    if any(
        marker in lowered
        for marker in (
            "mount",
            "bind",
            "tmpfs",
            "/proc",
            "dev/",
        )
    ):
        return "mount"
    if any(
        marker in lowered
        for marker in (
            "exec",
            "elf",
            "shared object",
            "dynamic linker",
            "ld-linux",
            "no such file or directory",
        )
    ):
        return "loader"
    if any(
        marker in lowered
        for marker in (
            "traceback",
            "syntaxerror",
            "nameerror",
            "importerror",
            "modulenotfounderror",
            "assertionerror",
        )
    ):
        return "script"
    return "unknown"


def _unavailable(
    reason: str,
    diagnostic: SandboxProbeDiagnostic,
) -> SandboxCapabilities:
    return SandboxCapabilities(
        backend="unavailable",
        available=False,
        reason=reason,
        diagnostic=diagnostic,
    )


def default_runtime_roots() -> tuple[Path, ...]:
    """Return narrow runtime mounts, never a broad home or filesystem mount."""
    roots: list[Path] = []

    def add_root(candidate: Path) -> None:
        if candidate.is_dir():
            if candidate in _SAFE_RUNTIME_ROOTS and candidate not in roots:
                roots.append(candidate)
            resolved = candidate.resolve()
            if resolved not in roots:
                roots.append(resolved)

    for root in _SAFE_RUNTIME_ROOTS:
        add_root(root)
    for candidate in (Path(sys.prefix), Path(sys.base_prefix)):
        if candidate == Path("/"):
            continue
        if candidate not in {Path("/usr"), Path("/usr/local")}:
            add_root(candidate)
        else:
            for relative in ("bin", "lib", "lib64"):
                add_root(candidate / relative)
    return tuple(roots)


def _validate_mount_root(root: Path, workspace_root: Path) -> Path:
    if not root.is_absolute() or root == Path("/"):
        raise SandboxExecutionError("sandbox runtime mount must be an absolute non-root path")
    resolved = root.resolve()
    if root.is_symlink():
        safe_targets = {candidate.resolve() for candidate in _SAFE_RUNTIME_ROOTS}
        if root not in _SAFE_RUNTIME_ROOTS or resolved not in safe_targets:
            raise SandboxExecutionError("sandbox runtime mount must be an absolute non-root path")
    if (
        resolved == workspace_root
        or workspace_root in resolved.parents
        or resolved in workspace_root.parents
    ):
        raise SandboxExecutionError("sandbox runtime mount overlaps the workspace slot")
    if resolved in {
        Path("/usr"),
        Path("/usr/local"),
        Path("/home"),
        Path("/root"),
        Path("/proc"),
        Path("/sys"),
        Path("/dev"),
    }:
        raise SandboxExecutionError("sandbox runtime mount is not an allowed runtime path")
    if not resolved.is_dir():
        raise SandboxExecutionError("sandbox runtime mount is unavailable")
    return root if root in _SAFE_RUNTIME_ROOTS else resolved


def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
    if not command:
        raise SandboxExecutionError("sandbox command must not be empty")
    values = tuple(command)
    for value in values:
        if not value or "\x00" in value:
            raise SandboxExecutionError("sandbox command contains an invalid argument")
    return values


def _runtime_closure_roots() -> tuple[Path, ...]:
    """Return the narrow runtime roots needed by the staged Python probes."""
    resolved_interpreter = Path(sys.executable).resolve()
    runtime_roots = list(default_runtime_roots())
    for candidate in (resolved_interpreter.parent, resolved_interpreter.parent.parent):
        if candidate in {Path("/"), Path("/usr"), Path("/usr/local")}:
            continue
        if candidate.is_dir() and candidate not in runtime_roots:
            runtime_roots.append(candidate)
    return tuple(runtime_roots)


def _readonly_probe_argv(
    command: Sequence[str],
    workspace_root: Path,
    *,
    executable: str,
    runtime_roots: Sequence[Path],
    unshare_network: bool = True,
    unshare_process: bool = True,
) -> tuple[str, ...]:
    argv = list(
        build_bwrap_argv(
            command,
            workspace_root,
            executable=executable,
            runtime_roots=runtime_roots,
        )
    )
    if not unshare_network:
        argv.remove("--unshare-net")
    if not unshare_process:
        argv.remove("--unshare-pid")
        argv.remove("--as-pid-1")
        for option in ("--proc", "--dev", "--tmpfs"):
            option_index = argv.index(option)
            del argv[option_index : option_index + 2]
    if "--unshare-user" not in argv:
        argv.insert(argv.index("--clearenv") + 1, "--unshare-user")
    bind_index = argv.index("--bind")
    argv[bind_index] = "--ro-bind"
    return tuple(argv)


def _active_probe_argv(executable: str, workspace_root: Path) -> tuple[str, ...]:
    """Build the legacy full probe argv with a read-only workspace."""
    return _staged_probe_argv("script", executable, workspace_root)


def _staged_probe_argv(
    stage: Literal["mount", "elf", "python", "runtime", "script"],
    executable: str,
    workspace_root: Path,
) -> tuple[str, ...]:
    """Build one bounded loader-triage variant without a shell or broad mount."""
    interpreter = Path(sys.executable).resolve()
    command: tuple[str, ...]
    runtime_roots: tuple[Path, ...]
    if stage == "mount":
        mount_executable = Path(_MOUNT_PROBE_COMMAND).resolve()
        command = (str(mount_executable),)
        mount_roots = [mount_executable.parent.resolve()]
        for library_root in (Path("/lib"), Path("/lib64")):
            if library_root.is_dir():
                if library_root not in mount_roots:
                    mount_roots.append(library_root)
                resolved_root = library_root.resolve()
                if resolved_root not in mount_roots:
                    mount_roots.append(resolved_root)
        runtime_roots = tuple(mount_roots)
    elif stage == "elf":
        command = (str(interpreter), "--version")
        runtime_roots = _runtime_closure_roots()
    elif stage == "python":
        command = (str(interpreter), "-S", "-c", _PYTHON_PROBE_SCRIPT)
        runtime_roots = _runtime_closure_roots()
    elif stage == "runtime":
        command = (str(interpreter), "-S", "-c", _RUNTIME_PROBE_SCRIPT)
        runtime_roots = _runtime_closure_roots()
    else:
        command = (str(interpreter), "-c", _ACTIVE_PROBE_SCRIPT)
        runtime_roots = _runtime_closure_roots()
    return _readonly_probe_argv(
        command,
        workspace_root,
        executable=executable,
        runtime_roots=runtime_roots,
        unshare_network=stage != "mount",
        unshare_process=stage != "mount",
    )


def _parse_active_probe(output: str) -> tuple[bool, bool, bool] | None:
    if len(output.encode("utf-8")) > _PROBE_OUTPUT_BYTES:
        return None
    try:
        value = strict_json_loads(output)
    except StrictJsonError:
        return None
    if not isinstance(value, dict):
        return None
    facts = (value.get("mount"), value.get("network"), value.get("pid"))
    if any(type(fact) is not bool for fact in facts):
        return None
    return facts  # type: ignore[return-value]


def _active_probe_summary(facts: tuple[bool, bool, bool]) -> str:
    names = ("mount", "network", "pid")
    missing = ",".join(name for name, fact in zip(names, facts) if not fact)
    return f"script probe reported partial isolation: {missing}"


def _probe_output_exceeds_limit(result: _ProbeResult) -> bool:
    return any(
        len(output.encode("utf-8")) > _PROBE_OUTPUT_BYTES
        for output in (result.stdout, result.stderr)
    )


def _staged_failure_category(
    stage: Literal["mount", "elf", "python", "runtime", "script"],
) -> ProbeCategory:
    if stage == "mount":
        return "mount"
    if stage in {"elf", "python", "runtime"}:
        return "loader"
    return "script"


def _staged_loader_probe(
    run: ProbeRunner,
    executable: str,
    workspace_root: Path,
) -> tuple[SandboxProbeDiagnostic | None, tuple[bool, bool, bool] | None]:
    """Run loader stages in order and retain only bounded stage facts."""
    stages: tuple[Literal["mount", "elf", "python", "runtime", "script"], ...] = (
        "mount",
        "elf",
        "python",
        "runtime",
        "script",
    )
    expected_output = {
        "python": "prp-python-stage\n",
        "runtime": "prp-runtime-stage\n",
    }
    for stage in stages:
        try:
            argv = _staged_probe_argv(stage, executable, workspace_root)
            result = run(argv)
        except subprocess.TimeoutExpired:
            return (
                _diagnostic(stage, "timeout", summary=f"{stage} probe timed out"),
                None,
            )
        except (OSError, SandboxExecutionError):
            category = _staged_failure_category(stage)
            return (
                _diagnostic(
                    stage,
                    category,
                    summary=f"{stage} probe failed: {category}",
                ),
                None,
            )
        if result.returncode != 0:
            observed_category = _classify_failure(result.stderr)
            category = (
                observed_category
                if observed_category != "unknown"
                else _staged_failure_category(stage)
            )
            return (
                _diagnostic(
                    stage,
                    category,
                    exit_code=result.returncode,
                    summary=f"{stage} probe failed: {category}",
                ),
                None,
            )
        if _probe_output_exceeds_limit(result):
            return (
                _diagnostic(
                    stage,
                    "output_limit",
                    exit_code=result.returncode,
                    summary=f"{stage} probe output exceeded limit",
                ),
                None,
            )
        if stage in expected_output and result.stdout != expected_output[stage]:
            return (
                _diagnostic(
                    stage,
                    "invalid_output",
                    exit_code=result.returncode,
                    summary=f"{stage} probe output was invalid",
                ),
                None,
            )
        if stage == "script":
            facts = _parse_active_probe(result.stdout)
            if facts is None:
                return (
                    _diagnostic(
                        stage,
                        "invalid_output",
                        exit_code=result.returncode,
                        summary="script probe output was invalid",
                    ),
                    None,
                )
            return None, facts
    return (
        _diagnostic(
            "script",
            "unknown",
            summary="staged loader probe did not reach the script stage",
        ),
        None,
    )


def build_bwrap_argv(
    command: Sequence[str],
    workspace_root: Path,
    *,
    executable: str = "bwrap",
    environment: Mapping[str, str] | None = None,
    runtime_roots: Sequence[Path] | None = None,
) -> tuple[str, ...]:
    """Build a bubblewrap argv with no shell, host home, or network access."""
    command_values = _validate_command(command)
    try:
        root_stat = workspace_root.lstat()
    except OSError as error:
        raise SandboxExecutionError("sandbox workspace is unavailable") from error
    if (
        not workspace_root.is_absolute()
        or workspace_root.is_symlink()
        or not workspace_root.is_dir()
    ):
        raise SandboxExecutionError("sandbox workspace must be an absolute directory")
    del root_stat
    resolved_workspace = workspace_root.resolve()
    mounts = tuple(runtime_roots or default_runtime_roots())
    resolved_mounts = tuple(
        dict.fromkeys(_validate_mount_root(root, resolved_workspace) for root in mounts)
    )
    executable_path = Path(command_values[0])
    if not executable_path.is_absolute():
        raise SandboxExecutionError("sandbox executable must be an absolute path")
    if not any(
        executable_path == root or root in executable_path.parents
        for root in resolved_mounts
    ):
        raise SandboxExecutionError("sandbox executable is outside read-only runtime mounts")
    for name, value in (environment or {}).items():
        if (
            _ENV_NAME_RE.fullmatch(name) is None
            or name in _RESERVED_ENVIRONMENT_NAMES
            or "\x00" in value
        ):
            raise SandboxExecutionError("sandbox environment contains an invalid value")
    argv: list[str] = [executable]
    argv.extend(("--die-with-parent", "--new-session", "--clearenv"))
    argv.extend(("--unshare-user", "--unshare-net", "--unshare-pid", "--as-pid-1"))
    argv.extend(("--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp"))
    for root in resolved_mounts:
        argv.extend(("--dir", str(root)))
        argv.extend(("--ro-bind", str(root), str(root)))
    argv.extend(("--bind", str(resolved_workspace), _SANDBOX_WORKSPACE))
    argv.extend(("--chdir", _SANDBOX_WORKSPACE))
    argv.extend(
        (
            "--setenv",
            "LANG",
            "C",
            "--setenv",
            "LC_ALL",
            "C",
            "--setenv",
            "PATH",
            "",
            "--unsetenv",
            "PWD",
        )
    )
    for name, value in sorted((environment or {}).items()):
        argv.extend(("--setenv", name, value))
    argv.extend(("--", *command_values))
    return tuple(argv)


def probe_bwrap(
    executable: str = "bwrap",
    *,
    runner: ProbeRunner | None = None,
) -> SandboxCapabilities:
    """Probe bubblewrap version and actual mount/network/process isolation."""
    resolved_executable = shutil.which(executable)
    if resolved_executable is None:
        return _unavailable(
            "bubblewrap executable is unavailable",
            _diagnostic(
                "lookup",
                "unavailable",
                summary="bubblewrap executable is unavailable",
            ),
        )
    run = _run_probe if runner is None else runner
    try:
        result = run((executable, "--version"))
    except subprocess.TimeoutExpired:
        return _unavailable(
            "bubblewrap capability probe timed out",
            _diagnostic("version", "timeout", summary="version probe timed out"),
        )
    except OSError:
        return _unavailable(
            "bubblewrap capability probe failed",
            _diagnostic("version", "unknown", summary="version probe failed"),
        )
    if result.returncode != 0:
        category = _classify_failure(result.stderr)
        return _unavailable(
            "bubblewrap capability probe returned an error",
            _diagnostic(
                "version",
                category,
                exit_code=result.returncode,
                summary=f"version probe failed: {category}",
            ),
        )
    if _probe_output_exceeds_limit(result):
        return _unavailable(
            "bubblewrap capability probe output exceeded limit",
            _diagnostic(
                "version",
                "output_limit",
                exit_code=result.returncode,
                summary="version probe output exceeded limit",
            ),
        )
    version = parse_bwrap_version(result.stdout)
    if version is None:
        version = parse_bwrap_version(result.stderr)
    if version is None:
        return _unavailable(
            "bubblewrap capability version is unrecognized",
            _diagnostic(
                "version",
                "invalid_output",
                exit_code=result.returncode,
                summary="version probe output was invalid",
            ),
        )
    try:
        with tempfile.TemporaryDirectory(prefix="prp-bwrap-probe-") as temporary_root:
            workspace_root = Path(temporary_root)
            (workspace_root / "sentinel").write_text("prp-probe\n", encoding="ascii")
            staged_diagnostic, facts = _staged_loader_probe(
                run, resolved_executable, workspace_root
            )
    except OSError:
        return SandboxCapabilities(
            backend="bubblewrap",
            available=True,
            version=version,
            reason="bubblewrap staged probe failed",
            diagnostic=_diagnostic("mount", "unknown", summary="mount probe failed"),
        )
    if staged_diagnostic is not None:
        return SandboxCapabilities(
            backend="bubblewrap",
            available=True,
            version=version,
            reason="bubblewrap staged loader probe failed",
            diagnostic=staged_diagnostic,
        )
    assert facts is not None
    supports_mount, supports_network, supports_process = facts
    reason = None
    if not all(facts):
        reason = "bubblewrap active probe did not confirm isolation boundaries"
    diagnostic = None
    if reason is not None:
        diagnostic = _diagnostic(
            "script",
            "isolation",
            exit_code=0,
            summary=_active_probe_summary(facts),
        )
    return SandboxCapabilities(
        backend="bubblewrap",
        available=True,
        version=version,
        supports_mount_isolation=supports_mount,
        supports_network_isolation=supports_network,
        supports_process_isolation=supports_process,
        reason=reason,
        diagnostic=diagnostic,
    )


class BubblewrapBackend:
    """Bubblewrap backend that separates capability probing from argv building."""

    def __init__(self, executable: str = "bwrap") -> None:
        self._executable = executable

    def probe(self) -> SandboxCapabilities:
        return probe_bwrap(self._executable)

    def build_argv(
        self,
        command: Sequence[str],
        workspace_root: Path,
        *,
        environment: Mapping[str, str],
        runtime_roots: Sequence[Path],
    ) -> tuple[str, ...]:
        capabilities = self.probe()
        require_sandbox(capabilities)
        executable = shutil.which(self._executable)
        if executable is None:
            raise SandboxUnavailableError("bubblewrap executable is unavailable")
        return build_bwrap_argv(
            command,
            workspace_root,
            executable=executable,
            environment=environment,
            runtime_roots=runtime_roots,
        )


def require_sandbox(capabilities: SandboxCapabilities) -> None:
    """Raise rather than silently running a SANDBOXED request on the host."""
    if not capabilities.ready:
        reason = capabilities.reason or "a usable bubblewrap sandbox is unavailable"
        if capabilities.diagnostic is not None:
            reason = f"{reason}: {capabilities.diagnostic.summary}"
        raise SandboxUnavailableError(
            reason
        )
