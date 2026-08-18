"""Capability and fail-closed readiness tests for the sandbox probe."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from prp_runtime.workspace.sandbox import (
    SandboxCapabilities,
    SandboxExecutionError,
    SandboxProbeDiagnostic,
    SandboxUnavailableError,
    _staged_probe_argv,
    build_bwrap_argv,
    default_runtime_roots,
    parse_bwrap_version,
    probe_bwrap,
    require_sandbox,
)


def _staged_result(
    argv: Sequence[str],
    *,
    script_stdout: str = '{"mount":true,"network":true,"pid":true}\n',
) -> subprocess.CompletedProcess[str]:
    if argv[1:] == ("--version",):
        return subprocess.CompletedProcess(
            argv, 0, stdout="bubblewrap 0.9.0\n", stderr=""
        )
    if argv[-1] == str(Path("/bin/true").resolve()):
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
    if any("prp-python-stage" in value for value in argv):
        return subprocess.CompletedProcess(
            argv, 0, stdout="prp-python-stage\n", stderr=""
        )
    if any("prp-runtime-stage" in value for value in argv):
        return subprocess.CompletedProcess(
            argv, 0, stdout="prp-runtime-stage\n", stderr=""
        )
    return subprocess.CompletedProcess(argv, 0, stdout=script_stdout, stderr="")


def test_parse_bwrap_version_accepts_only_bubblewrap_version_line() -> None:
    assert parse_bwrap_version("bubblewrap 0.9.0\n") == "0.9.0"
    assert parse_bwrap_version("warning\nbubblewrap 0.9.0\n") == "0.9.0"
    assert parse_bwrap_version("bwrap 0.9.0\n") is None


def test_probe_reports_full_capabilities_without_root_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Sequence[str]] = []

    def fake_which(executable: str) -> str:
        assert executable == "bwrap"
        return "/usr/bin/bwrap"

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        calls.append(argv)
        return _staged_result(argv)

    monkeypatch.setattr("prp_runtime.workspace.sandbox.shutil.which", fake_which)
    capabilities = probe_bwrap(runner=fake_runner)

    assert capabilities == SandboxCapabilities(
        backend="bubblewrap",
        available=True,
        version="0.9.0",
        supports_mount_isolation=True,
        supports_network_isolation=True,
        supports_process_isolation=True,
    )
    assert capabilities.ready is True
    assert calls[0] == ("bwrap", "--version")
    assert len(calls) == 6
    assert calls[1][-1] == str(Path("/bin/true").resolve())
    assert "--unshare-net" in calls[-1]
    assert "--unshare-pid" in calls[-1]
    assert calls[-1][calls[-1].index("/workspace") - 2] == "--ro-bind"


def test_probe_maps_observed_active_facts_and_fails_closed_on_partial_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_which(executable: str) -> str:
        assert executable == "bwrap"
        return "/usr/bin/bwrap"

    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        return _staged_result(
            argv,
            script_stdout='{"mount":true,"network":false,"pid":true}\n',
        )

    monkeypatch.setattr("prp_runtime.workspace.sandbox.shutil.which", fake_which)
    capabilities = probe_bwrap(runner=fake_runner)

    assert capabilities.available is True
    assert capabilities.supports_mount_isolation is True
    assert capabilities.supports_network_isolation is False
    assert capabilities.supports_process_isolation is True
    assert capabilities.ready is False
    assert capabilities.reason == (
        "bubblewrap active probe did not confirm isolation boundaries"
    )


def test_probe_staged_loader_failure_is_not_ready_without_exposing_host_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if argv[1:] == ("--version",):
            return subprocess.CompletedProcess(
                argv, 0, stdout="bubblewrap 0.9.0\n", stderr=""
            )
        if argv[-1] != str(Path("/bin/true").resolve()):
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="denied")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(
        "prp_runtime.workspace.sandbox.shutil.which", lambda _: "/usr/bin/bwrap"
    )
    capabilities = probe_bwrap(runner=fake_runner)

    assert capabilities.ready is False
    assert capabilities.reason == "bubblewrap staged loader probe failed"
    assert "/tmp" not in (capabilities.reason or "")
    assert capabilities.diagnostic is not None
    assert capabilities.diagnostic.stage == "elf"
    assert capabilities.diagnostic.category == "loader"
    assert capabilities.diagnostic.exit_code == 1
    assert capabilities.diagnostic.summary == "elf probe failed: loader"


@pytest.mark.parametrize(
    ("stage", "category"),
    (
        ("mount", "mount"),
        ("elf", "loader"),
        ("python", "loader"),
        ("runtime", "loader"),
        ("script", "script"),
    ),
)
def test_probe_staged_loader_classifies_safe_categories(
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    category: str,
) -> None:
    def fake_runner(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        if argv[1:] == ("--version",):
            return subprocess.CompletedProcess(
                argv, 0, stdout="bubblewrap 0.9.0\n", stderr=""
            )
        if stage == "mount" and argv[-1] == str(Path("/bin/true").resolve()):
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="denied")
        if stage == "elf" and argv[-1] == "--version":
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="denied")
        if stage == "python" and any("prp-python-stage" in value for value in argv):
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="denied")
        if stage == "runtime" and any("prp-runtime-stage" in value for value in argv):
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="denied")
        if stage == "script" and any("import json, os" in value for value in argv):
            return subprocess.CompletedProcess(argv, 23, stdout="", stderr="denied")
        return _staged_result(argv)

    monkeypatch.setattr(
        "prp_runtime.workspace.sandbox.shutil.which", lambda _: "/usr/bin/bwrap"
    )
    capabilities = probe_bwrap(runner=fake_runner)

    assert capabilities.ready is False
    assert capabilities.diagnostic is not None
    assert capabilities.diagnostic.stage == stage
    assert capabilities.diagnostic.category == category
    assert capabilities.diagnostic.exit_code == 23
    assert "denied" not in capabilities.diagnostic.summary


def test_staged_loader_argv_is_read_only_and_shell_free(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    for stage in ("mount", "elf", "python", "runtime", "script"):
        argv = _staged_probe_argv(stage, "/usr/bin/bwrap", workspace)  # type: ignore[arg-type]
        assert "--ro-bind" in argv
        assert "--bind" not in argv
        assert "--shell" not in argv
        assert argv[argv.index("--") + 1] != "sh"

    mount_argv = _staged_probe_argv("mount", "/usr/bin/bwrap", workspace)
    assert "--unshare-net" not in mount_argv
    assert "--unshare-pid" not in mount_argv


def test_probe_diagnostic_is_frozen_closed_and_bounded() -> None:
    diagnostic = SandboxProbeDiagnostic(
        stage="active",
        category="unknown",
        summary="active probe failed",
    )

    with pytest.raises(ValueError):
        diagnostic.summary = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError):
        SandboxProbeDiagnostic(
            stage="active",
            category="unknown",
            summary="active probe failed",
            stderr="secret",  # type: ignore[call-arg]
        )


def test_missing_bwrap_is_unavailable_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("prp_runtime.workspace.sandbox.shutil.which", lambda _: None)
    capabilities = probe_bwrap()

    assert capabilities.available is False
    assert capabilities.ready is False
    with pytest.raises(SandboxUnavailableError):
        require_sandbox(capabilities)


def test_staged_loader_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "prp_runtime.workspace.sandbox.shutil.which", lambda _: "/usr/bin/bwrap"
    )
    capabilities = probe_bwrap(runner=_staged_result)

    assert capabilities.backend == "bubblewrap"
    assert capabilities.available is True
    assert capabilities.version is not None
    assert capabilities.supports_mount_isolation is True
    assert capabilities.supports_network_isolation is True
    assert capabilities.supports_process_isolation is True
    assert capabilities.ready is True
    assert capabilities.diagnostic is None


def test_bwrap_argv_has_closed_mount_network_and_process_boundaries(tmp_path) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    workspace.mkdir()
    command = (str(runtime / "bin" / "fixture"), "--safe")

    argv = build_bwrap_argv(
        command,
        workspace,
        executable="/usr/bin/bwrap",
        environment={"CI": "1"},
        runtime_roots=(runtime,),
    )

    assert argv[0] == "/usr/bin/bwrap"
    assert "--clearenv" in argv
    assert "--unshare-user" in argv
    assert "--unshare-net" in argv
    assert "--unshare-pid" in argv
    assert "--die-with-parent" in argv
    ro_bind_args = tuple(
        tuple(argv[index : index + 3])
        for index in range(len(argv) - 2)
        if argv[index] == "--ro-bind"
    )
    bind_args = tuple(
        tuple(argv[index : index + 3])
        for index in range(len(argv) - 2)
        if argv[index] == "--bind"
    )
    assert ("--ro-bind", str(runtime), str(runtime)) in ro_bind_args
    assert ("--bind", str(workspace), "/workspace") in bind_args
    assert "--share-net" not in argv
    assert "--" in argv
    assert argv[argv.index("--") + 1 :] == command

    with pytest.raises(SandboxExecutionError, match="environment"):
        build_bwrap_argv(
            command,
            workspace,
            executable="/usr/bin/bwrap",
            environment={"PATH": "/tmp"},
            runtime_roots=(runtime,),
        )


def test_default_runtime_roots_are_narrow_and_existing() -> None:
    roots = default_runtime_roots()

    assert roots
    assert all(root.is_dir() for root in roots)
    assert Path("/") not in roots
    assert Path("/usr") not in roots
    assert Path("/usr/local") not in roots
    assert Path("/home") not in roots
    assert Path("/root") not in roots
    assert all(
        root.name in {"bin", "lib", "lib64"} or root == Path(sys.prefix).resolve()
        for root in roots
    )


def test_bwrap_argv_rejects_broad_missing_and_symlink_mounts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(
            (str(runtime / "fixture"),), workspace, runtime_roots=(Path("/usr"),)
        )
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(
            (str(runtime / "fixture"),), workspace, runtime_roots=(tmp_path / "missing",)
        )
    runtime_link = tmp_path / "runtime-link"
    runtime_link.symlink_to(runtime, target_is_directory=True)
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(
            (str(runtime / "fixture"),), workspace, runtime_roots=(runtime_link,)
        )


def test_bwrap_argv_rejects_workspace_overlap_symlink_and_unsafe_executable(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    runtime = tmp_path / "runtime"
    workspace.mkdir()
    runtime.mkdir()
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(
            (str(runtime / "fixture"),), workspace, runtime_roots=(workspace / "nested",)
        )
    workspace_link = tmp_path / "workspace-link"
    workspace_link.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(
            (str(runtime / "fixture"),), workspace_link, runtime_roots=(runtime,)
        )
    with pytest.raises(SandboxExecutionError):
        build_bwrap_argv(("fixture",), workspace, runtime_roots=(runtime,))
