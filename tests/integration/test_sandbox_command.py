"""L3 command isolation evidence; requires a qualified sandbox runner."""

import json
import sys
from pathlib import Path

import pytest

from prp_runtime.domain.enums import IsolationMode
from prp_runtime.policy.models import CommandClass
from prp_runtime.tools.command import (
    CommandArgumentKind,
    CommandInvocation,
    CommandParameter,
    CommandRegistry,
    CommandRunner,
    CommandSpec,
)
from prp_runtime.workspace.sandbox import default_runtime_roots

FIXTURE = Path(__file__).parents[1] / "fixtures" / "command_fixture.py"


def sandbox_fixture_runner(
    tmp_path: Path,
    *,
    argv_template: tuple[str, ...],
    extra_runtime_roots: tuple[Path, ...] = (),
) -> CommandRunner:
    workspace = tmp_path / "sandbox-workspace"
    workspace.mkdir()
    specification = CommandSpec(
        name="sandbox-fixture",
        executable=sys.executable,
        argv_template=argv_template,
        parameters=(CommandParameter(name="mode", kind=CommandArgumentKind.TOKEN),),
        command_class=CommandClass.TEST,
        environment_allowlist=("CI",),
        timeout_seconds=15,
        max_output_bytes=32_768,
    )
    return CommandRunner(
        CommandRegistry((specification,)),
        workspace_cwd=workspace,
        test_only=True,
        isolation_mode=IsolationMode.SANDBOXED,
        runtime_roots=(
            *default_runtime_roots(),
            FIXTURE.parent,
            *extra_runtime_roots,
        ),
    )


@pytest.mark.asyncio
async def test_sandbox_environment_is_allowlisted_and_network_isolated(tmp_path: Path) -> None:
    runner = sandbox_fixture_runner(tmp_path, argv_template=(str(FIXTURE), "{mode}"))
    environment = await runner.run(
        CommandInvocation(
            spec_name="sandbox-fixture",
            parameters={"mode": "env"},
            environment={"CI": "1"},
        )
    )
    names = {line.split("=", 1)[0] for line in environment.stdout.splitlines()}
    assert names == {"CI", "LANG", "LC_ALL", "PATH", "PWD"}
    assert "PWD=/workspace" in environment.stdout.splitlines()
    assert "HTTP_PROXY" not in environment.stdout
    assert "PRP_TEST_SECRET" not in environment.stdout

    network = await runner.run(
        CommandInvocation(
            spec_name="sandbox-fixture",
            parameters={"mode": "sandbox_network"},
        )
    )
    assert network.exit_code == 0
    assert json.loads(network.stdout) == {"network": "isolated"}


@pytest.mark.asyncio
async def test_sandbox_cannot_read_unmounted_sentinel(tmp_path: Path) -> None:
    sentinel = tmp_path / "host-sentinel"
    sentinel.write_text("host-only", encoding="ascii")
    runner = sandbox_fixture_runner(
        tmp_path,
        argv_template=(str(FIXTURE), "{mode}", str(sentinel)),
    )

    result = await runner.run(
        CommandInvocation(
            spec_name="sandbox-fixture",
            parameters={"mode": "sandbox_sentinel"},
        )
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "sentinel-unmounted"
    assert sentinel.read_text(encoding="ascii") == "host-only"


@pytest.mark.asyncio
async def test_sandbox_runtime_mount_is_read_only(tmp_path: Path) -> None:
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    target = runtime_root / "write.txt"
    runner = sandbox_fixture_runner(
        tmp_path,
        argv_template=(str(FIXTURE), "{mode}", str(target)),
        extra_runtime_roots=(runtime_root,),
    )

    result = await runner.run(
        CommandInvocation(
            spec_name="sandbox-fixture",
            parameters={"mode": "sandbox_write"},
        )
    )

    assert result.exit_code == 0
    assert result.stdout.strip() == "runtime-read-only"
    assert target.exists() is False
