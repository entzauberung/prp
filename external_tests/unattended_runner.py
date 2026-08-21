"""Single-stage, secret-safe launcher for bounded external validation."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

if __package__:
    from .credential_loader import (
        CredentialError,
        CredentialSet,
        PROFILE_CONTRACTS,
        load_credentials,
    )
    from .result_ledger import LedgerStore
    from .capability_ledger import CapabilityStore
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from credential_loader import (  # type: ignore[no-redef]
        CredentialError,
        CredentialSet,
        PROFILE_CONTRACTS,
        load_credentials,
    )
    from result_ledger import LedgerStore  # type: ignore[no-redef]
    from capability_ledger import CapabilityStore  # type: ignore[no-redef]


REPO_ROOT = Path(__file__).resolve().parent.parent
PREFLIGHT_TARGETS = (
    "external_tests/test_live_deepseek.py",
    "external_tests/test_live_luna_claude.py",
)
STRATEGY_TARGETS = ("external_tests/test_live_strategies.py",)
AGENT_TARGETS = ("external_tests/test_live_agent.py",)
CAPABILITY_TARGETS = ("external_tests/test_live_capabilities.py",)
PROTOCOL_TARGETS = (
    "external_tests/test_live_protocols.py",
    "external_tests/test_live_lifecycle.py",
)
LOG_ROOT = Path("/home/bruce/文档/prp测试日志/real-gap-closure")
HARNESS_LOG = LOG_ROOT / "00-harness.log"
DEFAULT_RESULT_FILE = LOG_ROOT / "10-providers.jsonl"
STAGE_RESULT_FILES = MappingProxyType(
    {
        "protocols": LOG_ROOT / "20-protocols.jsonl",
        "strategy": LOG_ROOT / "30-strategies.jsonl",
        "strategies": LOG_ROOT / "30-strategies.jsonl",
        "agent": LOG_ROOT / "40-agent.jsonl",
        "regression": LOG_ROOT / "50-regression.log",
    }
)
DEFAULT_CAPABILITY_FILE = LOG_ROOT / "10-capabilities.json"
DEFAULT_TIMEOUT_SECONDS = 180
_PROFILE_PREFIX = "PRP_EXTERNAL_"
_PROXY_NAMES = frozenset(
    {
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    pytest_args: tuple[str, ...]
    timeout_seconds: int
    supports_case: bool = False


STAGE_REGISTRY = MappingProxyType(
    {
        "preflight": StageSpec(
            name="preflight",
            pytest_args=("--collect-only", "-q", *PREFLIGHT_TARGETS),
            timeout_seconds=DEFAULT_TIMEOUT_SECONDS,
        ),
        "provider": StageSpec(
            name="provider",
            pytest_args=("-m", "live_provider", "-v", *PREFLIGHT_TARGETS),
            timeout_seconds=600,
        ),
        "providers": StageSpec(
            name="providers",
            pytest_args=("-m", "live_provider", "-v", *PREFLIGHT_TARGETS),
            timeout_seconds=600,
        ),
        "integration": StageSpec(
            name="integration",
            pytest_args=("-m", "live_integration", "-v", *PREFLIGHT_TARGETS),
            timeout_seconds=600,
            supports_case=True,
        ),
        "protocols": StageSpec(
            name="protocols",
            pytest_args=("-m", "live_protocols", "-v", *PREFLIGHT_TARGETS),
            timeout_seconds=600,
            supports_case=True,
        ),
        "strategy": StageSpec(
            name="strategy",
            pytest_args=("-m", "live_strategy", "-v", *STRATEGY_TARGETS),
            timeout_seconds=900,
            supports_case=True,
        ),
        "strategies": StageSpec(
            name="strategies",
            pytest_args=("-m", "live_strategy", "-v", *STRATEGY_TARGETS),
            timeout_seconds=900,
            supports_case=True,
        ),
        "agent": StageSpec(
            name="agent",
            pytest_args=("-m", "live_agent", "-v", *AGENT_TARGETS),
            timeout_seconds=900,
            supports_case=True,
        ),
        "regression": StageSpec(
            name="regression",
            pytest_args=("-m", "live_regression", "-v", *PREFLIGHT_TARGETS),
            timeout_seconds=600,
        ),
    }
)
STAGES = STAGE_REGISTRY


@dataclass(frozen=True)
class StageResult:
    stage: str
    status: str
    presence_count: int
    host_aliases: tuple[str, ...]
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    collected_count: int | None = None


def _is_proxy_name(name: str) -> bool:
    return name.lower() in _PROXY_NAMES


def build_child_env(
    credentials: CredentialSet,
    base_env: dict[str, str] | None = None,
    result_file_path: str | None = None,
) -> dict[str, str]:
    """Copy a base environment, then remove ambient test and proxy settings."""

    child_env = dict(os.environ if base_env is None else base_env)
    for name in tuple(child_env):
        if name.startswith(_PROFILE_PREFIX) or _is_proxy_name(name):
            child_env.pop(name, None)

    for alias in credentials.aliases:
        profile_env = credentials.profile_env(alias)
        overlap = set(profile_env).intersection(child_env)
        for name in overlap:
            child_env.pop(name, None)
        child_env.update(profile_env)

    # Enable external tests in child process
    child_env["PRP_EXTERNAL_TESTS"] = "1"
    child_env["PRP_EXTERNAL_PROFILE_ALIASES"] = ",".join(credentials.aliases)

    # Set result file path if provided
    if result_file_path:
        child_env["PRP_LIVE_RESULT_FILE"] = result_file_path

    return child_env


def _redact_output(output: str, credentials: CredentialSet) -> str:
    """Redact values before retaining child output for unit diagnostics."""

    redacted = output
    for alias in credentials.aliases:
        secret_name = f"{_PROFILE_PREFIX}{alias}_API_KEY"
        secret = credentials.profile_env(alias)[secret_name]
        if secret:
            redacted = redacted.replace(secret, "<redacted>")
    return redacted


def _host_aliases() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                urlsplit(contract["base_url"]).hostname
                for contract in PROFILE_CONTRACTS.values()
                if urlsplit(contract["base_url"]).hostname
            }
        )
    )


def run_stage(
    stage: str,
    credentials: CredentialSet,
    *,
    timeout_seconds: int | None = None,
    result_file_path: str | None = None,
    case: str | None = None,
    select: tuple[str, ...] | None = None,
    capability_probe: bool = False,
) -> StageResult:
    """Run exactly one registered pytest child process with structured argv."""

    spec = STAGE_REGISTRY.get(stage)
    if spec is None:
        raise ValueError("unknown stage")

    # Build pytest args with optional case filter
    pytest_args = list(spec.pytest_args)
    if capability_probe:
        pytest_args = ["-m", "live_provider", "-v", *CAPABILITY_TARGETS]
    elif stage == "protocols":
        pytest_args = ["-m", "live_protocols", "-v", *PROTOCOL_TARGETS]
    if case:
        if not spec.supports_case:
            raise ValueError(f"stage '{stage}' does not support --case")
        # Add -k filter for the specific case
        pytest_args.extend(["-k", case])
    if select:
        unknown = sorted(set(select) - set(PROFILE_CONTRACTS))
        if unknown:
            raise ValueError(f"unknown profile alias: {','.join(unknown)}")
        pytest_args.extend(["-k", " or ".join(select)])

    timeout = spec.timeout_seconds if timeout_seconds is None else timeout_seconds
    argv = [sys.executable, "-m", "pytest", *pytest_args]
    child_env = build_child_env(credentials, result_file_path=result_file_path)
    if stage == "preflight":
        child_env["PRP_LIVE_PREFLIGHT"] = "1"
        child_env["PRP_LIVE_EXPECTED_SCENARIOS"] = str(len(PROFILE_CONTRACTS))
    else:
        child_env.pop("PRP_LIVE_PREFLIGHT", None)
        child_env.pop("PRP_LIVE_EXPECTED_SCENARIOS", None)
    if capability_probe:
        child_env["PRP_LIVE_CAPABILITY_PROBE"] = "1"
        child_env["PRP_LIVE_CAPABILITY_FILE"] = str(DEFAULT_CAPABILITY_FILE)
        child_env["PRP_LIVE_CAPABILITY_ALIASES"] = ",".join(select or ())
    else:
        child_env.pop("PRP_LIVE_CAPABILITY_PROBE", None)
        child_env.pop("PRP_LIVE_CAPABILITY_FILE", None)
        child_env.pop("PRP_LIVE_CAPABILITY_ALIASES", None)
    try:
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=child_env,
            shell=False,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return StageResult(
            stage=stage,
            status="TIMEOUT",
            presence_count=len([name for name in child_env if name.startswith(_PROFILE_PREFIX)]),
            host_aliases=_host_aliases(),
            exit_code=124,
            collected_count=None,
        )

    output = completed.stdout or ""
    collected_match = re.search(r"\b(\d+) tests? collected\b", output)
    return StageResult(
        stage=stage,
        status="PASS" if completed.returncode == 0 else "FAIL",
        presence_count=len([name for name in child_env if name.startswith(_PROFILE_PREFIX)]),
        host_aliases=_host_aliases(),
        exit_code=completed.returncode,
        stdout=_redact_output(output, credentials),
        stderr=_redact_output(completed.stderr or "", credentials),
        collected_count=int(collected_match.group(1)) if collected_match else None,
    )


def _summary(result: StageResult) -> str:
    hosts = ",".join(result.host_aliases)
    return (
        f"stage={result.stage} status={result.status} "
        f"presence_count={result.presence_count} host_aliases={hosts} "
        f"exit_code={result.exit_code}"
        + (
            f" collected_count={result.collected_count}"
            if result.collected_count is not None
            else ""
        )
    )


def _prepare_run_artifacts(result_file_path: Path) -> None:
    """Create bounded harness/ledger artifacts without exposing credentials."""
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    if not LOG_ROOT.is_dir():
        raise OSError(f"log directory is not accessible: {LOG_ROOT}")
    LedgerStore(result_file_path).merge([])
    CapabilityStore(DEFAULT_CAPABILITY_FILE).merge(())


def _append_harness_log(message: str) -> None:
    with HARNESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one unattended validation stage")
    parser.add_argument("--credential-file", required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_REGISTRY), required=True)
    parser.add_argument("--case", help="Optional test case filter (pytest -k)")
    parser.add_argument(
        "--select",
        help="Comma-separated active profile aliases to include in the stage",
    )
    parser.add_argument(
        "--capability-probe",
        action="store_true",
        help="Probe structured output and tool calls for prior PASS profiles",
    )
    parser.add_argument("--result-file", help="Path to JSONL result ledger file")
    return parser.parse_args(argv)


def _parse_selection(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    aliases = tuple(part.strip() for part in raw.split(","))
    if not aliases or any(not alias for alias in aliases):
        raise ValueError("--select must contain non-empty profile aliases")
    if len(set(aliases)) != len(aliases):
        raise ValueError("--select must not contain duplicate profile aliases")
    unknown = sorted(set(aliases) - set(PROFILE_CONTRACTS))
    if unknown:
        raise ValueError(f"unknown profile alias: {','.join(unknown)}")
    return aliases


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result_file_path = Path(args.result_file) if args.result_file else STAGE_RESULT_FILES.get(
        args.stage, DEFAULT_RESULT_FILE
    )
    try:
        credentials = load_credentials(args.credential_file)
        selection = _parse_selection(args.select)
        _prepare_run_artifacts(result_file_path)
        if args.capability_probe:
            successful_aliases = tuple(
                entry.alias
                for entry in LedgerStore(result_file_path).read()
                if entry.status == "PASS"
            )
            selection = tuple(dict.fromkeys(successful_aliases))
            if not selection:
                raise ValueError("capability probe requires a prior provider PASS")
        _append_harness_log(
            f"stage_start stage={args.stage} aliases={len(credentials.aliases)} "
            f"credential_source=authorized_markdown"
        )
        result = run_stage(
            args.stage,
            credentials,
            result_file_path=str(result_file_path),
            case=args.case,
            select=selection,
            capability_probe=args.capability_probe,
        )
    except CredentialError as error:
        print(
            f"stage={args.stage} status=CREDENTIAL_ERROR_{error.code} "
            "presence_count=0 exit_code=2"
        )
        return 2
    except (OSError, ValueError) as e:
        print(f"stage={args.stage} status=LAUNCH_ERROR presence_count=0 exit_code=2 error={e}")
        return 2
    summary = _summary(result)
    if args.stage == "preflight" and result.collected_count != len(PROFILE_CONTRACTS):
        summary = (
            f"{summary} expected_scenarios={len(PROFILE_CONTRACTS)} "
            "status=FAIL_COLLECTION_GUARD"
        )
        _append_harness_log(summary)
        print(summary)
        return 3
    _append_harness_log(summary)
    print(summary)
    if result.stdout:
        print("\n=== STDOUT ===")
        print(result.stdout)
    if result.stderr:
        print("\n=== STDERR ===")
        print(result.stderr)
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
