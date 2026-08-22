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
    from .capability_ledger import CapabilityStore
    from .credential_loader import (
        PROFILE_CONTRACTS,
        TERRA_ALIAS,
        CredentialError,
        CredentialSet,
        OptionalProfileResolution,
        load_credentials_from_env,
        resolve_terra_profile,
    )
    from .result_ledger import LedgerStore
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from capability_ledger import CapabilityStore  # type: ignore[no-redef]
    from credential_loader import (  # type: ignore[no-redef]
        PROFILE_CONTRACTS,
        TERRA_ALIAS,
        CredentialError,
        CredentialSet,
        OptionalProfileResolution,
        load_credentials_from_env,
        resolve_terra_profile,
    )
    from result_ledger import LedgerStore  # type: ignore[no-redef]


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
LOG_ROOT = Path(os.environ.get("PRP_EXTERNAL_RESULT_DIR", "external_tests/.results"))
HARNESS_LOG = LOG_ROOT / "10-repair-gate.log"
DEFAULT_RESULT_FILE = LOG_ROOT / "20-chat.jsonl"
INTERFACE_RESULT_FILES = MappingProxyType(
    {
        "OPENAI_CHAT": LOG_ROOT / "20-chat.jsonl",
        "OPENAI_RESPONSES": LOG_ROOT / "30-responses.jsonl",
        "ANTHROPIC_MESSAGES": LOG_ROOT / "40-anthropic.jsonl",
    }
)
PROTOCOL_CASE_INTERFACES = MappingProxyType(
    {
        "chat": "OPENAI_CHAT",
        "responses": "OPENAI_RESPONSES",
        "anthropic": "ANTHROPIC_MESSAGES",
        "messages": "ANTHROPIC_MESSAGES",
    }
)


def _interface_arg(value: str) -> str:
    """Normalize documented lower-case CLI names to internal protocol names."""

    normalized = value.strip().upper()
    if normalized not in INTERFACE_CANDIDATES:
        raise argparse.ArgumentTypeError(f"unknown interface: {value}")
    return normalized


INTERFACE_CANDIDATES = MappingProxyType(
    {
        "OPENAI_CHAT": ("DEEPSEEK_FLASH_CHAT",),
        "OPENAI_RESPONSES": (
            "DEEPSEEK_FLASH_RESPONSES",
            "LUNA_GPT_56",
            TERRA_ALIAS,
        ),
        "ANTHROPIC_MESSAGES": (
            "DEEPSEEK_FLASH_ANTHROPIC",
            "CLAUDE_SONNET_5",
        ),
    }
)
STAGE_RESULT_FILES = MappingProxyType(
    {
        "protocols": LOG_ROOT / "30-responses.jsonl",
        "strategy": LOG_ROOT / "50-reasoning.jsonl",
        "strategies": LOG_ROOT / "50-reasoning.jsonl",
        "agent": LOG_ROOT / "60-agent.jsonl",
        "resilience": LOG_ROOT / "70-resilience.log",
        "regression": LOG_ROOT / "80-regression.log",
    }
)
DEFAULT_CAPABILITY_FILE = Path(
    os.environ.get(
        "PRP_LIVE_CAPABILITY_FILE",
        str(LOG_ROOT / "provider-capabilities.json"),
    )
)
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
            supports_case=True,
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
    interface: str | None = None
    candidate_aliases: tuple[str, ...] = ()
    stdout: str = ""
    stderr: str = ""
    collected_count: int | None = None


def _is_proxy_name(name: str) -> bool:
    return name.lower() in _PROXY_NAMES


def build_child_env(
    credentials: CredentialSet,
    base_env: dict[str, str] | None = None,
    result_file_path: str | None = None,
    selected_aliases: tuple[str, ...] | None = None,
    terra_resolution: OptionalProfileResolution | None = None,
) -> dict[str, str]:
    """Copy a base environment, then remove ambient test and proxy settings."""

    source_env = dict(os.environ if base_env is None else base_env)
    child_env = dict(source_env)
    for name in tuple(child_env):
        if name.startswith(_PROFILE_PREFIX) or _is_proxy_name(name):
            child_env.pop(name, None)

    aliases = credentials.aliases if selected_aliases is None else selected_aliases
    active_aliases = set(credentials.aliases)
    unknown = set(aliases) - active_aliases - {TERRA_ALIAS}
    if unknown:
        raise ValueError(f"unknown profile alias: {','.join(sorted(unknown))}")
    if TERRA_ALIAS in aliases:
        if terra_resolution is None:
            terra_resolution = resolve_terra_profile(
                source_env,
                allowed_hosts=tuple(_host_aliases()),
            )
        if terra_resolution is None or terra_resolution.status != "READY":
            raise ValueError("TERRA_GPT is not configured for child execution")
        if terra_resolution.api_key is None:
            raise ValueError("TERRA_GPT has no credential mapping")
    for alias in aliases:
        if alias == TERRA_ALIAS:
            profile_env = {
                "PRP_EXTERNAL_TERRA_GPT_API_KEY": terra_resolution.api_key,
                "PRP_EXTERNAL_TERRA_GPT_MODEL": terra_resolution.model,
                "PRP_EXTERNAL_TERRA_GPT_BASE_URL": terra_resolution.base_url,
                "PRP_EXTERNAL_TERRA_GPT_ALLOWED_HOST": terra_resolution.allowed_host,
            }
        else:
            profile_env = credentials.profile_env(alias)
        overlap = set(profile_env).intersection(child_env)
        for name in overlap:
            child_env.pop(name, None)
        child_env.update(profile_env)

    # Enable external tests in child process
    child_env["PRP_EXTERNAL_TESTS"] = "1"
    child_env["PRP_EXTERNAL_PROFILE_ALIASES"] = ",".join(aliases)

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


def _select_interface_candidates(
    interface: str,
    credentials: CredentialSet,
    requested: tuple[str, ...] | None,
) -> tuple[str, ...]:
    """Return active candidates in matrix order for one protocol interface."""

    candidates = INTERFACE_CANDIDATES.get(interface)
    if candidates is None:
        raise ValueError(f"unknown interface: {interface}")
    active = set(credentials.aliases)
    if requested is None:
        selected = tuple(alias for alias in candidates if alias in active)
    else:
        unknown = sorted(set(requested) - set(candidates))
        if unknown:
            raise ValueError(
                f"profile alias is not a candidate for {interface}: {','.join(unknown)}"
            )
        selected = tuple(alias for alias in candidates if alias in requested)
    if not selected:
        raise ValueError(f"no configured candidate for interface: {interface}")
    return selected


def _first_actual_pass_candidate(
    interface: str,
    candidates: tuple[str, ...],
    entries: tuple[object, ...],
) -> str:
    """Select the first interface candidate with actual outbound PASS evidence."""

    passed_aliases = {
        entry.alias
        for entry in entries
        if getattr(entry, "protocol", None) == interface
        and getattr(entry, "status", None) == "PASS"
        and getattr(entry, "actual_or_simulated", None) == "ACTUAL"
    }
    for alias in candidates:
        if alias in passed_aliases:
            return alias
    raise ValueError(f"protocol ingress requires a prior actual PASS for {interface}")


def _capability_probe_candidates(
    interface: str,
    candidates: tuple[str, ...],
    result_file_path: Path,
) -> tuple[str, ...]:
    """Keep candidates backed by actual provider or capability PASS evidence."""

    provider_passes = {
        entry.alias
        for entry in LedgerStore(result_file_path).read()
        if entry.protocol == interface
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }
    capability_passes = {
        entry.alias
        for entry in CapabilityStore(DEFAULT_CAPABILITY_FILE).read()
        if entry.protocol == interface
        and entry.status == "PASS"
        and entry.actual_or_simulated == "ACTUAL"
    }
    eligible = provider_passes | capability_passes
    return tuple(alias for alias in candidates if alias in eligible)


def run_stage(
    stage: str,
    credentials: CredentialSet,
    *,
    timeout_seconds: int | None = None,
    result_file_path: str | None = None,
    case: str | None = None,
    select: tuple[str, ...] | None = None,
    interface: str | None = None,
    capability_probe: bool = False,
    fallback_from: str | None = None,
    fallback_failure: str | None = None,
) -> StageResult:
    """Run exactly one registered pytest child process with structured argv."""

    spec = STAGE_REGISTRY.get(stage)
    if spec is None:
        raise ValueError("unknown stage")

    if interface is not None:
        selected_aliases = _select_interface_candidates(interface, credentials, select)
    else:
        selected_aliases = credentials.aliases if select is None else select
    if TERRA_ALIAS in selected_aliases:
        terra_resolution = resolve_terra_profile(
            os.environ,
            allowed_hosts=tuple(_host_aliases()),
            fallback_from=fallback_from,
            failure_classification=fallback_failure,
        )
        if terra_resolution.status != "READY":
            return StageResult(
                stage=stage,
                status=terra_resolution.status,
                presence_count=0,
                host_aliases=_host_aliases(),
                exit_code=0,
                interface=interface,
                candidate_aliases=selected_aliases,
            )
    else:
        terra_resolution = None

    # Build pytest args with optional case filter
    pytest_args = list(spec.pytest_args)
    if capability_probe:
        pytest_args = ["-m", "live_provider", "-v", *CAPABILITY_TARGETS]
    elif stage == "protocols":
        pytest_args = ["-m", "live_protocols", "-v", *PROTOCOL_TARGETS]
    if case == "intermediary-fallback" and stage == "providers":
        # The fallback campaign is the intermediary provider matrix. Keep the
        # case name as a runner contract while targeting its actual test module.
        pytest_args = [
            "-m",
            "live_provider",
            "-v",
            "external_tests/test_live_luna_claude.py",
        ]
    elif case == "engineering" and stage == "agent":
        # Engineering is the bounded local fixture matrix in the Agent module;
        # live-gated cases must remain excluded from this offline continuation.
        pytest_args = [
            "-v",
            "external_tests/test_live_agent.py",
            "-k",
            "local_agent",
        ]
    elif case:
        if not spec.supports_case:
            raise ValueError(f"stage '{stage}' does not support --case")
        if stage == "protocols" and case == "anthropic":
            case = "messages"
        # Add -k filter for the specific case
        pytest_args.extend(["-k", case])
    if select:
        unknown = sorted(set(select) - set(PROFILE_CONTRACTS) - {TERRA_ALIAS})
        if unknown:
            raise ValueError(f"unknown profile alias: {','.join(unknown)}")
        pytest_args.extend(["-k", " or ".join(select)])

    timeout = spec.timeout_seconds if timeout_seconds is None else timeout_seconds
    argv = [sys.executable, "-m", "pytest", *pytest_args]
    child_env = build_child_env(
        credentials,
        result_file_path=result_file_path,
        selected_aliases=selected_aliases,
        terra_resolution=terra_resolution,
    )
    if interface is not None:
        child_env["PRP_EXTERNAL_INTERFACE"] = interface
        child_env["PRP_EXTERNAL_CANDIDATE_ORDER"] = ",".join(selected_aliases)
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
            interface=interface,
            candidate_aliases=selected_aliases,
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
        interface=interface,
        candidate_aliases=selected_aliases,
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
        + (f" interface={result.interface}" if result.interface else "")
        + (
            f" candidate_order={','.join(result.candidate_aliases)}"
            if result.candidate_aliases
            else ""
        )
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
    parser.add_argument("--stage", choices=tuple(STAGE_REGISTRY), required=True)
    parser.add_argument("--case", help="Optional test case filter (pytest -k)")
    parser.add_argument(
        "--select",
        help="Comma-separated active profile aliases to include in the stage",
    )
    parser.add_argument(
        "--interface",
        type=_interface_arg,
        choices=tuple(INTERFACE_CANDIDATES),
        help="Select candidates for one protocol interface in matrix order",
    )
    parser.add_argument(
        "--capability-probe",
        action="store_true",
        help="Probe structured output and tool calls for prior PASS profiles",
    )
    parser.add_argument("--result-file", help="Path to JSONL result ledger file")
    parser.add_argument(
        "--fallback-from",
        choices=("LUNA_GPT_56",),
        help="Select Terra only as a fallback after the named profile fails",
    )
    parser.add_argument(
        "--fallback-failure",
        help="Classified failure for --fallback-from; only retryable values permit Terra",
    )
    return parser.parse_args(argv)


def _parse_selection(raw: str | None) -> tuple[str, ...] | None:
    if raw is None:
        return None
    aliases = tuple(part.strip() for part in raw.split(","))
    if not aliases or any(not alias for alias in aliases):
        raise ValueError("--select must contain non-empty profile aliases")
    if len(set(aliases)) != len(aliases):
        raise ValueError("--select must not contain duplicate profile aliases")
    unknown = sorted(set(aliases) - set(PROFILE_CONTRACTS) - {TERRA_ALIAS})
    if unknown:
        raise ValueError(f"unknown profile alias: {','.join(unknown)}")
    return aliases


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.result_file:
        result_file_path = Path(args.result_file)
    elif args.interface:
        result_file_path = INTERFACE_RESULT_FILES[args.interface]
    elif args.stage == "protocols" and args.case in PROTOCOL_CASE_INTERFACES:
        result_file_path = INTERFACE_RESULT_FILES[PROTOCOL_CASE_INTERFACES[args.case]]
    else:
        result_file_path = STAGE_RESULT_FILES.get(args.stage, DEFAULT_RESULT_FILE)
    try:
        credentials = load_credentials_from_env()
        selection = _parse_selection(args.select)
        if args.interface is not None and args.fallback_from is None:
            selection = _select_interface_candidates(args.interface, credentials, selection)
        _prepare_run_artifacts(result_file_path)
        if args.stage == "protocols" and args.interface is not None:
            if selection is None:
                raise ValueError("protocol ingress requires interface candidates")
            selection = (
                _first_actual_pass_candidate(
                    args.interface,
                    selection,
                    LedgerStore(result_file_path).read(),
                ),
            )
        if args.capability_probe:
            selection = _capability_probe_candidates(
                args.interface or "",
                selection or (),
                result_file_path,
            )
            if not selection:
                raise ValueError("capability probe requires a prior provider PASS")
        if args.fallback_from is not None:
            if args.interface not in (None, "OPENAI_RESPONSES"):
                raise ValueError("Terra fallback requires the OPENAI_RESPONSES interface")
            if selection is not None and selection != (TERRA_ALIAS,):
                raise ValueError("--fallback-from requires --select TERRA_GPT or no --select")
            selection = (TERRA_ALIAS,)
        _append_harness_log(
            f"stage_start stage={args.stage} aliases={len(credentials.aliases)} "
            "credential_source=environment"
        )
        result = run_stage(
            args.stage,
            credentials,
            result_file_path=str(result_file_path),
            case=args.case,
            select=selection,
            interface=args.interface,
            capability_probe=args.capability_probe,
            fallback_from=args.fallback_from,
            fallback_failure=args.fallback_failure,
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
