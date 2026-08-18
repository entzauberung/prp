"""CLI tests with a fake Bridge; no real network or token output."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from prp_runtime.client import cli
from prp_runtime.client.bridge import BridgeToolLoopPhase


class FakeBridge:
    instances: list[FakeBridge] = []

    def __init__(self, base_url: str, token: str, **kwargs: object) -> None:
        self.base_url = base_url
        self.token = token
        self.kwargs = kwargs
        self.state = SimpleNamespace(session_id=None, run_id=None)
        self.sessions: list[dict[str, object]] = []
        self.loop_calls: list[object] = []
        FakeBridge.instances.append(self)

    async def __aenter__(self) -> FakeBridge:
        return self

    async def __aexit__(self, *args: object) -> None:
        return None

    async def create_session(self, workspace_id: str, **kwargs: object) -> dict[str, object]:
        self.state.session_id = "sess-1"
        self.sessions.append({"workspace_id": workspace_id, **kwargs})
        return {"session_id": self.state.session_id}

    async def create_run(self, session_id: str | None, payload: object) -> dict[str, object]:
        self.state.run_id = "run-1"
        return {"run_id": self.state.run_id, "payload": payload}

    async def iter_events(self, **kwargs: object):
        yield {"sequence": 1, "event_type": "RUN_SUCCEEDED"}

    async def get_run(self) -> dict[str, object]:
        return {"run_id": "run-1", "status": "SUCCEEDED"}

    async def run_tool_loop(self, executor: object, **kwargs: object) -> object:
        self.loop_calls.append((executor, kwargs))
        return SimpleNamespace(phase=BridgeToolLoopPhase.TERMINAL, pending_call_ids=())

    async def _request(self, *args: object, **kwargs: object) -> dict[str, object]:
        return {"outcome": kwargs["body"]["outcome"]}  # type: ignore[index]


@pytest.fixture(autouse=True)
def reset_fake_bridge(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeBridge.instances.clear()
    monkeypatch.setattr(cli, "Bridge", FakeBridge)


def test_connect_reads_env_token_and_keeps_root_local(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    state_path = tmp_path / "state.json"
    root = tmp_path / "workspace"

    assert (
        cli.main(
            [
                "--base-url",
                "https://bridge.test",
                "--state",
                str(state_path),
                "--workspace-root",
                str(root),
                "connect",
                "workspace-1",
            ]
        )
        == 0
    )
    bridge = FakeBridge.instances[0]
    assert bridge.token == "do-not-print"
    assert bridge.sessions[0]["workspace_id"] == "workspace-1"
    assert str(root) not in capsys.readouterr().out
    assert "do-not-print" not in capsys.readouterr().out


def test_host_yolo_requires_interactive_confirmation(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: False)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: False)

    result = cli.main(
        ["connect", "workspace-1", "--agent-mode", "YOLO", "--isolation-mode", "HOST"]
    )

    assert result == 1
    assert "interactive confirmation" in capsys.readouterr().err
    assert not FakeBridge.instances


def test_host_yolo_confirmation_is_forwarded_as_explicit_user_intent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = SimpleNamespace(agent_mode="YOLO", isolation_mode="HOST")
    monkeypatch.setattr(cli.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(cli.sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt: "HOST YOLO")

    assert cli._confirm_host_yolo(args) is True
    assert cli._agent_options(args, user_explicit=True)["user_explicit"] is True


def test_help_explains_path_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "not an operating-system sandbox" in capsys.readouterr().out


def test_watch_runs_bounded_loop_and_prints_terminal_run_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")
    root = tmp_path / "workspace"
    root.mkdir()

    assert (
        cli.main(
            [
                "--workspace-root",
                str(root),
                "watch",
                "--max-calls",
                "2",
                "--max-events",
                "4",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert output.count('"status": "SUCCEEDED"') == 1
    assert str(root) not in output
    assert "do-not-print" not in output
    assert len(FakeBridge.instances[0].loop_calls) == 1


def test_watch_without_root_fails_before_local_execution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("PRP_TOKEN", "do-not-print")

    assert cli.main(["watch"]) == 1

    assert "workspace-root" in capsys.readouterr().err
