"""Targeted tests for the shell-free ripgrep search contract."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from prp_runtime.tools.search import (
    SearchExecutionError,
    SearchMatch,
    SearchRequest,
    SearchRunner,
    SearchUnavailableError,
    build_rg_argv,
    parse_rg_json,
    require_rg,
)
from prp_runtime.workspace import WorkspaceBackend, WorkspaceBackendError


def test_rg_argv_keeps_untrusted_pattern_and_glob_as_distinct_values() -> None:
    request = SearchRequest(
        pattern="$(touch should-not-run); --glob '*.secret'",
        root="src",
        glob="**/*.py",
        max_matches_per_file=7,
        context_lines=2,
    )
    argv = build_rg_argv(request, root="src")

    assert argv[:2] == ("rg", "--json")
    assert argv[argv.index("--") + 1] == request.pattern
    assert argv[-1] == "src"
    assert "shell" not in argv
    assert "--glob" in argv
    assert argv[argv.index("--glob") + 1] == "**/*.py"


@pytest.mark.parametrize("root", ["/tmp", "../outside", "src/../outside", "src\\main"])
def test_search_root_and_match_paths_reject_escape_syntax(root: str) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(pattern="needle", root=root)
    with pytest.raises(ValidationError):
        SearchMatch(path=root, line=1, column=1, text="needle")


@pytest.mark.parametrize(
    "values",
    [
        {"pattern": "x", "max_results": 0},
        {"pattern": "x", "max_matches_per_file": 101},
        {"pattern": "x", "context_lines": 11},
        {"pattern": "x", "glob": "../*"},
    ],
)
def test_search_limits_and_glob_are_bounded(values: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        SearchRequest(**values)


def test_rg_binary_must_be_bare_and_unavailable_is_stable(monkeypatch: pytest.MonkeyPatch) -> None:
    request = SearchRequest(pattern="needle")
    with pytest.raises(ValueError, match="bare executable"):
        build_rg_argv(request, root="", rg_binary="/usr/bin/rg")

    monkeypatch.setattr("prp_runtime.tools.search.which", lambda _: None)
    with pytest.raises(SearchUnavailableError, match="search backend is unavailable"):
        require_rg()


def test_parser_sorts_relative_matches_and_marks_count_truncation() -> None:
    def match(path: str, line: int, column: int, text: str) -> bytes:
        return json.dumps(
            {
                "type": "match",
                "data": {
                    "path": {"text": path},
                    "lines": {"text": text + "\n"},
                    "line_number": line,
                    "submatches": [{"start": column - 1}],
                },
            }
        ).encode()

    result = parse_rg_json(
        b"\n".join((match("b.py", 2, 3, "second"), match("a.py", 1, 1, "first"))),
        max_results=1,
    )
    assert [item.path for item in result.matches] == ["b.py"]
    assert result.truncated is True


@pytest.mark.parametrize(
    "output",
    [b"[]", b'{"type":"match","data":{"line_number":true}}'],
)
def test_parser_rejects_non_object_or_boolean_match_facts(output: bytes) -> None:
    with pytest.raises(SearchExecutionError):
        parse_rg_json(output, max_results=10)


@pytest.mark.asyncio
async def test_runner_uses_fixed_cwd_env_and_stops_on_cancellation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "src").mkdir()
    started = asyncio.Event()
    subprocess_started = asyncio.Event()

    class Process:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.returncode: int | None = None
            self.terminated = False

        async def wait(self) -> int:
            await started.wait()
            self.returncode = -15
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            started.set()

        def kill(self) -> None:
            self.terminated = True
            started.set()

    process = Process()
    captured: dict[str, object] = {}

    async def fake_exec(*argv: str, **kwargs: object) -> Process:
        captured["argv"] = argv
        captured["cwd"] = kwargs["cwd"]
        captured["env"] = kwargs["env"]
        subprocess_started.set()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    with WorkspaceBackend(tmp_path) as backend:
        runner = SearchRunner(backend, workspace_cwd=tmp_path, rg_path=Path("/srv/rg"))
        task = asyncio.create_task(runner.search(SearchRequest(pattern="needle", root="src")))
        await subprocess_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert process.terminated is True
    assert captured["cwd"] == tmp_path / "src"
    assert captured["env"] == {"LANG": "C", "LC_ALL": "C", "PATH": ""}
    assert "--" in captured["argv"]


@pytest.mark.asyncio
async def test_runner_respects_glob_empty_binary_and_relative_roots(tmp_path: Path) -> None:
    source = tmp_path / "src"
    source.mkdir()
    (source / "match.py").write_text("needle\n", encoding="utf-8")
    (source / "ignored.txt").write_text("needle\n", encoding="utf-8")
    (source / "binary.bin").write_bytes(b"\x00needle\n")

    with WorkspaceBackend(tmp_path) as backend:
        runner = SearchRunner(backend, workspace_cwd=tmp_path, rg_path=require_rg())
        python = await runner.search(
            SearchRequest(pattern="needle", root="src", glob="*.py")
        )
        empty = await runner.search(SearchRequest(pattern="absent", root="src"))
        binary = await runner.search(SearchRequest(pattern="needle", root="src", glob="*.bin"))

    assert [match.path for match in python.matches] == ["match.py"]
    assert empty.matches == ()
    assert empty.truncated is False
    assert binary.matches == ()
    assert binary.truncated is False


@pytest.mark.asyncio
async def test_runner_truncates_output_without_parsing_partial_json(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("needle\n" * 1_000, encoding="utf-8")

    with WorkspaceBackend(tmp_path) as backend:
        runner = SearchRunner(
            backend,
            workspace_cwd=tmp_path,
            rg_path=require_rg(),
            max_output_bytes=512,
        )
        result = await runner.search(SearchRequest(pattern="needle", max_results=1_000))

    assert result.matches
    assert result.truncated is True


@pytest.mark.asyncio
async def test_runner_terminates_on_timeout_and_rejects_escaped_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    release = asyncio.Event()

    class Process:
        def __init__(self) -> None:
            self.stdout = asyncio.StreamReader()
            self.returncode: int | None = None
            self.terminated = False

        async def wait(self) -> int:
            await release.wait()
            self.returncode = -15
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            release.set()

        def kill(self) -> None:
            self.terminated = True
            release.set()

    process = Process()

    async def fake_exec(*argv: str, **kwargs: object) -> Process:
        del argv, kwargs
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "escape").symlink_to(outside, target_is_directory=True)

    with WorkspaceBackend(tmp_path) as backend:
        runner = SearchRunner(
            backend,
            workspace_cwd=tmp_path,
            rg_path=Path("/srv/rg"),
            timeout_seconds=0.01,
        )
        with pytest.raises(SearchExecutionError, match="timed out"):
            await runner.search(SearchRequest(pattern="needle"))
        with pytest.raises(WorkspaceBackendError, match="symlink"):
            await runner.search(SearchRequest(pattern="needle", root="escape"))

    assert process.terminated is True
