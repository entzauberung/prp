"""Bounded, shell-free workspace search using a controlled ripgrep process."""

import asyncio
from pathlib import Path
from shutil import which
from typing import Annotated, cast

from pydantic import BaseModel, Field, StringConstraints, field_validator

from prp_runtime.domain.enums import ToolEffect
from prp_runtime.domain.models import DomainModel
from prp_runtime.json_support import StrictJsonError, strict_json_loads
from prp_runtime.tools.executor import ExecutionContext
from prp_runtime.tools.models import MAX_TOOL_OUTPUT_BYTES
from prp_runtime.tools.registry import ToolDefinition, ToolHandler
from prp_runtime.workspace.backend import WorkspaceBackend

__all__ = [
    "SearchMatch",
    "SearchRequest",
    "SearchResult",
    "SearchExecutionError",
    "SearchRunner",
    "SearchUnavailableError",
    "build_search_definition",
    "build_rg_argv",
    "parse_rg_json",
    "require_rg",
    "resolve_search_root",
]

_MAX_PATTERN_CHARS = 512
_MAX_RESULTS = 1_000
_MAX_PER_FILE = 100
_MAX_CONTEXT_LINES = 10
_MAX_MATCH_TEXT_CHARS = 8_192
_MAX_SEARCH_OUTPUT_BYTES = 512 * 1024
_SEARCH_TIMEOUT_SECONDS = 15.0
_SAFE_SEARCH_ENV = {"LANG": "C", "LC_ALL": "C", "PATH": ""}

Pattern = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=_MAX_PATTERN_CHARS),
]
RelativePath = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=1024),
]
MatchText = Annotated[str, StringConstraints(max_length=_MAX_MATCH_TEXT_CHARS)]


class SearchUnavailableError(RuntimeError):
    """The configured ripgrep capability is unavailable on this host."""


class SearchExecutionError(RuntimeError):
    """A controlled ripgrep process did not produce a usable result."""


class SearchRequest(DomainModel):
    """Finite, structured search input that cannot become a shell fragment."""

    pattern: Pattern
    root: str = ""
    glob: str = "**/*"
    max_results: int = Field(default=100, gt=0, le=_MAX_RESULTS)
    max_matches_per_file: int = Field(default=20, gt=0, le=_MAX_PER_FILE)
    context_lines: int = Field(default=0, ge=0, le=_MAX_CONTEXT_LINES)

    @field_validator("root")
    @classmethod
    def _root_is_relative(cls, value: str) -> str:
        _validate_relative_path(value, allow_root=True)
        return value

    @field_validator("glob")
    @classmethod
    def _glob_is_relative(cls, value: str) -> str:
        if not value or value.startswith(("/", "\\")) or "\\" in value:
            raise ValueError("glob must be a relative POSIX pattern")
        if any(part == ".." for part in value.split("/")):
            raise ValueError("glob must not contain parent segments")
        return value


class SearchMatch(DomainModel):
    """One bounded match returned by ripgrep JSON parsing."""

    path: RelativePath
    line: int = Field(ge=1)
    column: int = Field(ge=1)
    text: MatchText

    @field_validator("path")
    @classmethod
    def _path_is_relative(cls, value: str) -> str:
        _validate_relative_path(value)
        return value


class SearchResult(DomainModel):
    """Bounded parsed matches and an explicit truncation fact."""

    matches: tuple[SearchMatch, ...] = ()
    truncated: bool = False


class SearchRunner:
    """Run one bounded ripgrep query below a server-owned workspace cwd."""

    def __init__(
        self,
        backend: WorkspaceBackend,
        *,
        workspace_cwd: Path,
        rg_path: Path | None = None,
        timeout_seconds: float = _SEARCH_TIMEOUT_SECONDS,
        max_output_bytes: int = _MAX_SEARCH_OUTPUT_BYTES,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_bytes <= 0 or max_output_bytes > _MAX_SEARCH_OUTPUT_BYTES:
            raise ValueError("max_output_bytes exceeds the search limit")
        try:
            workspace_cwd.lstat()
        except OSError as error:
            raise SearchExecutionError("workspace search root is unavailable") from error
        if workspace_cwd.is_symlink() or not workspace_cwd.is_dir():
            raise SearchExecutionError("workspace search root is not a directory")
        self._backend = backend
        self._workspace_cwd = workspace_cwd
        self._rg_path = rg_path or require_rg()
        self._timeout_seconds = timeout_seconds
        self._max_output_bytes = max_output_bytes

    async def search(self, request: SearchRequest) -> SearchResult:
        """Search an authorized relative root, with bounded output and cancellation."""
        relative_root = resolve_search_root(self._backend, request)
        cwd = self._workspace_cwd if not relative_root else self._workspace_cwd / relative_root
        argv = (str(self._rg_path), *build_rg_argv(request, root=".")[1:])
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=cwd,
                env=_SAFE_SEARCH_ENV,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError as error:
            raise SearchUnavailableError("search backend is unavailable") from error
        try:
            output, output_truncated = await asyncio.wait_for(
                self._collect_output(process), timeout=self._timeout_seconds
            )
        except TimeoutError as error:
            await self._stop_process(process)
            raise SearchExecutionError("search timed out") from error
        except asyncio.CancelledError:
            await self._stop_process(process)
            raise

        if not output_truncated and process.returncode not in (0, 1):
            raise SearchExecutionError("search backend failed")
        parsed = parse_rg_json(output, max_results=request.max_results)
        return SearchResult(
            matches=parsed.matches,
            truncated=output_truncated or parsed.truncated,
        )

    async def _collect_output(self, process: asyncio.subprocess.Process) -> tuple[bytes, bool]:
        if process.stdout is None:
            raise SearchExecutionError("search output pipe is unavailable")
        chunks: list[bytes] = []
        size = 0
        truncated = False
        while True:
            chunk = await process.stdout.read(8_192)
            if not chunk:
                break
            remaining = self._max_output_bytes - size
            if remaining <= 0:
                truncated = True
                await self._stop_process(process)
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                size += remaining
                truncated = True
                await self._stop_process(process)
                break
            chunks.append(chunk)
            size += len(chunk)
        await process.wait()
        output = b"".join(chunks)
        if truncated:
            # A byte ceiling can split an rg JSON record. Keep only complete
            # records so truncation remains a result fact, not a parser error.
            output = output.rsplit(b"\n", 1)[0]
        return output, truncated

    @staticmethod
    async def _stop_process(process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=1.0)
        except TimeoutError:
            process.kill()
            await process.wait()


def resolve_search_root(backend: WorkspaceBackend, request: SearchRequest) -> str:
    """Resolve a request root through the authorized workspace backend."""
    return backend.resolve(request.root)


def build_rg_argv(
    request: SearchRequest,
    *,
    root: str,
    rg_binary: str = "rg",
) -> tuple[str, ...]:
    """Build a fixed ripgrep argv; callers must use it with ``shell=False``."""
    _validate_relative_path(root, allow_root=True)
    if not rg_binary or "/" in rg_binary or "\\" in rg_binary:
        raise ValueError("rg_binary must be a bare executable name")
    return (
        rg_binary,
        "--json",
        "--color",
        "never",
        "--line-number",
        "--column",
        "--max-count",
        str(request.max_matches_per_file),
        "--context",
        str(request.context_lines),
        "--glob",
        request.glob,
        "--",
        request.pattern,
        root or ".",
    )


def parse_rg_json(output: bytes, *, max_results: int) -> SearchResult:
    """Parse finite ripgrep JSON output without preserving raw process output."""
    matches: list[SearchMatch] = []
    for raw_line in output.splitlines():
        try:
            event = strict_json_loads(raw_line.decode("utf-8"))
        except (StrictJsonError, UnicodeDecodeError) as error:
            raise SearchExecutionError("search backend emitted invalid JSON") from error
        if not isinstance(event, dict):
            raise SearchExecutionError("search backend emitted an invalid event")
        if event.get("type") != "match":
            continue
        data = event.get("data")
        if not isinstance(data, dict):
            raise SearchExecutionError("search backend emitted an invalid match")
        path = _rg_text(data.get("path"))
        lines = _rg_text(data.get("lines"))
        line_number = data.get("line_number")
        submatches = data.get("submatches")
        if (
            path is None
            or lines is None
            or isinstance(line_number, bool)
            or not isinstance(line_number, int)
            or not isinstance(submatches, list)
            or not submatches
        ):
            raise SearchExecutionError("search backend emitted an invalid match")
        first = submatches[0]
        if (
            not isinstance(first, dict)
            or isinstance(first.get("start"), bool)
            or not isinstance(first.get("start"), int)
        ):
            raise SearchExecutionError("search backend emitted an invalid match")
        if len(matches) >= max_results:
            return SearchResult(matches=tuple(matches), truncated=True)
        try:
            matches.append(
                SearchMatch(
                    path=path.removeprefix("./"),
                    line=line_number,
                    column=int(first["start"]) + 1,
                    text=lines.rstrip("\r\n"),
                )
            )
        except ValueError as error:
            raise SearchExecutionError("search backend returned an unsafe path") from error
    return SearchResult(
        matches=tuple(sorted(matches, key=lambda item: (item.path, item.line, item.column))),
        truncated=False,
    )


def build_search_definition(runner: SearchRunner) -> ToolDefinition:
    """Build the registered read-only search tool around one controlled runner."""

    async def handler(context: BaseModel) -> dict[str, object]:
        if not isinstance(context, ExecutionContext):
            raise TypeError("search_text requires an execution context")
        if not isinstance(context.arguments, SearchRequest):
            raise TypeError("search_text received an invalid argument model")
        return (await runner.search(context.arguments)).model_dump(mode="json")

    return ToolDefinition(
        name="search_text",
        description="Search authorized workspace text with bounded results.",
        effect=ToolEffect.READ,
        argument_model=SearchRequest,
        handler=cast(ToolHandler, handler),
        max_output_bytes=MAX_TOOL_OUTPUT_BYTES,
    )


def require_rg(rg_binary: str = "rg") -> Path:
    """Return the available ripgrep executable or raise a stable error."""
    resolved = which(rg_binary)
    if resolved is None:
        raise SearchUnavailableError("search backend is unavailable")
    return Path(resolved)


def _validate_relative_path(value: str, *, allow_root: bool = False) -> None:
    if not isinstance(value, str):
        raise ValueError("path must be a relative POSIX string")
    if allow_root and value in {"", "."}:
        return
    if not value or value.startswith(("/", "\\")) or "\\" in value:
        raise ValueError("path must be a relative POSIX string")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("path must be a relative POSIX string")


def _rg_text(value: object) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    return text if isinstance(text, str) else None
