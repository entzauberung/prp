"""Python AST symbol facts without importing or executing source code."""

from __future__ import annotations

import ast
import copy
import hashlib
from enum import StrEnum, unique
from typing import Self

from collections.abc import Mapping

from pydantic import Field, model_validator

from prp_runtime.domain.models import DomainModel

__all__ = [
    "BoundSyntaxReport",
    "MAX_SOURCE_BYTES",
    "NodeSpan",
    "SymbolChange",
    "SymbolChangeAction",
    "SymbolFact",
    "SymbolKind",
    "SyntaxAnalyzer",
    "SyntaxReport",
    "analyze_bounded_observation",
    "analyze_python",
    "redact_local_paths",
    "source_pair_from_observation",
]

MAX_SOURCE_BYTES = 4 * 1024 * 1024


@unique
class SymbolKind(StrEnum):
    """AST declarations that are stable enough for conflict evidence."""

    CLASS = "CLASS"
    FUNCTION = "FUNCTION"
    ASYNC_FUNCTION = "ASYNC_FUNCTION"
    IMPORT = "IMPORT"
    ASSIGNMENT = "ASSIGNMENT"


@unique
class SymbolChangeAction(StrEnum):
    """How a symbol differs between the old and new source."""

    ADD = "ADD"
    MODIFY = "MODIFY"
    DELETE = "DELETE"


class NodeSpan(DomainModel):
    """One source location copied from AST node coordinates."""

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_column: int = Field(ge=0)
    end_column: int = Field(ge=0)

    @model_validator(mode="after")
    def _span_is_forward(self) -> Self:
        if (self.end_line, self.end_column) < (self.start_line, self.start_column):
            raise ValueError("AST node span must be forward")
        return self


class SymbolFact(DomainModel):
    """One named AST symbol and its bounded structural fingerprint."""

    key: str = Field(min_length=1, max_length=512)
    kind: SymbolKind
    name: str = Field(min_length=1, max_length=256)
    qualified_name: str = Field(min_length=1, max_length=512)
    span: NodeSpan
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class SymbolChange(DomainModel):
    """One stable symbol addition, modification, or deletion."""

    key: str = Field(min_length=1, max_length=512)
    action: SymbolChangeAction
    before: SymbolFact | None = None
    after: SymbolFact | None = None

    @model_validator(mode="after")
    def _change_is_complete(self) -> Self:
        if self.action is SymbolChangeAction.ADD and (
            self.before is not None or self.after is None
        ):
            raise ValueError("added symbol must only have an after fact")
        if self.action is SymbolChangeAction.MODIFY and (
            self.before is None or self.after is None
        ):
            raise ValueError("modified symbol must have before and after facts")
        if self.action is SymbolChangeAction.DELETE and (
            self.before is None or self.after is not None
        ):
            raise ValueError("deleted symbol must only have a before fact")
        return self


class SyntaxReport(DomainModel):
    """Auditable AST result for one old/new Python source pair."""

    language: str = "python"
    parse_ok: bool
    unknown: bool
    symbols: tuple[SymbolFact, ...] = ()
    changes: tuple[SymbolChange, ...] = ()
    before_parse_error: str | None = None
    after_parse_error: str | None = None

    @property
    def parse_error(self) -> str | None:
        """Return the first parse error for callers needing one reason."""
        return self.before_parse_error or self.after_parse_error

    @property
    def parse_errors(self) -> tuple[str, ...]:
        """Return parse errors in deterministic old/new order."""
        return tuple(
            error
            for error in (self.before_parse_error, self.after_parse_error)
            if error is not None
        )


class _SymbolCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.scope: tuple[str, ...] = ()
        self.symbols: list[SymbolFact] = []

    def _qualified_name(self, name: str) -> str:
        return ".".join((*self.scope, name))

    def _add(self, node: ast.AST, kind: SymbolKind, name: str, payload: ast.AST) -> None:
        qualified_name = self._qualified_name(name)
        key = f"{kind.value.lower()}:{qualified_name}"
        self.symbols.append(
            SymbolFact(
                key=key,
                kind=kind,
                name=name,
                qualified_name=qualified_name,
                span=_span(node),
                fingerprint=_fingerprint(payload),
            )
        )

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add(node, SymbolKind.CLASS, node.name, node)
        self.scope = (*self.scope, node.name)
        self.generic_visit(node)
        self.scope = self.scope[:-1]

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add(node, SymbolKind.FUNCTION, node.name, node)
        self.scope = (*self.scope, node.name)
        self.generic_visit(node)
        self.scope = self.scope[:-1]

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add(node, SymbolKind.ASYNC_FUNCTION, node.name, node)
        self.scope = (*self.scope, node.name)
        self.generic_visit(node)
        self.scope = self.scope[:-1]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            name = alias.asname or alias.name.split(".", maxsplit=1)[0]
            self._add(node, SymbolKind.IMPORT, name, alias)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            name = alias.asname or alias.name
            self._add(node, SymbolKind.IMPORT, name, alias)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            for name in _target_names(target):
                self._add(node, SymbolKind.ASSIGNMENT, name, node)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        for name in _target_names(node.target):
            self._add(node, SymbolKind.ASSIGNMENT, name, node)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        for name in _target_names(node.target):
            self._add(node, SymbolKind.ASSIGNMENT, name, node)
        self.generic_visit(node)


class _DynamicSyntaxDetector(ast.NodeVisitor):
    """Mark syntax whose runtime symbol effects cannot be proven statically."""

    _DYNAMIC_CALLS = frozenset(
        {
            "__import__",
            "compile",
            "eval",
            "exec",
            "globals",
            "locals",
            "setattr",
            "vars",
        }
    )
    _DYNAMIC_ATTRIBUTES = frozenset({"exec_module", "import_module"})

    def __init__(self) -> None:
        self.unknown = False

    def _mark_unsupported_binding(self, node: ast.AST) -> None:
        del node
        self.unknown = True

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in self._DYNAMIC_CALLS:
            self.unknown = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr in self._DYNAMIC_ATTRIBUTES:
            self.unknown = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(alias.name == "*" for alias in node.names):
            self.unknown = True
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        self.unknown = True

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        self.unknown = True

    def visit_For(self, node: ast.For) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name is not None:
            self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_Match(self, node: ast.Match) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        self._mark_unsupported_binding(node)
        self.generic_visit(node)


def _span(node: ast.AST) -> NodeSpan:
    start_line = getattr(node, "lineno", 1)
    end_line = getattr(node, "end_lineno", start_line)
    start_column = getattr(node, "col_offset", 0)
    end_column = getattr(node, "end_col_offset", start_column)
    return NodeSpan(
        start_line=start_line,
        end_line=end_line,
        start_column=start_column,
        end_column=end_column,
    )


def _fingerprint(node: ast.AST) -> str:
    payload = copy.deepcopy(node)
    if isinstance(payload, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
        stripper = _NestedScopeStripper()
        payload.body = [
            statement
            for statement in (stripper.visit(statement) for statement in payload.body)
            if statement is not None
        ]
    dumped = ast.dump(payload, annotate_fields=True, include_attributes=False)
    return hashlib.sha256(dumped.encode("utf-8")).hexdigest()


class _NestedScopeStripper(ast.NodeTransformer):
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        return None

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return None

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        return None


def _target_names(node: ast.AST) -> tuple[str, ...]:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, (ast.Tuple, ast.List)):
        names: list[str] = []
        for element in node.elts:
            names.extend(_target_names(element))
        return tuple(names)
    return ()


def _collect(source: str) -> tuple[tuple[SymbolFact, ...], str | None, bool]:
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        return (), "source exceeds the syntax analysis byte limit", True
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as error:
        return (), f"SyntaxError: {error.msg} (line {error.lineno})", True
    collector = _SymbolCollector()
    detector = _DynamicSyntaxDetector()
    try:
        collector.visit(tree)
        detector.visit(tree)
        symbols = tuple(sorted(collector.symbols, key=lambda symbol: symbol.key))
    except (TypeError, ValueError):
        return (), "syntax facts exceeded their bounded representation", True
    return symbols, None, detector.unknown


def _diff(
    before: tuple[SymbolFact, ...], after: tuple[SymbolFact, ...]
) -> tuple[SymbolChange, ...]:
    before_by_key = {symbol.key: symbol for symbol in before}
    after_by_key = {symbol.key: symbol for symbol in after}
    changes: list[SymbolChange] = []
    for key in sorted(set(before_by_key) | set(after_by_key)):
        old = before_by_key.get(key)
        new = after_by_key.get(key)
        if old is None:
            changes.append(SymbolChange(key=key, action=SymbolChangeAction.ADD, after=new))
        elif new is None:
            changes.append(SymbolChange(key=key, action=SymbolChangeAction.DELETE, before=old))
        elif old.fingerprint != new.fingerprint:
            changes.append(
                SymbolChange(
                    key=key,
                    action=SymbolChangeAction.MODIFY,
                    before=old,
                    after=new,
                )
            )
    return tuple(changes)


def analyze_python(before_source: str, after_source: str) -> SyntaxReport:
    """Parse old/new Python source and return structural symbol changes."""
    before, before_error, before_unknown = _collect(before_source)
    after, after_error, after_unknown = _collect(after_source)
    if before_error is not None or after_error is not None:
        return SyntaxReport(
            parse_ok=False,
            unknown=True,
            before_parse_error=before_error,
            after_parse_error=after_error,
        )
    return SyntaxReport(
        parse_ok=True,
        unknown=before_unknown or after_unknown,
        symbols=after,
        changes=_diff(before, after),
    )


class SyntaxAnalyzer:
    """Small stateless facade for callers that prefer an analyzer object."""

    def analyze(self, before_source: str, after_source: str) -> SyntaxReport:
        return analyze_python(before_source, after_source)


analyze = analyze_python


class BoundSyntaxReport(DomainModel):
    """AST facts linked to one returned artifact and round identities."""

    report: SyntaxReport
    artifact_id: str = Field(min_length=1, max_length=128)
    work_unit_id: str = Field(min_length=1, max_length=128)
    run_id: str = Field(min_length=1, max_length=128)
    round_id: str | None = None
    snapshot_id: str | None = None

    @property
    def unknown(self) -> bool:
        return (not self.report.parse_ok) or self.report.unknown


def redact_local_paths(text: str, *roots: object) -> str:
    """Remove host roots from bounded public observations."""
    redacted = text
    candidates: list[str] = []
    for root in roots:
        if root is None:
            continue
        value = str(root)
        if value:
            candidates.append(value)
            candidates.append(value.replace("\\", "/"))
    for value in sorted(set(candidates), key=len, reverse=True):
        redacted = redacted.replace(value, "")
    return redacted


def source_pair_from_observation(
    payload: Mapping[str, object] | str | None,
) -> tuple[str | None, str | None]:
    """Extract before/after source from a bounded observation. Never reads a root."""
    if payload is None:
        return None, None
    if isinstance(payload, str):
        return "", payload
    before = _optional_source(payload.get("before") or payload.get("before_source"))
    after = _optional_source(
        payload.get("after")
        or payload.get("after_source")
        or payload.get("source")
        or payload.get("content")
        or payload.get("output")
    )
    if before is None and after is None:
        nested = payload.get("result")
        if isinstance(nested, Mapping):
            before, after = source_pair_from_observation(nested)
    if before is None and after is None:
        patch = payload.get("patch")
        if isinstance(patch, Mapping):
            diff = patch.get("unified_diff")
            if isinstance(diff, str) and diff.strip():
                before, after = _source_pair_from_unified_diff(diff)
    return before, after


def _source_pair_from_unified_diff(diff: str) -> tuple[str | None, str | None]:
    """Rebuild hunk text from a bounded unified diff. Never reads a workspace."""
    before_lines: list[str] = []
    after_lines: list[str] = []
    in_hunk = False
    for raw in diff.splitlines(keepends=True):
        if raw.startswith("@@"):
            in_hunk = True
            continue
        if raw.startswith(("---", "+++", "diff ", "index ")):
            in_hunk = False
            continue
        if not in_hunk:
            continue
        if raw.startswith("\\"):
            continue
        if raw.startswith("-"):
            before_lines.append(raw[1:])
        elif raw.startswith("+"):
            after_lines.append(raw[1:])
        elif raw.startswith(" "):
            before_lines.append(raw[1:])
            after_lines.append(raw[1:])
        else:
            in_hunk = False
    before = "".join(before_lines) if before_lines else None
    after = "".join(after_lines) if after_lines else None
    return before, after


def _optional_source(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def analyze_bounded_observation(
    *,
    artifact_id: str,
    work_unit_id: str,
    run_id: str,
    before_source: str | None = None,
    after_source: str | None = None,
    round_id: str | None = None,
    snapshot_id: str | None = None,
    language: str = "python",
) -> BoundSyntaxReport:
    """Parse returned text only. Missing or non-Python source stays unknown."""
    if not artifact_id.strip() or not work_unit_id.strip() or not run_id.strip():
        raise ValueError("AST analysis requires artifact, work unit and run scope")
    if language != "python":
        report = SyntaxReport(
            language=language,
            parse_ok=False,
            unknown=True,
            before_parse_error="unsupported language is unknown",
        )
    elif before_source is None and after_source is None:
        report = SyntaxReport(
            parse_ok=False,
            unknown=True,
            before_parse_error="source is absent",
        )
    else:
        report = analyze_python(before_source or "", after_source or "")
    return BoundSyntaxReport(
        report=report,
        artifact_id=artifact_id,
        work_unit_id=work_unit_id,
        run_id=run_id,
        round_id=round_id,
        snapshot_id=snapshot_id,
    )
