"""Contract tests for Python AST symbol facts."""

import pytest

from prp_runtime.analysis.syntax import (
    SymbolChangeAction,
    SymbolKind,
    SyntaxAnalyzer,
    analyze_bounded_observation,
    analyze_python,
    redact_local_paths,
    source_pair_from_observation,
)
from prp_runtime.domain.models import Artifact, ArtifactKind, new_artifact_id
from prp_runtime.domain.values import new_attempt_id, new_run_id, new_work_unit_id
from prp_runtime.runtime.bridge import syntax_facts_from_bridge_artifact
from prp_runtime.runtime.conflicts import ConflictKind, classify_syntax_conflict
from prp_runtime.workspace.changes import FileChange, FileChangeAction, FileContent


def changes_by_key(source_before: str, source_after: str):
    return {change.key: change for change in analyze_python(source_before, source_after).changes}


def test_nested_symbols_and_spans_are_stable_without_execution() -> None:
    report = analyze_python(
        "class Outer:\n    def old(self):\n        return 1\n",
        "class Outer:\n    def new(self):\n        return 2\n",
    )

    assert report.parse_ok is True
    changes = {change.key: change for change in report.changes}
    assert changes["function:Outer.old"].action is SymbolChangeAction.DELETE
    assert changes["function:Outer.new"].action is SymbolChangeAction.ADD
    assert report.symbols[0].kind is SymbolKind.CLASS
    assert report.symbols[0].span.start_line == 1
    assert all(change.before is not None or change.after is not None for change in report.changes)


def test_import_assignment_and_function_modifications_are_diffed() -> None:
    before = "import os\nvalue = 1\ndef run():\n    return value\n"
    after = "import sys\nvalue = 2\ndef run():\n    return value + 1\n"

    changes = changes_by_key(before, after)

    assert changes["import:os"].action is SymbolChangeAction.DELETE
    assert changes["import:sys"].action is SymbolChangeAction.ADD
    assert changes["assignment:value"].action is SymbolChangeAction.MODIFY
    assert changes["function:run"].action is SymbolChangeAction.MODIFY


def test_class_import_from_and_nested_assignments_have_qualified_keys() -> None:
    report = analyze_python(
        "from pkg import value as imported\n"
        "class Outer:\n"
        "    inner = 1\n"
        "    def run(self):\n"
        "        nested = 2\n",
        "from pkg import value as imported\n"
        "class Outer:\n"
        "    inner = 1\n"
        "    def run(self):\n"
        "        nested = 3\n",
    )

    keys = {change.key for change in report.changes}
    assert "assignment:Outer.run.nested" in keys
    assert "function:Outer.run" in keys
    assert next(
        change
        for change in report.changes
        if change.key == "assignment:Outer.run.nested"
    ).action is SymbolChangeAction.MODIFY


def test_parse_failure_is_unknown_and_does_not_guess_changes() -> None:
    report = analyze_python("def broken(:\n    pass\n", "def fixed():\n    pass\n")

    assert report.parse_ok is False
    assert report.unknown is True
    assert report.parse_error is not None
    assert report.changes == ()
    assert report.symbols == ()


def test_dynamic_python_effects_are_unknown_without_leaking_source_facts() -> None:
    report = analyze_python(
        "from package import *\nvalue = eval('1')\n",
        "from package import *\nvalue = eval('2')\n",
    )

    assert report.parse_ok is True
    assert report.unknown is True
    assert report.parse_error is None
    assert report.changes
    assert "eval" not in report.model_dump_json()


def test_cross_scope_global_effects_are_unknown() -> None:
    report = analyze_python(
        "def update():\n    global value\n    value = 1\n",
        "def update():\n    global value\n    value = 2\n",
    )

    assert report.parse_ok is True
    assert report.unknown is True


def test_unsupported_binding_constructs_are_unknown() -> None:
    for source in (
        "for value in values:\n    pass\n",
        "with manager() as value:\n    pass\n",
        "try:\n    pass\nexcept Exception as error:\n    pass\n",
        "match value:\n    case item:\n        pass\n",
        "result = (item := value)\n",
        "del value\n",
    ):
        report = analyze_python(source, source)
        assert report.parse_ok is True
        assert report.unknown is True


def test_analyzer_facade_matches_function_and_empty_diff_is_clean() -> None:
    source = "class Ready:\n    pass\n"
    report = SyntaxAnalyzer().analyze(source, source)

    assert report == analyze_python(source, source)
    assert report.changes == ()


def file_change(path: str) -> FileChange:
    content = FileContent(sha256="a" * 64, size=1)
    return FileChange(
        path=path,
        action=FileChangeAction.MODIFY,
        before=content,
        after=content,
    )


def test_same_symbol_is_structural_but_disjoint_symbols_are_candidates() -> None:
    left = analyze_python("def run():\n    return 1\n", "def run():\n    return 2\n")
    same = analyze_python("def run():\n    return 1\n", "def run():\n    return 3\n")
    disjoint = analyze_python("def other():\n    return 1\n", "def other():\n    return 2\n")

    structural = classify_syntax_conflict(
        file_change("src/app.py"), file_change("src/app.py"), left_report=left, right_report=same
    )
    candidate = classify_syntax_conflict(
        file_change("src/app.py"),
        file_change("src/app.py"),
        left_report=left,
        right_report=disjoint,
    )

    assert structural.kind is ConflictKind.STRUCTURAL
    assert structural.facts[0].symbols == ("function:run",)
    assert candidate.kind is ConflictKind.NO_CONFLICT
    assert "semantic compatibility remains unverified" in candidate.reason


def test_global_or_import_change_is_conservative_and_fallback_is_safe() -> None:
    global_change = analyze_python("setting = 1\n", "setting = 2\n")
    function_change = analyze_python("def run():\n    return 1\n", "def run():\n    return 2\n")
    parse_error = analyze_python("def broken(:\n    pass\n", "def run():\n    pass\n")
    non_python = parse_error.model_copy(
        update={"language": "javascript", "parse_ok": True, "unknown": False}
    )

    assert (
        classify_syntax_conflict(
            file_change("src/app.py"),
            file_change("src/app.py"),
            left_report=global_change,
            right_report=function_change,
        ).kind
        is ConflictKind.STRUCTURAL
    )
    assert (
        classify_syntax_conflict(
            file_change("src/app.py"),
            file_change("src/app.py"),
            left_report=parse_error,
            right_report=function_change,
        ).kind
        is ConflictKind.UNKNOWN
    )
    assert (
        classify_syntax_conflict(
            file_change("src/app.js"),
            file_change("src/app.js"),
            left_report=non_python,
            right_report=non_python,
        ).kind
        is ConflictKind.PATH
    )


def test_bounded_observation_requires_artifact_scope_and_links_identities() -> None:
    with pytest.raises(ValueError, match="artifact"):
        analyze_bounded_observation(
            artifact_id=" ",
            work_unit_id="wu_scope",
            run_id="run_scope",
            after_source="x = 1\n",
        )
    bound = analyze_bounded_observation(
        artifact_id="art_scope",
        work_unit_id="wu_scope",
        run_id="run_scope",
        round_id="round_" + "a" * 32,
        snapshot_id="snap_" + "a" * 32,
        after_source="def run():\n    return 1\n",
    )
    assert bound.report.parse_ok is True
    assert bound.unknown is False
    assert bound.artifact_id == "art_scope"
    assert bound.work_unit_id == "wu_scope"
    assert bound.run_id == "run_scope"
    assert bound.round_id.endswith("a" * 32)
    assert bound.snapshot_id is not None
    again = analyze_bounded_observation(
        artifact_id="art_scope",
        work_unit_id="wu_scope",
        run_id="run_scope",
        after_source="def run():\n    return 1\n",
    )
    assert again.report.symbols == bound.report.symbols


def test_absent_or_dynamic_observation_is_unknown_not_proof() -> None:
    absent = analyze_bounded_observation(
        artifact_id="art_missing",
        work_unit_id="wu_missing",
        run_id="run_missing",
    )
    assert absent.report.parse_ok is False
    assert absent.unknown is True
    dynamic = analyze_bounded_observation(
        artifact_id="art_dyn",
        work_unit_id="wu_dyn",
        run_id="run_dyn",
        after_source="value = eval('1')\n",
    )
    assert dynamic.unknown is True
    unsupported = analyze_bounded_observation(
        artifact_id="art_js",
        work_unit_id="wu_js",
        run_id="run_js",
        language="javascript",
        after_source="const x = 1;",
    )
    assert unsupported.unknown is True


def test_bridge_artifact_adapter_parses_returned_text_without_a_root() -> None:
    import json

    snapshot_id = "snap_" + "b" * 32
    artifact = Artifact(
        artifact_id=new_artifact_id(),
        run_id=new_run_id(),
        work_unit_id=new_work_unit_id(),
        attempt_id=new_attempt_id(),
        name="bridge-result",
        kind=ArtifactKind.JSON,
        content=json.dumps(
            {
                "content": "def run():\n    return 1\n",
                "snapshot_id": snapshot_id,
            }
        ),
    )
    bound = syntax_facts_from_bridge_artifact(artifact)
    assert bound.artifact_id == artifact.artifact_id
    assert bound.work_unit_id == artifact.work_unit_id
    assert bound.run_id == artifact.run_id
    assert bound.snapshot_id == snapshot_id
    assert bound.report.parse_ok is True
    assert any(symbol.name == "run" for symbol in bound.report.symbols)
    before, after = source_pair_from_observation({"content": "x = 1\n"})
    assert before is None
    assert after == "x = 1\n"
    patch_before, patch_after = source_pair_from_observation(
        {
            "result": {
                "patch": {
                    "base_snapshot_id": snapshot_id,
                    "unified_diff": (
                        "--- a/app.py\n"
                        "+++ b/app.py\n"
                        "@@ -1,2 +1,2 @@\n"
                        " def answer() -> int:\n"
                        "-    return 1\n"
                        "+    return 2\n"
                    ),
                }
            }
        }
    )
    assert patch_before == "def answer() -> int:\n    return 1\n"
    assert patch_after == "def answer() -> int:\n    return 2\n"
    patched = syntax_facts_from_bridge_artifact(
        Artifact(
            artifact_id=new_artifact_id(),
            run_id=new_run_id(),
            work_unit_id=new_work_unit_id(),
            attempt_id=new_attempt_id(),
            name="bridge-result",
            kind=ArtifactKind.JSON,
            content=json.dumps(
                {
                    "result": {
                        "patch": {
                            "base_snapshot_id": snapshot_id,
                            "unified_diff": (
                                "--- a/app.py\n"
                                "+++ b/app.py\n"
                                "@@ -1,2 +1,2 @@\n"
                                " def answer() -> int:\n"
                                "-    return 1\n"
                                "+    return 2\n"
                            ),
                        }
                    },
                    "snapshot_id": snapshot_id,
                }
            ),
        )
    )
    assert patched.report.parse_ok is True
    assert any(symbol.name == "answer" for symbol in patched.report.symbols)
    assert "/tmp/project" not in redact_local_paths(
        "/tmp/project/src/main.py", "/tmp/project"
    )
