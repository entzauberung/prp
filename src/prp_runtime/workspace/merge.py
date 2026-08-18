"""Temporary Git three-way merge for isolated ChangeSet snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum, unique
from pathlib import Path
from typing import Literal

from pydantic import Field

from prp_runtime.domain.models import DomainModel
from prp_runtime.domain.values import new_snapshot_id
from prp_runtime.workspace.changes import ChangeSet, FileChangeAction
from prp_runtime.workspace.models import SnapshotEntry, SnapshotEntryType, SnapshotManifest

__all__ = [
    "GitMergeBackend",
    "MergeError",
    "MergeResult",
    "MergeStatus",
    "StagedChangeSet",
    "merge_change_sets",
    "merge_input_digest",
    "merge_candidate_manifest",
    "promote_merge",
]

_GIT_ENV = {
    "GIT_AUTHOR_NAME": "prp-runtime",
    "GIT_AUTHOR_EMAIL": "prp-runtime@invalid",
    "GIT_COMMITTER_NAME": "prp-runtime",
    "GIT_COMMITTER_EMAIL": "prp-runtime@invalid",
}
_MAX_CONFLICT_PATHS = 10_000


class MergeError(RuntimeError):
    """A staging merge or verified promotion cannot be completed."""


@unique
class MergeStatus(StrEnum):
    MERGED = "MERGED"
    CONFLICT = "CONFLICT"
    FAILED = "FAILED"
    VERIFICATION_FAILED = "VERIFICATION_FAILED"


@dataclass(frozen=True)
class StagedChangeSet:
    """A ChangeSet fact paired with its private complete snapshot root."""

    change_set: ChangeSet
    root: Path


class MergeResult(DomainModel):
    """Auditable merge outcome; no outcome is implicitly promoted."""

    dev_only: Literal[True]
    status: MergeStatus
    change_set_ids: tuple[str, ...]
    applied_change_set_ids: tuple[str, ...] = ()
    merged_snapshot_id: str | None = None
    merged_content_hash: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    staging_root: Path = Field(exclude=True)
    conflict_report: ConflictReport | None = None
    reason: str | None = None
    verified: bool = False
    promoted: bool = False


def merge_input_digest(base_snapshot_id: str, change_set_ids: Sequence[str]) -> str:
    """Return the stable idempotency digest for one merge input set."""
    ids = tuple(change_set_ids)
    if len(ids) != len(set(ids)):
        raise ValueError("merge input contains duplicate ChangeSet ids")
    payload = json.dumps(
        {
            "base_snapshot_id": base_snapshot_id,
            "change_set_ids": sorted(ids),
        },
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _safe_tree(root: Path) -> None:
    if root.is_symlink() or not root.is_dir():
        raise MergeError("merge snapshot root is not a directory")
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in (*directories, *files):
            if (current_path / name).is_symlink():
                raise MergeError("merge snapshots must not contain symbolic links")


def _file_facts(root: Path) -> dict[str, tuple[str, int]]:
    _safe_tree(root)
    facts: dict[str, tuple[str, int]] = {}
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories.sort()
        files.sort()
        current_path = Path(current)
        for name in files:
            path = current_path / name
            if not path.is_file():
                raise MergeError("merge snapshot contains an unsupported file")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            facts[path.relative_to(root).as_posix()] = (digest.hexdigest(), path.stat().st_size)
    return facts


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for relative, (file_hash, size) in sorted(_file_facts(root).items()):
        if relative == ".git" or relative.startswith(".git/"):
            continue
        digest.update(relative.encode("utf-8") + b"\0")
        digest.update(str(size).encode("ascii") + b"\0")
        digest.update(file_hash.encode("ascii") + b"\0")
    return digest.hexdigest()


def merge_candidate_manifest(root: Path) -> SnapshotManifest:
    """Build a durable payload manifest while excluding the temporary Git dir."""
    _safe_tree(root)
    entries: list[SnapshotEntry] = []
    for current, directories, files in os.walk(root, topdown=True, followlinks=False):
        directories[:] = sorted(name for name in directories if name != ".git")
        files[:] = sorted(name for name in files if name != ".git")
        current_path = Path(current)
        for name in directories:
            relative_directory = (current_path / name).relative_to(root).as_posix()
            entries.append(
                SnapshotEntry(
                    path=relative_directory,
                    sha256=hashlib.sha256(b"").hexdigest(),
                    size=0,
                    entry_type=SnapshotEntryType.DIRECTORY,
                )
            )
        for name in files:
            file_path = current_path / name
            relative = file_path.relative_to(root).as_posix()
            digest = hashlib.sha256()
            with file_path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            entries.append(
                SnapshotEntry(
                    path=relative,
                    sha256=digest.hexdigest(),
                    size=file_path.stat().st_size,
                    entry_type=SnapshotEntryType.FILE,
                )
            )
    return SnapshotManifest(entries=tuple(entries))


def _copy_payload(source: Path, destination: Path) -> None:
    _safe_tree(source)
    for child in destination.iterdir():
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()
    for child in source.iterdir():
        target = destination / child.name
        if child.name == ".git":
            continue
        if child.is_dir():
            shutil.copytree(child, target)
        else:
            shutil.copy2(child, target)


def _apply_changeset_payload(staged: StagedChangeSet, destination: Path) -> None:
    """Apply only declared file facts from a complete staged snapshot."""
    for change in staged.change_set.files:
        source = staged.root / change.path
        target = destination / change.path
        if change.action is FileChangeAction.DELETE:
            if not target.is_file() or target.is_symlink():
                raise MergeError("ChangeSet delete target is unavailable")
            target.unlink()
            continue
        if not source.is_file() or source.is_symlink():
            raise MergeError("ChangeSet source file is unavailable")
        if target.exists() and not target.is_file():
            raise MergeError("ChangeSet target is not a file")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def _git_failure(label: str) -> MergeError:
    """Return a bounded failure without persisting raw Git stderr."""
    return MergeError(label)


def _roots_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    )


def _conflict_paths(stdout: str) -> tuple[str, ...] | None:
    paths: list[str] = []
    for line in stdout.splitlines()[:_MAX_CONFLICT_PATHS]:
        path = line.strip()
        if not path:
            continue
        if (
            len(path) > 1024
            or path.startswith(("/", "\\"))
            or re.match(r"^[A-Za-z]:", path)
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
        ):
            return None
        paths.append(path)
    return tuple(paths)


def _require_git(result: subprocess.CompletedProcess[str], label: str) -> None:
    if result.returncode != 0:
        raise _git_failure(label)


def _run_git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(_GIT_ENV)
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=repo,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError as error:
        raise MergeError("git backend is unavailable") from error


def _failed(
    status: MergeStatus,
    ids: tuple[str, ...],
    staging_root: Path,
    reason: str,
    *,
    applied: tuple[str, ...] = (),
    report: ConflictReport | None = None,
) -> MergeResult:
    return MergeResult(
        dev_only=True,
        status=status,
        change_set_ids=ids,
        applied_change_set_ids=applied,
        staging_root=staging_root,
        conflict_report=report,
        reason=reason,
    )


def _validate_changeset(base_root: Path, staged: StagedChangeSet) -> None:
    before = _file_facts(base_root)
    after = _file_facts(staged.root)
    declared = {change.path for change in staged.change_set.files}
    changed = {
        path for path in set(before) | set(after) if before.get(path) != after.get(path)
    }
    if changed != declared:
        raise MergeError("ChangeSet file facts do not match its isolated snapshot")
    for change in staged.change_set.files:
        before_fact = before.get(change.path)
        after_fact = after.get(change.path)
        expected_before = (
            None if change.before is None else (change.before.sha256, change.before.size)
        )
        expected_after = (
            None if change.after is None else (change.after.sha256, change.after.size)
        )
        if before_fact != expected_before or after_fact != expected_after:
            raise MergeError("ChangeSet content facts do not match its isolated snapshot")
        if change.action is FileChangeAction.ADD and before_fact is not None:
            raise MergeError("added ChangeSet file already exists in base snapshot")
        if change.action is FileChangeAction.DELETE and after_fact is not None:
            raise MergeError("deleted ChangeSet file remains in isolated snapshot")


class GitMergeBackend:
    """Use only a caller-provided temporary repository as merge state."""

    def merge(
        self,
        base_root: Path,
        changes: Sequence[StagedChangeSet],
        *,
        staging_root: Path,
        verify: Callable[[Path], None] | None = None,
    ) -> MergeResult:
        ordered_changes = tuple(
            sorted(changes, key=lambda staged: staged.change_set.change_set_id)
        )
        ids = tuple(staged.change_set.change_set_id for staged in ordered_changes)
        roots = (base_root, staging_root, *(staged.root for staged in ordered_changes))
        if any(
            _roots_overlap(left, right)
            for index, left in enumerate(roots)
            for right in roots[index + 1 :]
        ):
            return _failed(MergeStatus.FAILED, ids, staging_root, "merge roots must be distinct")
        if not ordered_changes:
            return _failed(
                MergeStatus.FAILED,
                ids,
                staging_root,
                "merge requires at least one ChangeSet",
            )
        if len(ids) != len(set(ids)):
            return _failed(MergeStatus.FAILED, ids, staging_root, "duplicate ChangeSet id")
        try:
            _safe_tree(base_root)
            for staged in ordered_changes:
                _validate_changeset(base_root, staged)
        except MergeError as error:
            return _failed(MergeStatus.FAILED, ids, staging_root, str(error))
        base_ids = {staged.change_set.base_snapshot_id for staged in ordered_changes}
        if len(base_ids) > 1:
            report = ConflictReport(
                kind=ConflictKind.BASE,
                reason="ChangeSets do not share one base snapshot",
                facts=(
                    ConflictFact(
                        kind=ConflictKind.BASE,
                        reason="ChangeSets do not share one base snapshot",
                    ),
                ),
            )
            return _failed(MergeStatus.CONFLICT, ids, staging_root, report.reason, report=report)
        run_ids = {staged.change_set.run_id for staged in ordered_changes}
        workspace_ids = {staged.change_set.workspace_id for staged in ordered_changes}
        if len(run_ids) > 1 or len(workspace_ids) > 1:
            report = ConflictReport(
                kind=ConflictKind.UNKNOWN,
                reason="ChangeSets do not share one run and workspace scope",
                facts=(
                    ConflictFact(
                        kind=ConflictKind.UNKNOWN,
                        reason="ChangeSets do not share one run and workspace scope",
                    ),
                ),
            )
            return _failed(MergeStatus.CONFLICT, ids, staging_root, report.reason, report=report)
        for index, left in enumerate(ordered_changes):
            for right in ordered_changes[index + 1 :]:
                report = classify_conflict(left.change_set, right.change_set)
                if report.conflict:
                    return _failed(
                        MergeStatus.CONFLICT,
                        ids,
                        staging_root,
                        report.reason,
                        report=report,
                    )
        try:
            if staging_root.exists():
                if not staging_root.is_dir() or any(staging_root.iterdir()):
                    raise MergeError("staging root must be a new empty directory")
            else:
                staging_root.mkdir(parents=True)
            _copy_payload(base_root, staging_root)
            init = _run_git(staging_root, "init", "--quiet")
            if init.returncode != 0:
                raise _git_failure("git init failed")
            for key, value in (("user.name", "prp-runtime"), ("user.email", "prp-runtime@invalid")):
                config = _run_git(staging_root, "config", key, value)
                if config.returncode != 0:
                    raise _git_failure("git config failed")
            _require_git(_run_git(staging_root, "add", "--all"), "base index failed")
            commit = _run_git(staging_root, "commit", "--quiet", "--allow-empty", "-m", "base")
            if commit.returncode != 0:
                raise _git_failure("base commit failed")
            branch = _run_git(staging_root, "branch", "-M", "staging")
            if branch.returncode != 0:
                raise _git_failure("staging branch failed")
            base_commit = _run_git(staging_root, "rev-parse", "HEAD").stdout.strip()
            applied: list[str] = []
            for index, staged in enumerate(ordered_changes):
                checkout = _run_git(
                    staging_root, "checkout", "--quiet", "-b", f"change-{index}", base_commit
                )
                if checkout.returncode != 0:
                    raise _git_failure("change branch failed")
                _apply_changeset_payload(staged, staging_root)
                _require_git(_run_git(staging_root, "add", "--all"), "change index failed")
                commit = _run_git(
                    staging_root,
                    "commit",
                    "--quiet",
                    "--allow-empty",
                    "-m",
                    f"changeset {staged.change_set.change_set_id}",
                )
                if commit.returncode != 0:
                    raise _git_failure("change commit failed")
                checkout = _run_git(staging_root, "checkout", "--quiet", "staging")
                if checkout.returncode != 0:
                    raise _git_failure("staging checkout failed")
                merged = _run_git(
                    staging_root,
                    "merge",
                    "--no-ff",
                    "--no-commit",
                    f"change-{index}",
                )
                if merged.returncode != 0:
                    conflict_diff = _run_git(
                        staging_root, "diff", "--name-only", "--diff-filter=U"
                    )
                    paths = _conflict_paths(conflict_diff.stdout)
                    abort = _run_git(staging_root, "merge", "--abort")
                    if abort.returncode != 0:
                        return _failed(
                            MergeStatus.FAILED,
                            ids,
                            staging_root,
                            "git merge cleanup failed",
                            applied=tuple(applied),
                        )
                    conflict_kind = (
                        ConflictKind.PATH
                        if conflict_diff.returncode == 0 and paths is not None
                        else ConflictKind.UNKNOWN
                    )
                    conflict_reason = (
                        "Git three-way merge reported a conflict"
                        if conflict_kind is ConflictKind.PATH
                        else "Git conflict paths were unavailable"
                    )
                    report = ConflictReport(
                        kind=conflict_kind,
                        reason=conflict_reason,
                        facts=(
                            ConflictFact(
                                kind=conflict_kind,
                                reason=conflict_reason,
                                paths=() if paths is None else paths,
                            ),
                        ),
                    )
                    return _failed(
                        MergeStatus.CONFLICT,
                        ids,
                        staging_root,
                        report.reason,
                        applied=tuple(applied),
                        report=report,
                    )
                commit = _run_git(
                    staging_root,
                    "commit",
                    "--quiet",
                    "-m",
                    f"merge {staged.change_set.change_set_id}",
                )
                if commit.returncode != 0:
                    raise _git_failure("merge commit failed")
                applied.append(staged.change_set.change_set_id)
            if verify is not None:
                try:
                    verify(staging_root)
                except Exception:
                    return _failed(
                        MergeStatus.VERIFICATION_FAILED,
                        ids,
                        staging_root,
                        "post-merge verification failed",
                        applied=tuple(applied),
                    )
            return MergeResult(
                dev_only=True,
                status=MergeStatus.MERGED,
                change_set_ids=ids,
                applied_change_set_ids=tuple(applied),
                merged_snapshot_id=new_snapshot_id(),
                merged_content_hash=_tree_hash(staging_root),
                staging_root=staging_root,
                verified=verify is not None,
            )
        except MergeError as error:
            return _failed(
                MergeStatus.FAILED,
                ids,
                staging_root,
                str(error),
            )


def promote_merge(result: MergeResult, destination: Path) -> MergeResult:
    """Promote only a verified, unchanged merge into a new destination snapshot."""
    if result.status is not MergeStatus.MERGED:
        raise MergeError("only a verified merged result can be promoted")
    if not result.verified:
        raise MergeError("only a verified merged result can be promoted")
    if result.promoted:
        expected_hash = result.merged_content_hash
        if expected_hash is None:
            raise MergeError("promoted candidate has no content hash")
        destination = Path(destination)
        if destination.is_symlink() or not destination.is_dir():
            raise MergeError("replayed promotion destination is unavailable")
        if _tree_hash(destination) != expected_hash:
            raise MergeError("replayed promotion destination changed")
        return result
    expected_hash = result.merged_content_hash
    if expected_hash is None:
        raise MergeError("merged candidate has no content hash")

    staging_root = result.staging_root
    _safe_tree(staging_root)
    if _tree_hash(staging_root) != expected_hash:
        raise MergeError("merged candidate changed before promotion")

    destination = Path(destination)
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise MergeError("promotion destination must be a directory")
    parent = destination.parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise MergeError("promotion destination parent is unavailable") from error
    if parent.is_symlink() or not parent.is_dir():
        raise MergeError("promotion destination parent is not a safe directory")
    resolved_staging = staging_root.resolve()
    resolved_destination = destination.resolve(strict=False)
    if (
        resolved_staging == resolved_destination
        or resolved_staging in resolved_destination.parents
        or resolved_destination in resolved_staging.parents
    ):
        raise MergeError("promotion source and destination overlap")

    temporary: Path | None = None
    try:
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.promotion-", dir=parent)
        )
        _copy_payload(staging_root, temporary)
        if _tree_hash(temporary) != expected_hash:
            raise MergeError("promotion candidate failed content verification")
        if destination.exists():
            raise MergeError("promotion destination must be new")
        os.replace(temporary, destination)
        temporary = None
    except OSError as error:
        raise MergeError("atomic promotion failed") from error
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
    return result.model_copy(update={"promoted": True})


merge_change_sets = GitMergeBackend().merge

# Import after this module's public symbols exist: runtime.__init__ re-exports
# the coordinator, which also imports this merge backend.
from prp_runtime.runtime.conflicts import (  # noqa: E402  # isort: skip
    ConflictFact,
    ConflictKind,
    ConflictReport,
    classify_conflict,
)

MergeResult.model_rebuild()
