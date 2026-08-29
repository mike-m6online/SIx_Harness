#!/usr/bin/env python3
"""PostToolUse hook: auto-stage and commit files after Edit/Write.

Reads the tool input from stdin (JSON), extracts the file path, stages it,
and commits with a descriptive message ("auto: update <relative path>").
This implements the harness's "git is the rollback net" discipline: every
harness-relevant edit lands as its own commit so any change can be rolled
back without archaeology.

Only commits files under the configured --tracked-dir entries. Skips files
outside the project root and everything else. Once the flags parse, the
hook NEVER fails the tool call: every git error, timeout, or malformed
stdin payload is swallowed and the hook exits 0 (a PostToolUse hook that
could error would block the harness on transient git states such as
lock-file contention).

Harness-kit parameterization (deltas from the origin-project original):
  - `--project-root` is a REQUIRED flag (was hardcoded to the origin
    project's root); omitting it exits 2 with argparse's standard clear
    message.
  - The tracked list (was hardcoded to the origin project's dirs) is
    supplied via repeatable REQUIRED `--tracked-dir` flags. Matching
    semantics are the original's exactly: each entry is compared against
    the project-relative
    forward-slash path with `startswith(entry) or == entry`, so entries
    ending in "/" act as directory prefixes and bare names match the exact
    file (or any path extending it, as in the original).
  - Path containment uses case-normalized comparison (correct on Windows
    where the original's naive string replace missed drive-letter case
    differences); the commit message keeps the caller's casing.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence

GIT_ADD_TIMEOUT_S = 5
GIT_DIFF_TIMEOUT_S = 5
GIT_COMMIT_TIMEOUT_S = 10


def _relative_to_root(file_path: str, project_root: str) -> Optional[str]:
    """The path of `file_path` relative to `project_root`, or None when the
    file is not strictly inside the root. Containment is decided on
    case-normalized absolute paths (Windows-correct); the returned relative
    path preserves the caller's original casing for the commit message."""
    try:
        root = os.path.normpath(os.path.abspath(project_root))
        target = os.path.normpath(os.path.abspath(file_path))
    except (OSError, ValueError):
        return None
    root_cmp = os.path.normcase(root).rstrip(os.sep)
    target_cmp = os.path.normcase(target)
    if not target_cmp.startswith(root_cmp + os.sep):
        return None
    return target[len(root_cmp) + 1:]


def _is_tracked(rel_posix: str, tracked: Sequence[str]) -> bool:
    """Original matching rule: prefix-or-exact against the tracked entries."""
    return any(rel_posix.startswith(d) or rel_posix == d for d in tracked)


def _run_git(args: List[str], cwd: str, timeout_s: int
             ) -> Optional[subprocess.CompletedProcess]:
    """Run a git command, returning None on any failure to launch or
    timeout. Never raises -- the hook must not block the tool call."""
    try:
        return subprocess.run(
            ["git"] + args,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="git_auto_commit",
        description="PostToolUse hook: auto-stage + commit edited files "
                    "under the tracked dirs. Both flags are REQUIRED: the "
                    "harness init tool bakes concrete values into the "
                    "installed hook command.",
    )
    p.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the git repository root to commit into. "
             "REQUIRED.",
    )
    p.add_argument(
        "--tracked-dir",
        action="append",
        required=True,
        metavar="ENTRY",
        help="Project-relative path entry to auto-commit (repeatable, at "
             "least one REQUIRED). Entries ending in '/' are directory "
             "prefixes (e.g. 'src/'); bare names match the exact file "
             "(e.g. 'CLAUDE.md').",
    )
    return p


def process(stdin_text: str, project_root: str,
            tracked_dirs: Sequence[str]) -> None:
    """The hook body: parse the PostToolUse payload and stage+commit the
    edited file when it falls under a tracked entry. Silent on every
    failure path -- see the module docstring's never-block contract."""
    try:
        data = json.loads(stdin_text)
    except (json.JSONDecodeError, ValueError):
        return
    if not isinstance(data, dict):
        return

    tool_input = data.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return
    file_path = tool_input.get("file_path", "")
    if not file_path or not isinstance(file_path, str):
        return

    # Normalize separators before containment/matching (hook payloads on
    # Windows carry backslash paths).
    file_path = file_path.replace("\\", "/")

    rel = _relative_to_root(file_path, project_root)
    if rel is None:
        return
    rel_posix = rel.replace("\\", "/")
    if not _is_tracked(rel_posix, tracked_dirs):
        return

    # Stage the file.
    if _run_git(["add", file_path], cwd=project_root,
                timeout_s=GIT_ADD_TIMEOUT_S) is None:
        return

    # Check if there are staged changes.
    result = _run_git(["diff", "--cached", "--name-only"], cwd=project_root,
                      timeout_s=GIT_DIFF_TIMEOUT_S)
    if result is None or not result.stdout.strip():
        return

    # Commit with the relative path as the message.
    msg = f"auto: update {rel_posix}"
    _run_git(["commit", "-m", msg], cwd=project_root,
             timeout_s=GIT_COMMIT_TIMEOUT_S)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        stdin_text = sys.stdin.read()
    except (OSError, UnicodeDecodeError):
        return 0
    process(stdin_text, args.project_root, tuple(args.tracked_dir))
    return 0  # never block the tool call once the flags parse


if __name__ == "__main__":
    raise SystemExit(main())
