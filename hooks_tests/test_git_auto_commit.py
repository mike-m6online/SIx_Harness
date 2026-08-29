"""Tests for hooks/git_auto_commit.py -- flag contract, tracked-dir
matching (dir prefixes, exact files, backslash payloads), containment
(files outside the root never committed), and the never-fail contract."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )


def _commit_count(repo: Path) -> int:
    proc = _git(repo, "rev-list", "--count", "HEAD")
    if proc.returncode != 0:  # no commits yet
        return 0
    return int(proc.stdout.strip())


def _last_subject(repo: Path) -> str:
    proc = _git(repo, "log", "-1", "--pretty=%s")
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = tmp_path / "proj"
    (repo / "src").mkdir(parents=True)
    assert _git(repo, "init", "-q").returncode == 0
    assert _git(repo, "config", "user.email", "test@example.com").returncode == 0
    assert _git(repo, "config", "user.name", "Hook Test").returncode == 0
    # Neutralize any global signing config -- the hook must commit unattended.
    _git(repo, "config", "commit.gpgsign", "false")
    return repo


def _payload(file_path: str) -> str:
    return json.dumps({
        "tool_name": "Edit",
        "tool_input": {"file_path": file_path},
    })


def test_no_flags_exits_2(run_hook):
    proc = run_hook("git_auto_commit.py")
    assert proc.returncode == 2
    assert "required" in proc.stderr.lower()


def test_missing_tracked_dir_exits_2(run_hook, tmp_path):
    proc = run_hook("git_auto_commit.py", "--project-root", tmp_path)
    assert proc.returncode == 2
    assert "--tracked-dir" in proc.stderr


def test_missing_project_root_exits_2(run_hook):
    proc = run_hook("git_auto_commit.py", "--tracked-dir", "src/")
    assert proc.returncode == 2
    assert "--project-root" in proc.stderr


def test_tracked_file_is_committed(run_hook, repo):
    target = repo / "src" / "foo.py"
    target.write_text("x = 1\n", encoding="utf-8")
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        stdin_text=_payload(str(target)),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 1
    assert _last_subject(repo) == "auto: update src/foo.py"


def test_backslash_payload_path_is_committed(run_hook, repo):
    target = repo / "src" / "bar.py"
    target.write_text("y = 2\n", encoding="utf-8")
    backslashed = str(target).replace("/", "\\")
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        stdin_text=_payload(backslashed),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 1
    assert _last_subject(repo) == "auto: update src/bar.py"


def test_exact_file_entry_is_committed(run_hook, repo):
    target = repo / "CLAUDE.md"
    target.write_text("# instructions\n", encoding="utf-8")
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        "--tracked-dir", "CLAUDE.md",
        stdin_text=_payload(str(target)),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 1
    assert _last_subject(repo) == "auto: update CLAUDE.md"


def test_untracked_path_is_skipped(run_hook, repo):
    (repo / "docs").mkdir()
    target = repo / "docs" / "note.md"
    target.write_text("note\n", encoding="utf-8")
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        stdin_text=_payload(str(target)),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 0


def test_file_outside_root_is_skipped(run_hook, repo, tmp_path):
    outside = tmp_path / "outside.py"
    outside.write_text("z = 3\n", encoding="utf-8")
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        # An entry that would prefix-match the bare filename if containment
        # were skipped -- it must not be committed regardless.
        "--tracked-dir", "outside.py",
        stdin_text=_payload(str(outside)),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 0


def test_garbage_stdin_never_fails(run_hook, repo):
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        stdin_text="this is not json {",
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 0


def test_payload_without_file_path_is_noop(run_hook, repo):
    proc = run_hook(
        "git_auto_commit.py",
        "--project-root", repo,
        "--tracked-dir", "src/",
        stdin_text=json.dumps({"tool_name": "Edit", "tool_input": {}}),
    )
    assert proc.returncode == 0
    assert _commit_count(repo) == 0
