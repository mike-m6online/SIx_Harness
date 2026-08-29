"""Tests for hooks/memory_decay.py -- flag contract, archive mutation,
report-only detections, and the parameterized (no-default) path handling."""
from __future__ import annotations

from pathlib import Path

MEMORY_TEXT = """# Project Memory - Fixture

Preamble line before any section.

## ⏯ LATEST (2026-08-18) — RESUME HERE
- [THE ANCHOR](checkpoint_current.md) — read it first.
- [foo topic](foo.md) and linked again: [foo again](foo.md)

## SUPERSEDED HOOK 2026-08-01 — old resume anchor
- old content line 1
- old content line 2

## INVARIANTS
- [the one invariant](invariant_one.md)
"""


def _build_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A minimal (project_root, memory_dir) pair: MEMORY.md with one
    SUPERSEDED section, one duplicated link, one orphan topic file, and one
    underscore-prefixed file that must be excluded from orphan listing."""
    project_root = tmp_path / "proj"
    project_root.mkdir()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text(MEMORY_TEXT, encoding="utf-8")
    (memory_dir / "checkpoint_current.md").write_text(
        "# current checkpoint\n", encoding="utf-8")
    (memory_dir / "foo.md").write_text("# foo\n", encoding="utf-8")
    (memory_dir / "invariant_one.md").write_text("# inv\n", encoding="utf-8")
    (memory_dir / "orphan_topic.md").write_text("# orphan\n", encoding="utf-8")
    (memory_dir / "_private.md").write_text("# private\n", encoding="utf-8")
    return project_root, memory_dir


def test_no_flags_exits_2(run_hook):
    proc = run_hook("memory_decay.py")
    assert proc.returncode == 2
    assert "required" in proc.stderr.lower()


def test_missing_memory_dir_exits_2(run_hook, tmp_path):
    proc = run_hook("memory_decay.py", "--project-root", tmp_path)
    assert proc.returncode == 2
    assert "--memory-dir" in proc.stderr


def test_missing_project_root_exits_2(run_hook, tmp_path):
    proc = run_hook("memory_decay.py", "--memory-dir", tmp_path)
    assert proc.returncode == 2
    assert "--project-root" in proc.stderr


def test_print_archives_superseded_section(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    proc = run_hook(
        "memory_decay.py",
        "--project-root", project_root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    assert proc.stdout.startswith("memory_decay -- MEMORY.md maintenance report")
    assert "SUPERSEDED sections found: 1" in proc.stdout
    assert "archiving" in proc.stdout

    # The section moved: gone from MEMORY.md, present verbatim in the archive.
    memory_text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "SUPERSEDED HOOK" not in memory_text
    assert "old content line 1" not in memory_text
    # Non-superseded content untouched.
    assert "## INVARIANTS" in memory_text
    assert "Preamble line before any section." in memory_text

    archive = memory_dir / "MEMORY_ARCHIVE.md"
    assert archive.is_file()
    archive_text = archive.read_text(encoding="utf-8")
    assert "SUPERSEDED HOOK 2026-08-01" in archive_text
    assert "old content line 1" in archive_text
    assert "old content line 2" in archive_text
    assert archive_text.startswith("# MEMORY ARCHIVE")


def test_second_run_is_noop(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    flags = ("--project-root", project_root, "--memory-dir", memory_dir,
             "--print")
    run_hook("memory_decay.py", *flags)
    memory_after_first = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    archive_after_first = (memory_dir / "MEMORY_ARCHIVE.md").read_text(
        encoding="utf-8")

    proc = run_hook("memory_decay.py", *flags)
    assert proc.returncode == 0
    assert "SUPERSEDED sections found: 0 (no-op)" in proc.stdout
    assert (memory_dir / "MEMORY.md").read_text(
        encoding="utf-8") == memory_after_first
    assert (memory_dir / "MEMORY_ARCHIVE.md").read_text(
        encoding="utf-8") == archive_after_first


def test_dry_run_writes_nothing(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    proc = run_hook(
        "memory_decay.py",
        "--project-root", project_root,
        "--memory-dir", memory_dir,
        "--dry-run",
    )
    assert proc.returncode == 0
    assert "dry-run, NOT archiving" in proc.stdout
    assert (memory_dir / "MEMORY.md").read_text(
        encoding="utf-8") == MEMORY_TEXT
    assert not (memory_dir / "MEMORY_ARCHIVE.md").exists()


def test_dup_links_and_orphans_reported(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    proc = run_hook(
        "memory_decay.py",
        "--project-root", project_root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    # foo.md is linked twice -> duplicate report.
    assert "duplicate index links (appear >1x):" in proc.stdout
    assert "foo.md  (x2)" in proc.stdout
    # orphan_topic.md is unlinked -> orphan candidate; underscore files and
    # linked files are excluded.
    assert "orphan_topic.md" in proc.stdout
    assert "_private.md" not in proc.stdout
    assert "indexed-orphan candidates (1 *.md files" in proc.stdout


def test_budget_warning_when_over(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    proc = run_hook(
        "memory_decay.py",
        "--project-root", project_root,
        "--memory-dir", memory_dir,
        "--print",
        "--max-lines", 2,
    )
    assert proc.returncode == 0
    assert "WARNING: MEMORY.md is OVER BUDGET" in proc.stdout


def test_budget_ok_under_defaults(run_hook, tmp_path):
    project_root, memory_dir = _build_fixture(tmp_path)
    proc = run_hook(
        "memory_decay.py",
        "--project-root", project_root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    assert "budget: OK (under both limits)" in proc.stdout


def test_missing_memory_md_reports_error_but_exits_0(run_hook, tmp_path):
    empty_dir = tmp_path / "empty_memory"
    empty_dir.mkdir()
    proc = run_hook(
        "memory_decay.py",
        "--project-root", tmp_path,
        "--memory-dir", empty_dir,
        "--print",
    )
    assert proc.returncode == 0  # maintenance reporter, never a gate
    assert "ERROR: MEMORY.md not found" in proc.stdout
