"""Tests for hooks/gen_decisions_state.py -- flag contract, --print render,
file-writing mode, and the silent no-op paths (missing DB / missing tables)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


def _make_capture_db(root: Path) -> Path:
    """A minimal claude-mem capture layer: one open thread with a confirmed
    Mike-approved decision (with rejected options), a pending decision, a
    superseded dead-end, and a pending dead-end. The confirmed decision
    title carries U+2192 to exercise the cp1252 stdout guard."""
    db = root / ".claude-mem" / "index.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, name TEXT, "
            "state TEXT, summary TEXT, last_updated TEXT)"
        )
        conn.execute(
            "CREATE TABLE decisions (id TEXT, thread_id TEXT, title TEXT, "
            "date TEXT, state TEXT, mike_approved INTEGER, "
            "options_rejected TEXT)"
        )
        conn.execute(
            "CREATE TABLE dead_ends (id TEXT, thread_id TEXT, approach TEXT, "
            "date TEXT, state TEXT, superseded_by TEXT)"
        )
        conn.execute(
            "INSERT INTO threads VALUES ('t1', 'homeostasis-arc', 'open', "
            "'Restore the force.', '2026-08-18')"
        )
        conn.execute(
            "INSERT INTO decisions VALUES ('d1', 't1', "
            "'Adopt reception → pin removal', '2026-08-10', 'confirmed', 1, "
            "'[\"option B\", \"option C\"]')"
        )
        conn.execute(
            "INSERT INTO decisions VALUES ('d2', 't1', "
            "'Unreviewed capture item', '2026-08-15', 'pending', 0, NULL)"
        )
        conn.execute(
            "INSERT INTO dead_ends VALUES ('e1', 't1', "
            "'grounding-primary line', '2026-08-05', 'confirmed', "
            "'ddx-return')"
        )
        conn.execute(
            "INSERT INTO dead_ends VALUES ('e2', 't1', "
            "'untriaged dead end', '2026-08-16', 'pending', NULL)"
        )
        conn.commit()
    finally:
        conn.close()
    return db


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "proj"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    _make_capture_db(root)
    return root, memory_dir


def test_no_flags_exits_2(run_hook):
    proc = run_hook("gen_decisions_state.py")
    assert proc.returncode == 2
    assert "required" in proc.stderr.lower()


def test_missing_memory_dir_exits_2(run_hook, tmp_path):
    proc = run_hook("gen_decisions_state.py", "--project-root", tmp_path)
    assert proc.returncode == 2
    assert "--memory-dir" in proc.stderr


def test_missing_project_root_exits_2(run_hook, tmp_path):
    proc = run_hook("gen_decisions_state.py", "--memory-dir", tmp_path)
    assert proc.returncode == 2
    assert "--project-root" in proc.stderr


def test_print_renders_capture_state(run_hook, tmp_path):
    root, memory_dir = _fixture(tmp_path)
    proc = run_hook(
        "gen_decisions_state.py",
        "--project-root", root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    out = proc.stdout
    assert out.startswith("=== GENERATED ")
    assert "DO NOT HAND-EDIT" in out
    assert ("threads: 1 (open: 1); decisions: 2 (pending: 1); "
            "dead-ends: 2 (pending: 1)") in out
    assert "## OPEN THREADS" in out
    assert "### homeostasis-arc [open] (updated 2026-08-18)" in out
    assert "Restore the force." in out
    # Confirmed Mike-approved decision with the U+2192 title survives the
    # UTF-8 stdout guard and carries no state suffix.
    assert ("- DECISION (Mike-approved) [2026-08-10] "
            "Adopt reception → pin removal" in out)
    assert "rejected: option B; option C" in out
    # Pending decision shows its state suffix inside the thread.
    assert "- DECISION [pending] [2026-08-15] Unreviewed capture item" in out
    assert ("- DEAD-END [2026-08-05] grounding-primary line "
            "(superseded_by: ddx-return)" in out)
    assert "## RECENT CONFIRMED DECISIONS" in out
    assert "## PENDING CAPTURE -- agent: review + formalize" in out
    assert "- decision [2026-08-15] Unreviewed capture item" in out
    assert "- dead-end [2026-08-16] untriaged dead end" in out
    # --print mode never writes the state file.
    assert not (root / "DECISIONS_STATE.md").exists()


def test_write_mode_creates_decisions_state_md(run_hook, tmp_path):
    root, memory_dir = _fixture(tmp_path)
    proc = run_hook(
        "gen_decisions_state.py",
        "--project-root", root,
        "--memory-dir", memory_dir,
    )
    assert proc.returncode == 0
    assert "wrote DECISIONS_STATE.md" in proc.stdout
    out_file = root / "DECISIONS_STATE.md"
    assert out_file.is_file()
    text = out_file.read_text(encoding="utf-8")
    assert "## OPEN THREADS" in text
    assert "Adopt reception → pin removal" in text


def test_missing_db_is_silent_noop(run_hook, tmp_path):
    root = tmp_path / "proj_nodb"
    root.mkdir()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    proc = run_hook(
        "gen_decisions_state.py",
        "--project-root", root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
    assert not (root / "DECISIONS_STATE.md").exists()


def test_missing_tables_is_silent_noop(run_hook, tmp_path):
    root = tmp_path / "proj_notables"
    db = root / ".claude-mem" / "index.db"
    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE meta (key TEXT, value TEXT)")
        conn.commit()
    finally:
        conn.close()
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()
    proc = run_hook(
        "gen_decisions_state.py",
        "--project-root", root,
        "--memory-dir", memory_dir,
        "--print",
    )
    assert proc.returncode == 0
    assert proc.stdout == ""
