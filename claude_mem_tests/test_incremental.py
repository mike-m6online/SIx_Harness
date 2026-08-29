"""Tests for claude_mem.incremental (spec R3).

Incremental ingestion: docs/memory/ledgers are scanned by mtime against a
per-source watermark (`incr:docs` / `incr:memory` / `incr:ledgers` in the
`meta` table) -- only files newer than the watermark are (re-)chunked, and
re-chunking a changed file is a delete-and-replace of THAT file's chunks by
file_path (the one permitted delete in this codebase: a stale chunk of a
rewritten file is wrong data, not provenance). Sessions are scanned by
byte offset (`incr:sessions:<file>`), reusing the capture-pipeline
watermark pattern, with every session message passed through
`claude_mem.filters.is_harness_content` before ingestion. A hard
wall-clock budget stops the run cleanly mid-list; the watermark persists
only the progress actually made, so a second run resumes forward.

Every test builds its own throwaway fixture project dir (docs/, memory/,
session jsonls) + a fixture index.db via schema.init_db -- never touches
the live .claude-mem/index.db. A hand-rolled test-double embedder
substitutes for Ollama (no real HTTP calls in this file).
"""
from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from claude_mem.incremental import run_incremental
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _ConstEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        return [0.1] * 1024


def _chunks(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute("SELECT * FROM chunks").fetchall()
    finally:
        conn.close()


def _get_meta(db: Path, key: str) -> str | None:
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    return root


# ---------------------------------------------------------------------------
# docs/memory/ledgers: mtime-scan, new/changed/unchanged semantics
# ---------------------------------------------------------------------------

def test_new_doc_file_is_ingested(tmp_path):
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "# Title\n\nSome new doc content about apollo.\n", encoding="utf-8",
    )
    embedder = _ConstEmbedder()

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    rows = _chunks(root / ".claude-mem" / "index.db")
    assert len(rows) >= 1
    assert any("apollo" in r["content"] for r in rows)
    assert summary["docs"]["files_ingested"] == 1


def test_unchanged_doc_file_is_skipped_on_second_run(tmp_path):
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.md").write_text(
        "# Title\n\nStable content.\n", encoding="utf-8",
    )
    embedder = _ConstEmbedder()
    run_incremental(root, embedder=embedder, budget_s=55.0)
    n_calls_after_first = len(embedder.calls)

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["docs"]["files_ingested"] == 0
    # No new embed calls for the unchanged file.
    assert len(embedder.calls) == n_calls_after_first


def test_modified_doc_file_is_rechunked_without_duplicates(tmp_path):
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    doc_path = docs / "a.md"
    doc_path.write_text("# Title\n\nOriginal content v1.\n", encoding="utf-8")
    embedder = _ConstEmbedder()
    run_incremental(root, embedder=embedder, budget_s=55.0)
    db = root / ".claude-mem" / "index.db"
    first_rows = _chunks(db)
    assert len(first_rows) >= 1

    # mtime must strictly advance past the stored watermark -- bump it
    # explicitly since some filesystems have coarse mtime resolution.
    import os
    new_mtime = doc_path.stat().st_mtime + 5
    doc_path.write_text("# Title\n\nRewritten content v2, totally different.\n", encoding="utf-8")
    os.utime(doc_path, (new_mtime, new_mtime))

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["docs"]["files_ingested"] == 1
    rows = _chunks(db)
    # Old content gone, new content present, no leftover duplicate chunks
    # for this file_path.
    file_rows = [r for r in rows if r["file_path"] and Path(r["file_path"]) == doc_path]
    assert all("Rewritten content v2" in r["content"] for r in file_rows)
    assert not any("Original content v1" in r["content"] for r in rows)


def test_rechunk_delete_is_scoped_to_the_changed_file_only(tmp_path):
    """The delete-and-replace on re-chunk must touch ONLY the file being
    re-ingested -- an untouched sibling file's chunks must survive."""
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    stable = docs / "stable.md"
    changed = docs / "changed.md"
    stable.write_text("# Stable\n\nNever touched content.\n", encoding="utf-8")
    changed.write_text("# Changed\n\nOriginal.\n", encoding="utf-8")
    embedder = _ConstEmbedder()
    run_incremental(root, embedder=embedder, budget_s=55.0)

    import os
    new_mtime = changed.stat().st_mtime + 5
    changed.write_text("# Changed\n\nRewritten now.\n", encoding="utf-8")
    os.utime(changed, (new_mtime, new_mtime))
    run_incremental(root, embedder=embedder, budget_s=55.0)

    db = root / ".claude-mem" / "index.db"
    rows = _chunks(db)
    assert any("Never touched content" in r["content"] for r in rows)
    assert any("Rewritten now" in r["content"] for r in rows)
    assert not any("Original." in r["content"] for r in rows)


def test_memory_source_ingested_and_tagged(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    mem_dir = home / ".claude" / "projects" / proj_slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "note.md").write_text(
        "# Memory note\n\nSome durable fact worth remembering.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)
    embedder = _ConstEmbedder()

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["memory"]["files_ingested"] == 1
    rows = _chunks(root / ".claude-mem" / "index.db")
    assert any(r["source"] == "memory" for r in rows)


def test_ledger_glob_ingested(tmp_path):
    root = _make_project(tmp_path)
    ledger_dir = root / "sub" / ".superpowers" / "sdd"
    ledger_dir.mkdir(parents=True)
    (ledger_dir / "progress.md").write_text(
        "# Progress\n\nTask 4 underway.\n", encoding="utf-8",
    )
    embedder = _ConstEmbedder()

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["ledgers"]["files_ingested"] == 1
    rows = _chunks(root / ".claude-mem" / "index.db")
    assert any(r["source"] == "ledger" for r in rows)


# ---------------------------------------------------------------------------
# sessions: byte-offset watermark + harness filter
# ---------------------------------------------------------------------------

def _write_session_jsonl(path: Path, messages: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps({"message": m}) for m in messages) + "\n",
        encoding="utf-8",
    )


def test_session_new_file_first_observation_anchors_without_ingesting(tmp_path, monkeypatch):
    """Mirrors the capture-pipeline pattern: first sight of a session file
    anchors the watermark at EOF (historical content is covered by bulk
    backfill) and ingests nothing from it yet."""
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    proj_dir = home / ".claude" / "projects" / f"x-{proj_slug}-x"
    proj_dir.mkdir(parents=True)
    jsonl = proj_dir / "sess1.jsonl"
    _write_session_jsonl(jsonl, [
        {"role": "user", "content": "This is genuine user prose about apollo."},
    ])
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)
    embedder = _ConstEmbedder()

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["sessions"]["files_ingested"] == 0
    db = root / ".claude-mem" / "index.db"
    assert _get_meta(db, f"incr:sessions:{jsonl.name}") == str(jsonl.stat().st_size)
    rows = _chunks(db)
    assert len(rows) == 0


def test_session_bytes_past_watermark_are_ingested_on_second_run(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    proj_dir = home / ".claude" / "projects" / f"x-{proj_slug}-x"
    proj_dir.mkdir(parents=True)
    jsonl = proj_dir / "sess1.jsonl"
    _write_session_jsonl(jsonl, [
        {"role": "user", "content": "First message before any incremental run."},
    ])
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)
    embedder = _ConstEmbedder()
    run_incremental(root, embedder=embedder, budget_s=55.0)  # anchors watermark

    with open(jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"message": {
            "role": "user",
            "content": "We decided to go with the incremental ingestion design.",
        }}) + "\n")

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    assert summary["sessions"]["files_ingested"] == 1
    rows = _chunks(root / ".claude-mem" / "index.db")
    assert any("incremental ingestion design" in r["content"] for r in rows)


def test_session_harness_content_is_filtered(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    proj_dir = home / ".claude" / "projects" / f"x-{proj_slug}-x"
    proj_dir.mkdir(parents=True)
    jsonl = proj_dir / "sess1.jsonl"
    _write_session_jsonl(jsonl, [
        {"role": "user", "content": "anchor message so watermark starts past this"},
    ])
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)
    embedder = _ConstEmbedder()
    run_incremental(root, embedder=embedder, budget_s=55.0)

    with open(jsonl, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"message": {
            "role": "user",
            "content": "<system-reminder>harness-injected content that must "
                       "never be ingested as genuine user prose</system-reminder>",
        }}) + "\n")
        fh.write(json.dumps({"message": {
            "role": "user",
            "content": "Genuine follow-up decided we should filter harness text.",
        }}) + "\n")

    run_incremental(root, embedder=embedder, budget_s=55.0)

    rows = _chunks(root / ".claude-mem" / "index.db")
    assert not any("system-reminder" in r["content"] for r in rows)
    assert not any("harness-injected content" in r["content"] for r in rows)
    assert any("filter harness text" in r["content"] for r in rows)


# ---------------------------------------------------------------------------
# source ordering (Task-4 review Finding 2): memory must not be starved
# by a large docs corpus under the shared budget
# ---------------------------------------------------------------------------

def test_memory_is_scanned_before_docs_under_a_tight_budget(tmp_path, monkeypatch):
    """Root cause: docs/ can hold hundreds of files (375 on the live
    project) while memory/ holds a handful of curated notes. Scanning
    docs first under one shared wall-clock budget starves memory/
    ledgers/sessions of any budget at all -- confirmed on the live DB
    (`incr:docs` watermark advancing across 5 runs while `incr:memory`
    had never been written). The fix reorders the scan to
    memory -> ledgers -> sessions -> docs. This test pins that: with a
    budget only large enough for a couple of files, a project with one
    memory file and many doc files must ingest the memory file, not just
    doc files, in the first run."""
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    for i in range(20):
        (docs / f"doc{i}.md").write_text(
            f"# Doc {i}\n\nContent number {i} about a distinct topic.\n",
            encoding="utf-8",
        )
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    mem_dir = home / ".claude" / "projects" / proj_slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "note.md").write_text(
        "# Memory note\n\nSome durable fact worth remembering.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)
    embedder = _ConstEmbedder()

    # Budget expires after the memory file's scan has had a chance to run
    # but well before 20 doc files could all be processed: the _Budget
    # deadline is computed from the first now_fn() call, and each source
    # checks budget.expired() once per file before processing it.
    calls = {"n": 0}

    def _now():
        calls["n"] += 1
        # First call establishes the deadline (now_fn() + budget_s). Keep
        # reporting "no time has passed" for a few more calls (enough for
        # the single memory file plus a couple of docs) then jump forward
        # past the deadline for the rest of the run.
        if calls["n"] <= 3:
            return 0.0
        return 1000.0

    summary = run_incremental(root, embedder=embedder, budget_s=1.0, now_fn=_now)

    assert summary["memory"]["files_ingested"] == 1
    rows = _chunks(root / ".claude-mem" / "index.db")
    assert any(r["source"] == "memory" for r in rows)
    assert summary["docs"]["files_ingested"] < 20


def test_source_scan_order_is_memory_ledgers_sessions_docs(tmp_path, monkeypatch):
    """Directly pins the scan order (not just an outcome under a tight
    budget): instrument each _scan_* function and assert the call order
    is memory, ledgers, sessions, docs."""
    root = _make_project(tmp_path)
    embedder = _ConstEmbedder()

    import claude_mem.incremental as incr_mod

    order: list[str] = []
    originals = {
        "_scan_memory": incr_mod._scan_memory,
        "_scan_ledgers": incr_mod._scan_ledgers,
        "_scan_sessions": incr_mod._scan_sessions,
        "_scan_docs": incr_mod._scan_docs,
    }

    def _wrap(name, fn):
        def _inner(*args, **kwargs):
            order.append(name)
            return fn(*args, **kwargs)
        return _inner

    for name, fn in originals.items():
        monkeypatch.setattr(incr_mod, name, _wrap(name, fn))

    run_incremental(root, embedder=embedder, budget_s=55.0)

    assert order == ["_scan_memory", "_scan_ledgers", "_scan_sessions", "_scan_docs"]


def test_zero_budget_stops_early_and_second_run_resumes(tmp_path):
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    for i in range(5):
        (docs / f"doc{i}.md").write_text(
            f"# Doc {i}\n\nContent number {i} about a distinct topic.\n",
            encoding="utf-8",
        )
    embedder = _ConstEmbedder()

    # A budget that expires essentially immediately must stop after at most
    # a small amount of progress, not process every file in one shot.
    summary = run_incremental(root, embedder=embedder, budget_s=0.01)
    first_ingested = summary["docs"]["files_ingested"]
    assert first_ingested < 5

    # A second run with a normal budget picks up exactly where the first
    # left off -- total files ingested across both runs is 5, no dupes.
    summary2 = run_incremental(root, embedder=embedder, budget_s=55.0)
    assert first_ingested + summary2["docs"]["files_ingested"] == 5

    db = root / ".claude-mem" / "index.db"
    rows = _chunks(db)
    file_paths = {r["file_path"] for r in rows if r["file_path"]}
    assert len(file_paths) == 5


def test_same_mtime_siblings_are_not_permanently_orphaned(tmp_path):
    """Root-cause regression: several files sharing the EXACT SAME mtime
    (common on filesystems with coarse mtime resolution, or files written
    in one batch -- a git checkout, a sync, a bulk edit) must all
    eventually be ingested. A naive `mtime > watermark` scheme ingests
    the first such file, advances the watermark to that shared mtime, and
    then silently skips every sibling at the same mtime FOREVER -- this
    test pins identical mtimes deterministically via os.utime rather than
    relying on incidental filesystem timing."""
    import os

    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    paths = []
    for i in range(4):
        p = docs / f"doc{i}.md"
        p.write_text(f"# Doc {i}\n\nDistinct content {i}.\n", encoding="utf-8")
        paths.append(p)
    shared_mtime = 1_700_000_000.0
    for p in paths:
        os.utime(p, (shared_mtime, shared_mtime))
    embedder = _ConstEmbedder()

    # A budget generous enough for only ~1-2 files forces the run to stop
    # mid-list while multiple files remain at the SAME shared mtime.
    class _CountedNow:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self) -> float:
            self.n += 1
            # Deadline (start + budget_s) is computed from the first
            # call; every call after the 3rd reports time has passed,
            # forcing a stop after at most ~1 file ingested.
            return 0.0 if self.n <= 2 else 1000.0

    summary = run_incremental(
        root, embedder=embedder, budget_s=1.0, now_fn=_CountedNow(),
    )
    first_ingested = summary["docs"]["files_ingested"]
    assert 0 < first_ingested < 4

    # Subsequent runs (normal budget) must eventually ingest ALL siblings
    # at the shared mtime -- none permanently orphaned.
    total = first_ingested
    for _ in range(5):
        if total == 4:
            break
        s = run_incremental(root, embedder=embedder, budget_s=55.0)
        total += s["docs"]["files_ingested"]
    assert total == 4

    db = root / ".claude-mem" / "index.db"
    rows = _chunks(db)
    file_paths = {r["file_path"] for r in rows if r["file_path"]}
    assert len(file_paths) == 4


def test_budget_expiry_is_wall_clock_not_per_source_reset(tmp_path):
    """The budget is a single overall wall-clock deadline for the whole
    run (docs + memory + ledgers + sessions), not reset per source."""
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    for i in range(3):
        (docs / f"doc{i}.md").write_text(
            f"# Doc {i}\n\nContent {i}.\n", encoding="utf-8",
        )
    embedder = _ConstEmbedder()

    def _slow_now():
        # Every check reports the deadline as already passed after the
        # first call, forcing an immediate stop regardless of source.
        _slow_now.n += 1
        return 0.0 if _slow_now.n <= 1 else 1000.0
    _slow_now.n = 0

    summary = run_incremental(
        root, embedder=embedder, budget_s=1.0, now_fn=_slow_now,
    )
    total_ingested = (
        summary["docs"]["files_ingested"]
        + summary["memory"]["files_ingested"]
        + summary["ledgers"]["files_ingested"]
        + summary["sessions"]["files_ingested"]
    )
    assert total_ingested <= 1


# ---------------------------------------------------------------------------
# embedding failure visibility (Task-3 carry-item)
# ---------------------------------------------------------------------------

def test_embed_failure_logs_and_continues(tmp_path):
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# T\n\nContent that will fail embedding.\n", encoding="utf-8")
    (docs / "b.md").write_text("# T2\n\nContent that succeeds.\n", encoding="utf-8")

    class _FailFirst:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, text: str) -> list[float]:
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("simulated embed timeout")
            return [0.1] * 1024

    embedder = _FailFirst()
    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    # Ingestion of both files proceeds despite one embed failure -- chunks
    # rows exist, and the failure was logged, not swallowed / crashing.
    assert summary["docs"]["files_ingested"] == 2
    db = root / ".claude-mem" / "index.db"
    conn = sqlite3.connect(db)
    try:
        fail_rows = conn.execute(
            "SELECT COUNT(*) FROM ingestion_log WHERE action='embed_fail'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert fail_rows >= 1


def test_unreadable_file_logs_read_fail_and_continues(tmp_path, monkeypatch):
    """A file that raises OSError when parsed (deleted/rotated between the
    directory listing and the read, a permission error, a network-drive
    hiccup) must not crash the whole incremental-ingest run -- it is
    logged (action='read_fail') and the run continues to the next file.
    Root-cause fix: parse_fn(path) inside _run_mtime_source was
    previously unguarded."""
    root = _make_project(tmp_path)
    docs = root / "docs"
    docs.mkdir()
    (docs / "bad.md").write_text("# Bad\n\nWill fail to parse.\n", encoding="utf-8")
    (docs / "good.md").write_text("# Good\n\nParses fine.\n", encoding="utf-8")

    import claude_mem.incremental as incr_mod
    real_parse = incr_mod.parse_markdown_doc

    def _flaky_parse(path):
        if path.name == "bad.md":
            raise OSError("simulated transient read failure")
        return real_parse(path)

    monkeypatch.setattr(incr_mod, "parse_markdown_doc", _flaky_parse)
    embedder = _ConstEmbedder()

    summary = run_incremental(root, embedder=embedder, budget_s=55.0)

    # The good file is still ingested despite the bad one raising.
    assert summary["docs"]["files_ingested"] == 1
    db = root / ".claude-mem" / "index.db"
    rows = _chunks(db)
    assert any("Parses fine" in r["content"] for r in rows)
    assert not any("Will fail to parse" in r["content"] for r in rows)

    conn = sqlite3.connect(db)
    try:
        fail_rows = conn.execute(
            "SELECT COUNT(*) FROM ingestion_log WHERE action='read_fail'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert fail_rows == 1


def test_missing_index_db_raises_or_noops_gracefully(tmp_path):
    root = tmp_path / "no_index_proj"
    root.mkdir()
    embedder = _ConstEmbedder()
    # No .claude-mem/index.db exists -- must not crash SessionEnd; either
    # a clean no-op summary or a RuntimeError the caller can catch, never
    # an unhandled traceback via a different exception type.
    try:
        summary = run_incremental(root, embedder=embedder, budget_s=5.0)
    except RuntimeError:
        return
    assert summary == {} or all(
        v.get("files_ingested", 0) == 0 for v in summary.values()
    )


# ---------------------------------------------------------------------------
# hooks/session_end.run_incremental_ingest -- SessionEnd-safe wrapper
# ---------------------------------------------------------------------------

def test_run_incremental_ingest_never_raises_on_missing_index(tmp_path):
    from claude_mem.hooks import session_end

    root = tmp_path / "no_index"
    root.mkdir()
    # No .claude-mem/index.db -- must return "" cleanly, never raise.
    assert session_end.run_incremental_ingest(root) == ""


def test_run_incremental_ingest_reports_ingested_docs(tmp_path, monkeypatch):
    from claude_mem.hooks import session_end
    from claude_mem.schema import init_db

    root = tmp_path / "proj"
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# T\n\nContent about apollo.\n", encoding="utf-8")

    class _StubEmbedder:
        def embed(self, text):
            return [0.1] * 1024
        def close(self):
            pass

    monkeypatch.setattr(
        "claude_mem.hooks.session_end.EmbeddingClient",
        lambda **kw: _StubEmbedder(),
    )
    msg = session_end.run_incremental_ingest(root, budget_s=55.0)
    assert "incremental-ingest" in msg
    assert "docs=1" in msg


# ---------------------------------------------------------------------------
# capture-extract CLI wiring -- code-controlled ordering: candidates
# first, then incremental ingest, in one SessionEnd invocation.
# ---------------------------------------------------------------------------

def test_capture_extract_cli_runs_incremental_ingest_after_candidates(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from claude_mem.cli import cli
    from claude_mem.schema import init_db

    root = tmp_path / "proj"
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    docs = root / "docs"
    docs.mkdir()
    (docs / "a.md").write_text("# T\n\nContent about kmi.\n", encoding="utf-8")

    calls: list[str] = []

    def _fake_run_candidates(session_id, project_root):
        calls.append("candidates")
        return ""

    def _fake_run_incremental_ingest(project_root, **kwargs):
        calls.append("incremental")
        return "[claude-mem] session-end incremental-ingest: 1 file(s), 1 chunk(s) added."

    monkeypatch.setattr(
        "claude_mem.hooks.session_end.run_candidates", _fake_run_candidates,
    )
    monkeypatch.setattr(
        "claude_mem.hooks.session_end.run_incremental_ingest",
        _fake_run_incremental_ingest,
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["capture-extract", "--project-root", str(root)],
    )
    assert result.exit_code == 0, result.output
    assert calls == ["candidates", "incremental"]
    assert "incremental-ingest" in result.output


def test_capture_extract_cli_skip_incremental_flag(tmp_path, monkeypatch):
    from click.testing import CliRunner

    from claude_mem.cli import cli
    from claude_mem.schema import init_db

    root = tmp_path / "proj"
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")

    calls: list[str] = []
    monkeypatch.setattr(
        "claude_mem.hooks.session_end.run_candidates",
        lambda session_id, project_root: "",
    )
    monkeypatch.setattr(
        "claude_mem.hooks.session_end.run_incremental_ingest",
        lambda project_root, **kw: calls.append("incremental") or "",
    )

    runner = CliRunner()
    result = runner.invoke(
        cli, ["capture-extract", "--project-root", str(root), "--skip-incremental"],
    )
    assert result.exit_code == 0, result.output
    assert calls == []
