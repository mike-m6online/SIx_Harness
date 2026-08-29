"""A3: the curated-correction rule (feedback_*.md / invariant_*.md
memory files carry is_correction=1 by NAME) at every surface:

  - bulk.is_curated_correction_file (the single rule definition)
  - incremental._scan_memory ingestion path (tags at write time)
  - cli `bulk` step-3 memory path (tags at write time)
  - maintenance.retag_corrections (repair pass for already-ingested rows)
  - cli `maintenance --retag-corrections` (operator surface + counts)
  - session-start render integration (a retagged chunk actually surfaces)

Root cause being closed: the phrase scanner has ~zero recall on genuine
operator corrections (replay over live transcripts: 0 hits), while the
operator's CURATED corrections live in these files -- ingested as chunks
but never flagged, so the RECENT CORRECTIONS pool froze at 9 chunks.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from click.testing import CliRunner

from claude_mem.bulk import is_curated_correction_file
from claude_mem.cli import cli
from claude_mem.incremental import run_incremental
from claude_mem.ingest import Chunk, Ingester
from claude_mem.maintenance import retag_corrections
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _make_project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    return root


def _chunk_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT file_path, is_correction, source FROM chunks"
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# the rule itself
# ---------------------------------------------------------------------------

def test_rule_matches_feedback_and_invariant_basenames():
    assert is_curated_correction_file(r"C:\mem\feedback_no_padding_ever.md")
    assert is_curated_correction_file("/home/u/memory/invariant_pe_only.md")
    assert is_curated_correction_file("feedback_x.md")


def test_rule_is_case_insensitive():
    assert is_curated_correction_file(r"C:\mem\Feedback_No_Padding.md")
    assert is_curated_correction_file(r"C:\mem\INVARIANT_PE_ONLY.MD")


def test_rule_rejects_other_memory_files():
    assert not is_curated_correction_file(r"C:\mem\project_pipeline_atlas.md")
    assert not is_curated_correction_file(r"C:\mem\checkpoint_2026_08_16.md")
    assert not is_curated_correction_file(r"C:\mem\reference_graphify_usage.md")


def test_rule_matches_basename_not_directory():
    # A feedback_-named DIRECTORY does not curse its contents...
    assert not is_curated_correction_file(r"C:\mem\feedback_dir\note.md")
    # ...and a curated basename inside any directory still matches.
    assert is_curated_correction_file(r"C:\other\dir\feedback_rule.md")


# ---------------------------------------------------------------------------
# ingestion paths tag at write time
# ---------------------------------------------------------------------------

def test_incremental_memory_scan_tags_curated_corrections(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    mem_dir = home / ".claude" / "projects" / proj_slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "feedback_no_padding_ever.md").write_text(
        "# No padding ever\n\nMike: never pad output with filler.\n",
        encoding="utf-8",
    )
    (mem_dir / "invariant_pe_only.md").write_text(
        "# PE only\n\nPrediction error is the sole learning signal.\n",
        encoding="utf-8",
    )
    (mem_dir / "project_note.md").write_text(
        "# Project note\n\nAn ordinary durable fact.\n", encoding="utf-8",
    )
    monkeypatch.setattr("claude_mem.incremental.Path.home", lambda: home)

    run_incremental(root, embedder=_ConstEmbedder(), budget_s=55.0)

    rows = _chunk_rows(root / ".claude-mem" / "index.db")
    by_name = {Path(r["file_path"]).name: r["is_correction"] for r in rows
               if r["source"] == "memory"}
    assert by_name["feedback_no_padding_ever.md"] == 1
    assert by_name["invariant_pe_only.md"] == 1
    assert by_name["project_note.md"] == 0


def test_bulk_cli_memory_step_tags_curated_corrections(tmp_path, monkeypatch):
    root = _make_project(tmp_path)
    home = tmp_path / "home"
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    mem_dir = home / ".claude" / "projects" / proj_slug / "memory"
    mem_dir.mkdir(parents=True)
    (mem_dir / "feedback_dialogue_is_the_window.md").write_text(
        "# Dialogue is the window\n\nMike: dialogue is the window.\n",
        encoding="utf-8",
    )
    (mem_dir / "checkpoint_note.md").write_text(
        "# Checkpoint\n\nOrdinary checkpoint content.\n", encoding="utf-8",
    )
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))

    runner = CliRunner()
    res = runner.invoke(
        cli, ["bulk", "--project-root", str(root), "--no-include-git"],
    )
    assert res.exit_code == 0, res.output

    rows = _chunk_rows(root / ".claude-mem" / "index.db")
    by_name = {Path(r["file_path"]).name: r["is_correction"] for r in rows
               if r["source"] == "memory"}
    assert by_name["feedback_dialogue_is_the_window.md"] == 1
    assert by_name["checkpoint_note.md"] == 0


# ---------------------------------------------------------------------------
# retag pass for already-ingested chunks
# ---------------------------------------------------------------------------

def _seed_untagged_memory_chunks(db: Path) -> None:
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content="Mike: audit existing design before parallel mechanisms.",
        source="memory", file_path=r"C:\mem\feedback_audit_existing.md",
    ))
    ing.add(Chunk(
        content="Prediction error is the sole learning signal.",
        source="memory", file_path=r"C:\mem\invariant_pe_only.md",
    ))
    ing.add(Chunk(
        content="An ordinary project memory note.",
        source="memory", file_path=r"C:\mem\project_note.md",
    ))
    ing.add(Chunk(
        content="A doc chunk that happens to share the basename.",
        source="doc", file_path=r"C:\docs\feedback_lookalike.md",
    ))
    ing.close()


def test_retag_corrections_tags_matching_memory_chunks(tmp_path):
    root = _make_project(tmp_path)
    db = root / ".claude-mem" / "index.db"
    _seed_untagged_memory_chunks(db)

    counts = retag_corrections(db)

    assert counts["scanned"] == 3          # memory chunks only
    assert counts["matched"] == 2
    assert counts["retagged"] == 2
    assert counts["already_tagged"] == 0
    rows = _chunk_rows(db)
    flags = {Path(r["file_path"]).name: r["is_correction"] for r in rows}
    assert flags["feedback_audit_existing.md"] == 1
    assert flags["invariant_pe_only.md"] == 1
    assert flags["project_note.md"] == 0
    # source='doc' rows are outside the memory ingestion rule's scope.
    assert flags["feedback_lookalike.md"] == 0


def test_retag_corrections_is_idempotent(tmp_path):
    root = _make_project(tmp_path)
    db = root / ".claude-mem" / "index.db"
    _seed_untagged_memory_chunks(db)

    first = retag_corrections(db)
    second = retag_corrections(db)

    assert first["retagged"] == 2
    assert second["retagged"] == 0
    assert second["already_tagged"] == 2
    assert second["matched"] == 2


def test_cli_maintenance_retag_corrections_reports_counts(tmp_path):
    root = _make_project(tmp_path)
    db = root / ".claude-mem" / "index.db"
    _seed_untagged_memory_chunks(db)

    runner = CliRunner()
    res = runner.invoke(cli, [
        "maintenance", "--project-root", str(root), "--retag-corrections",
    ])
    assert res.exit_code == 0, res.output
    assert "3 memory chunk(s) scanned" in res.output
    assert "2 matched" in res.output
    assert "2 newly tagged" in res.output
    assert "0 already tagged" in res.output


def test_cli_maintenance_without_pass_says_so(tmp_path):
    root = _make_project(tmp_path)
    runner = CliRunner()
    res = runner.invoke(cli, ["maintenance", "--project-root", str(root)])
    assert res.exit_code == 0, res.output
    assert "no pass selected" in res.output


def test_cli_maintenance_requires_index(tmp_path):
    runner = CliRunner()
    res = runner.invoke(cli, [
        "maintenance", "--project-root", str(tmp_path / "nothing"),
        "--retag-corrections",
    ])
    assert res.exit_code != 0


# ---------------------------------------------------------------------------
# end-to-end: a retagged curated correction surfaces at session start
# ---------------------------------------------------------------------------

def test_retagged_curated_correction_surfaces_in_render(tmp_path):
    from claude_mem.hooks import session_start

    root = _make_project(tmp_path)
    db = root / ".claude-mem" / "index.db"
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content="Mike: never trade architectural correctness for speed.",
        source="memory", file_path=r"C:\mem\feedback_time_is_weightless.md",
    ))
    ing.close()

    before = session_start.run(root)
    assert "never trade architectural correctness" not in before

    retag_corrections(db)

    after = session_start.run(root)
    assert "never trade architectural correctness" in after
