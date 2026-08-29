"""Tests for claude_mem.migrate_regrade (spec R2a).

Corpus re-grade + module-tag migration: demote harness-content chunks
that were ingested BEFORE the Task-1 filter existed (sw=0, flags
cleared, module left untouched -- provenance is never deleted, only
re-graded in place), and retag NULL-module chunks whose content or
doc file_path identifies a substrate flag / module family.

Every test builds its own throwaway fixture DB via schema.init_db +
Ingester -- never touches the live .claude-mem/index.db.
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from claude_mem.ingest import Chunk, Ingester
from claude_mem.migrate_regrade import migrate_regrade
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _embedder():
    return _ConstEmbedder()


def _fixture_db(tmp: str) -> Path:
    db = Path(tmp) / "index.db"
    init_db(db)
    return db


def _fixture_project_root(tmp: str) -> Path:
    """Synthetic project root carrying the docs/marathon/module_states
    state files the retag roster is collected from (see
    bulk.collect_module_names). Hermetic: the retag tests must not
    depend on any real project checkout being present on the machine."""
    root = Path(tmp) / "proj"
    state_dir = root / "docs" / "marathon" / "module_states"
    state_dir.mkdir(parents=True)
    for flag in ("use_apollo", "use_l6_innovation"):
        (state_dir / f"{flag}.state.yaml").write_text(
            f"auto_derived:\n  config_flag: {flag}\n", encoding="utf-8",
        )
    return root


def _row(db: Path, content_substr: str) -> sqlite3.Row:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM chunks WHERE content LIKE ?",
            (f"%{content_substr}%",),
        ).fetchone()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 1: failing tests describing the desired migration behavior
# ---------------------------------------------------------------------------

def test_boilerplate_high_signal_chunk_is_demoted_module_untouched():
    """A pre-Task-1 chunk that phrase-graded as sw=100 (is_correction=1)
    but is actually harness content (skill-file body) must be demoted to
    sw=0 with is_correction/is_decision cleared. `module` is untouched by
    the demotion step -- retagging is a separate, additive concern."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content=(
                "Base directory for this skill: /skills/brainstorming\n"
                "we already built that. no shortcuts. the approach is locked."
            ),
            source="claude_code",
            role="user",
            module="use_apollo",
            signal_weight=100,
            is_correction=True,
            is_decision=True,
        ))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="test1")

        row = _row(db, "Base directory for this skill")
        assert row is not None
        assert row["signal_weight"] == 0
        assert row["is_correction"] == 0
        assert row["is_decision"] == 0
        assert row["module"] == "use_apollo"  # untouched by demotion
        assert summary["demoted"] == 1


def test_genuine_correction_keeps_sw_100():
    """A genuine user correction (no harness marker) must be left alone."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="No, we already built that -- check the existing apollo loop.",
            source="claude_code",
            role="user",
            signal_weight=100,
            is_correction=True,
        ))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="test2")

        row = _row(db, "check the existing apollo loop")
        assert row is not None
        assert row["signal_weight"] == 100
        assert row["is_correction"] == 1
        assert summary["demoted"] == 0


def test_chunk_naming_flag_gets_module_retagged():
    """A NULL-module chunk whose content names a substrate flag
    (use_l6_innovation, carried by the fixture project root's
    docs/marathon/module_states) is retagged via detect_module."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content=(
                "We enabled use_l6_innovation in a sibling config to test "
                "sustained divergence."
            ),
            source="doc",
            module=None,
        ))
        ing.close()

        project_root = _fixture_project_root(tmp)
        summary = migrate_regrade(
            db, backup_suffix="test3", project_root=project_root,
        )

        row = _row(db, "sustained divergence")
        assert row is not None
        assert row["module"] == "use_l6_innovation"
        assert summary["retagged_module"] == 1


def test_second_run_is_idempotent_zero_changes():
    """Running the migration twice must report 0 changes on the second
    pass -- the migration is UPDATE-only and re-running it must be safe."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="<system-reminder>be terse</system-reminder>",
            source="claude_code",
            role="user",
            signal_weight=100,
            is_correction=True,
        ))
        ing.add(Chunk(
            content="We enabled use_apollo for the hypothesis loop test.",
            source="doc",
            module=None,
        ))
        ing.close()

        project_root = _fixture_project_root(tmp)
        first = migrate_regrade(db, backup_suffix="test4a", project_root=project_root)
        assert first["demoted"] == 1
        assert first["retagged_module"] == 1

        second = migrate_regrade(db, backup_suffix="test4b", project_root=project_root)
        assert second["demoted"] == 0
        assert second["retagged_module"] == 0


def test_no_rows_deleted_row_count_stable():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="<task-notification>ping</task-notification>",
            source="claude_code",
            role="user",
            signal_weight=50,
            is_decision=True,
        ))
        ing.add(Chunk(
            content="genuine note about the architecture", source="memory",
        ))
        ing.close()

        conn = sqlite3.connect(db)
        before = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()

        migrate_regrade(db, backup_suffix="test5")

        conn = sqlite3.connect(db)
        after = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        conn.close()
        assert before == after == 2


def test_backup_file_created_alongside_db():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="a plain note", source="memory"))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="mybackup")

        backup_path = db.parent / f"{db.name}.bak-mybackup"
        assert backup_path.is_file()
        assert summary["backed_up_to"] == str(backup_path)


def test_refuses_to_run_if_backup_would_fail(monkeypatch):
    """If the backup copy raises, the migration must refuse to touch the
    DB at all -- no UPDATEs, no partial state."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="<system-reminder>be terse</system-reminder>",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()

        import claude_mem.migrate_regrade as mr

        def _boom(*a, **kw):
            raise OSError("disk full")

        monkeypatch.setattr(mr.shutil, "copy2", _boom)

        with pytest.raises(RuntimeError):
            migrate_regrade(db, backup_suffix="willfail")

        row = _row(db, "system-reminder")
        assert row["signal_weight"] == 100  # untouched -- refused before UPDATE


def test_per_source_demotion_counts_reported():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="<system-reminder>be terse</system-reminder>",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.add(Chunk(
            content="<system-reminder>be terse too</system-reminder>",
            source="claude_code", role="assistant",
            signal_weight=20, is_decision=True,
        ))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="test6")

        assert summary["demoted"] == 2
        assert summary["demoted_by_source"] == {"claude_code": 2}


def test_doc_path_fallback_retags_module_family():
    """A doc chunk with module=None whose file_path sits under
    docs/marathon/module_states/ names its flag in the path even when the
    content prose doesn't mention the flag token verbatim."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="Status: DORMANT. Never wired into the dispatch table.",
            source="doc",
            file_path="docs/marathon/module_states/use_apollo.state.yaml",
            module=None,
        ))
        ing.close()

        summary = migrate_regrade(
            db, backup_suffix="test7", project_root=_fixture_project_root(tmp),
        )

        row = _row(db, "Never wired into the dispatch table")
        assert row["module"] == "use_apollo"
        assert summary["retagged_module"] == 1


def test_unchanged_count_reported_for_untouched_rows():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="an ordinary already-correct chunk", source="memory"))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="test8")
        assert summary["unchanged"] == 1
        assert summary["demoted"] == 0
        assert summary["retagged_module"] == 0


# ---------------------------------------------------------------------------
# Pass 3: assistant-corrections demotion. A correction is the HUMAN
# correcting the assistant; the old content-only grader set
# is_correction=1 on assistant prose that merely echoed a correction
# phrase, so RECENT CORRECTIONS surfaced assistant essays instead of the
# operator's words. The pass clears the flag on non-user chunks ONLY --
# signal_weight is a separate axis and stays untouched.
# ---------------------------------------------------------------------------

def test_assistant_correction_flag_cleared_sw_untouched():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="you're right -- we already built the apollo loop, "
                    "my earlier framing missed it",
            source="claude_code", role="assistant",
            signal_weight=100, is_correction=True,
        ))
        ing.add(Chunk(
            content="no, we already built the apollo loop -- check the "
                    "existing implementation",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()

        summary = migrate_regrade(db, backup_suffix="ac1")

        a = _row(db, "my earlier framing missed it")
        assert a["is_correction"] == 0
        assert a["signal_weight"] == 100  # separate axis, untouched
        u = _row(db, "check the existing implementation")
        assert u["is_correction"] == 1  # the human's words keep the flag
        assert summary["demoted_assistant_corrections"] == 1


def test_correction_event_chunks_survive_assistant_pass():
    # correction_event chunks are role="user" by construction
    # (corrections.apply_corrections) -- the pass must never touch them.
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="no, the dwell_climb_analyzer already exists",
            source="correction_event", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        summary = migrate_regrade(db, backup_suffix="ac2")
        row = _row(db, "dwell_climb_analyzer")
        assert row["is_correction"] == 1
        assert summary["demoted_assistant_corrections"] == 0


def test_assistant_corrections_pass_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(
            content="you're right, we already built the gate mechanism",
            source="claude_code", role="assistant",
            signal_weight=80, is_correction=True,
        ))
        ing.close()
        first = migrate_regrade(db, backup_suffix="ac3a")
        assert first["demoted_assistant_corrections"] == 1
        second = migrate_regrade(db, backup_suffix="ac3b")
        assert second["demoted_assistant_corrections"] == 0


def test_backup_refuses_to_overwrite_existing_backup():
    # A re-used suffix silently destroyed the Task-2 pre-migration
    # snapshot (2026-07-02 live incident: the CLI's old fixed default
    # aimed every run at the same file). A backup must never clobber a
    # backup.
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db(tmp)
        first = migrate_regrade(db, backup_suffix="same")
        assert Path(first["backed_up_to"]).exists()
        with pytest.raises(RuntimeError, match="already exists"):
            migrate_regrade(db, backup_suffix="same")
