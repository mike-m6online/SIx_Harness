"""Curated session-start injection (spec R4).

Replaces the old static `sw>=50 ORDER BY` render (14/15 lines were
harness boilerplate) with a hard-capped, curated render:
  ## INVARIANTS           -- titles from memory files with `type: invariant`
  ## RECENT CORRECTIONS   -- 5 most recent genuine (non-harness) corrections
  ## RECENT DECISIONS     -- 5 most recently CONFIRMED decisions
  one memory-health line  -- placeholder until Task 7 installs the gate
plus a novelty guard: SHA256 of the rendered block is stored in
meta:last_sessionstart_hash; an identical consecutive render gets an
appended "(unchanged since last session -- possible frozen corpus)" flag.
"""
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from claude_mem.capture import CaptureStore, Decision
from claude_mem.config import ProjectConfig
from claude_mem.hooks import session_start
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db
from claude_mem.telemetry import record_hook_heartbeat


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _init(tmp: Path) -> Path:
    root = Path(tmp)
    db = root / ".claude-mem" / "index.db"
    init_db(db)
    return root


def test_render_hard_cap_25_lines():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        # Seed far more corrections than the 5-line cap allows.
        for i in range(20):
            ing.add(Chunk(
                content=f"user correction number {i}: do not rebuild X",
                source="claude_code", role="user",
                signal_weight=100, is_correction=True,
            ))
        ing.close()
        store = CaptureStore(db)
        for i in range(20):
            store.add_decision(Decision(
                title=f"decision {i}", state="confirmed",
            ))
        store.close()
        out = session_start.run(root)
        lines = [l for l in out.splitlines() if l.strip()]
        assert len(lines) <= 25


def test_render_excludes_harness_content():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        # A harness-originated chunk that was mis-graded as a correction
        # before the Task-1 filter existed (is_correction=1 pre-migration
        # artifact) -- session-start must not surface it as a genuine
        # correction line.
        ing.add(Chunk(
            content="Base directory for this skill: /some/path -- no shortcuts allowed",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.add(Chunk(
            content="Mike correction: do not rebuild the differential dispatcher",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        out = session_start.run(root)
        assert "Base directory for this skill" not in out
        assert "differential dispatcher" in out


def test_render_includes_invariants_none_recorded_when_absent(monkeypatch, tmp_path):
    """No memory files with type: invariant exist yet (Task 8 creates
    them) -- the INVARIANTS section renders the explicit fallback
    instead of silently omitting the section."""
    root = _init(tmp_path)
    db = root / ".claude-mem" / "index.db"
    monkeypatch.setattr(
        session_start, "_memory_dir_for_project", lambda _root: tmp_path / "no_such_memory_dir"
    )
    out = session_start.run(root)
    assert "INVARIANTS" in out
    assert "none recorded" in out.lower()


def test_render_includes_invariant_titles_when_present(monkeypatch, tmp_path):
    root = _init(tmp_path)
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "invariant_pe_only.md").write_text(
        "---\n"
        "name: invariant_pe_only\n"
        "description: prediction error is the sole learning signal\n"
        "type: invariant\n"
        "---\n\n"
        "# PE is the sole learning signal\n",
        encoding="utf-8",
    )
    (mem_dir / "checkpoint_unrelated.md").write_text(
        "---\nname: checkpoint_unrelated\ndescription: not an invariant\ntype: project\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        session_start, "_memory_dir_for_project", lambda _root: mem_dir
    )
    out = session_start.run(root)
    assert "INVARIANTS" in out
    assert "PE is the sole learning signal" in out or "invariant_pe_only" in out
    assert "checkpoint_unrelated" not in out


def test_render_includes_invariants_normalized_frontmatter(monkeypatch, tmp_path):
    """The memory-dir normalizer folds top-level `type:` into
    `metadata: {node_type, type, originSessionId}` and slugifies `name`.
    Real invariant files land in that shape (Task 8) -- the render must
    still detect them (type read from metadata) and title them by the
    readable body H1, not the slugified name."""
    root = _init(tmp_path)
    mem_dir = tmp_path / "memory"
    mem_dir.mkdir()
    (mem_dir / "invariant_experience_over_description.md").write_text(
        "---\n"
        "name: invariant-experience-over-description\n"
        "description: an LLM can never know what chocolate tastes like\n"
        "metadata:\n"
        "  node_type: memory\n"
        "  type: invariant\n"
        "  originSessionId: abc123\n"
        "---\n\n"
        "# Experience over description -- the throughline\n\n"
        "body text\n",
        encoding="utf-8",
    )
    (mem_dir / "project_unrelated.md").write_text(
        "---\nname: project-unrelated\nmetadata:\n  node_type: memory\n  type: project\n---\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        session_start, "_memory_dir_for_project", lambda _root: mem_dir
    )
    out = session_start.run(root)
    assert "INVARIANTS" in out
    # readable H1 title surfaces, NOT the slugified name
    assert "Experience over description -- the throughline" in out
    assert "invariant-experience-over-description" not in out
    # a metadata.type=project file is NOT an invariant
    assert "project-unrelated" not in out


def test_render_shows_5_most_recent_confirmed_decisions_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        store = CaptureStore(db)
        for i in range(7):
            store.add_decision(Decision(
                title=f"confirmed decision {i}",
                date=f"2026-06-{10+i:02d}",
                state="confirmed",
            ))
        store.add_decision(Decision(
            title="pending decision should not appear",
            date="2026-06-20", state="pending",
        ))
        store.close()
        out = session_start.run(root)
        assert "pending decision should not appear" not in out
        # Only the 5 MOST RECENT confirmed decisions (by date) surface.
        assert "confirmed decision 6" in out
        assert "confirmed decision 5" in out
        assert "confirmed decision 0" not in out


def test_render_shows_5_most_recent_genuine_corrections_only():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        for i in range(8):
            ing.add(Chunk(
                content=f"correction number {i}",
                source="claude_code", role="user",
                signal_weight=100, is_correction=True,
                date=f"2026-06-{10+i:02d}",
            ))
        ing.close()
        out = session_start.run(root)
        assert "correction number 7" in out
        assert "correction number 0" not in out


def test_render_health_line_reports_latest_gate_result():
    # Task 7 wired the real gate as its own SessionStart hook; the render
    # must surface the LATEST gate result from its heartbeat row, not the
    # Task-5 "gate not installed" placeholder (false once the gate exists).
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        tpath = ProjectConfig(project_root=root).telemetry_path
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="6/10 green")
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="7/10 green")
        out = session_start.run(root)
        assert "7/10 green" in out
        assert "gate not installed" not in out.lower()


def test_render_health_line_flags_errored_gate_run():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        tpath = ProjectConfig(project_root=root).telemetry_path
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=False, detail="KeyError('x')")
        out = session_start.run(root)
        assert "memory-health" in out.lower()
        assert "errored" in out.lower()


def test_render_health_line_when_no_gate_run_recorded():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        out = session_start.run(root)
        assert "memory-health" in out.lower()
        assert "no gate run recorded" in out.lower()


def _ingest_render_invisible_chunk(db: Path, tag: str) -> None:
    """Advance the corpus WITHOUT changing the curated render: a low-signal
    non-correction chunk never surfaces in INVARIANTS/CORRECTIONS/DECISIONS,
    but it does grow the chunks table (part of the corpus signature)."""
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content=f"render-invisible filler {tag}",
        source="claude_code", role="assistant",
        signal_weight=5, is_correction=False,
    ))
    ing.close()


def test_novelty_guard_fires_when_new_session_gets_stale_render():
    """The frozen-corpus symptom the flag exists for: the corpus advanced,
    a NEW session starts, and the curated render is still byte-identical."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction that will not change across sessions",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        first = session_start.run(root, session_id="sess-A")
        assert "unchanged since last session" not in first
        _ingest_render_invisible_chunk(db, "between sessions")
        second = session_start.run(root, session_id="sess-B")
        assert "unchanged since last session" in second


def test_novelty_guard_silent_within_same_session():
    """Compact boundaries re-render inside ONE session; ingestion happens at
    SessionEnd, so an identical render there is expected -- no flag (the
    2026-07-03 false positive)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction that will not change across sessions",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        session_start.run(root, session_id="sess-A")
        _ingest_render_invisible_chunk(db, "mid session")
        second = session_start.run(root, session_id="sess-A")
        assert "unchanged since last session" not in second


def test_novelty_guard_silent_when_corpus_idle():
    """A new session whose predecessor ingested nothing gets an unchanged
    render legitimately (nothing new to surface) -- no flag; a dead
    ingestion pipeline is the health gate's check 1, not this hint."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction that will not change across sessions",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        session_start.run(root, session_id="sess-A")
        second = session_start.run(root, session_id="sess-B")
        assert "unchanged since last session" not in second


def test_novelty_guard_silent_when_session_unknown():
    """Without session identity (manual runs, pre-v2 wiring) cross-session
    repetition cannot be asserted -- never flag on adjacency alone."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction that will not change across sessions",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        session_start.run(root)
        _ingest_render_invisible_chunk(db, "between unknown sessions")
        second = session_start.run(root)
        assert "unchanged since last session" not in second


def test_novelty_guard_records_receiving_session_meta_pair():
    """The hash + receiving-session + corpus-sig meta triple is what the
    health gate's check 9 reads; it must be written coherently every run."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a correction so the render is non-empty",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        session_start.run(root, session_id="sess-A")
        store = CaptureStore(db)
        try:
            assert store.get_meta("last_sessionstart_hash")
            assert store.get_meta("last_sessionstart_session") == "sess-A"
            assert store.get_meta("last_sessionstart_corpus_sig")
        finally:
            store.close()


def test_novelty_guard_silent_when_content_changes():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="first correction",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        session_start.run(root, session_id="sess-A")
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a brand new second correction that changes the render",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
            date="2026-06-30",
        ))
        ing.close()
        second = session_start.run(root, session_id="sess-B")
        assert "unchanged since last session" not in second


def test_render_empty_when_no_index():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out = session_start.run(root)
        assert out == ""


def test_render_health_line_flags_stale_gate_run():
    # The cross-watch that closes the watchdog loop: the gate monitors
    # every hook, and the render (an independent hook process) flags a
    # gate that has itself stopped running -- e.g. its SessionStart
    # entry was removed. Without this, a dead gate is only detectable
    # by a human noticing the MEMORY-HEALTH line's absence.
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        tpath = ProjectConfig(project_root=root).telemetry_path
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="9/10 green")
        stale = (datetime.now(timezone.utc)
                 - timedelta(days=3)).isoformat()
        conn = sqlite3.connect(tpath)
        conn.execute(
            "UPDATE hook_heartbeat SET timestamp = ? "
            "WHERE hook = 'memory_health'", (stale,))
        conn.commit()
        conn.close()
        out = session_start.run(root)
        assert "GATE HAS NOT RUN" in out
        assert "9/10 green" in out  # last known verdict still shown


# ---------------------------------------------------------------------------
# A1: correction lines are whole-word clipped at 240 with an ellipsis
# ---------------------------------------------------------------------------

def test_correction_head_whole_word_clip_240():
    from claude_mem.textutil import ELLIPSIS

    long_correction = (
        "Mike correction: the dormancy-with-reception construction must "
        "keep all four discharge proofs intact while the epistemic "
        "metabolism dial rides behind the use_epistemic_metabolism flag, "
        "and the wake-tick ledger map plus the aspiration-satiation half "
        "must ride the launch package unchanged for the matched control "
        "run at the 300k frozen horizon."
    )
    assert len(long_correction) > 240
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content=long_correction,
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        out = session_start.run(root)
        line = next(
            l for l in out.splitlines() if l.startswith("- Mike correction")
        )
        head = line[2:]  # strip the "- " bullet
        assert len(head) <= 240
        assert head.endswith(ELLIPSIS)
        # Whole-word clip: every token of the head is a whole source token.
        assert all(tok in long_correction.split() for tok in head[:-1].split())


def test_correction_within_budget_not_clipped():
    content = "Mike correction: do not rebuild the differential dispatcher."
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content=content, source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        out = session_start.run(root)
        assert f"- {content}" in out


# ---------------------------------------------------------------------------
# A3c: rotation-aware corrections selection (least-recently-shown first)
# ---------------------------------------------------------------------------

def test_corrections_pool_rotates_across_renders():
    """Pool of 8 > the 5-line budget: the second render surfaces the 3
    chunks the first left out (never-accessed sorts ahead of the freshly
    access-bumped 5), so the pool self-rotates instead of freezing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        for i in range(1, 9):
            ing.add(Chunk(
                content=f"rotating correction number {i}",
                source="claude_code", role="user",
                signal_weight=100, is_correction=True,
            ))
        ing.close()
        first = session_start.run(root, session_id="sess-A")
        second = session_start.run(root, session_id="sess-A")
        first_shown = {
            i for i in range(1, 9)
            if f"rotating correction number {i}" in first
        }
        second_shown = {
            i for i in range(1, 9)
            if f"rotating correction number {i}" in second
        }
        assert len(first_shown) == 5
        assert len(second_shown) == 5
        left_out = set(range(1, 9)) - first_shown
        # Every chunk the first render left out leads the second render.
        assert left_out <= second_shown
        assert first_shown != second_shown


def test_corrections_stable_when_pool_fits_budget():
    """A pool of <= 5 keeps rendering all of it -- rotation only changes
    WHICH subset shows when the pool exceeds the per-render budget."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        for i in range(3):
            ing.add(Chunk(
                content=f"small-pool correction {i}",
                source="claude_code", role="user",
                signal_weight=100, is_correction=True,
            ))
        ing.close()
        first = session_start.run(root, session_id="sess-A")
        second = session_start.run(root, session_id="sess-A")
        for i in range(3):
            assert f"small-pool correction {i}" in first
            assert f"small-pool correction {i}" in second


# ---------------------------------------------------------------------------
# A4: novelty digest hashes only stable content (health line excluded)
# ---------------------------------------------------------------------------

def test_digest_stable_when_only_health_line_changes():
    """Two renders whose ONLY difference is the memory-health line (new
    gate timestamp + detail) must store the same novelty digest -- the
    pre-fix digest hashed the whole render, was unique every run, and
    permanently disarmed the frozen-render check."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        tpath = ProjectConfig(project_root=root).telemetry_path
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction for the digest check",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="9/10 green")
        first = session_start.run(root, session_id="sess-A")
        store = CaptureStore(db)
        try:
            hash_a = store.get_meta("last_sessionstart_hash")
        finally:
            store.close()
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="10/10 green")
        second = session_start.run(root, session_id="sess-B")
        store = CaptureStore(db)
        try:
            hash_b = store.get_meta("last_sessionstart_hash")
        finally:
            store.close()
        # The rendered blocks DIFFER (health line moved) ...
        assert "9/10 green" in first
        assert "10/10 green" in second
        # ... but the stable-content digest is identical.
        assert hash_a == hash_b
        # Corpus idle across the two sessions -> still no frozen flag.
        assert "unchanged since last session" not in second


def test_novelty_guard_fires_despite_changing_health_line():
    """The live-install scenario the fix exists for: the gate stamps a
    fresh timestamp into the health line every session, the corpus moved,
    the curated content did NOT -- the flag must fire."""
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        tpath = ProjectConfig(project_root=root).telemetry_path
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a stable correction that will not change across sessions",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="9/10 green")
        first = session_start.run(root, session_id="sess-A")
        assert "unchanged since last session" not in first
        _ingest_render_invisible_chunk(db, "between sessions")
        record_hook_heartbeat(
            tpath, hook="memory_health", ok=True, detail="9/10 green run 2")
        second = session_start.run(root, session_id="sess-B")
        assert "unchanged since last session" in second


# ---------------------------------------------------------------------------
# A5: capture-triage debt line
# ---------------------------------------------------------------------------

def test_debt_line_rendered_when_pending_rows_exist():
    from claude_mem.capture import DeadEnd

    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        store = CaptureStore(db)
        store.add_decision(Decision(
            title="pending decision alpha about the ledger map wiring",
            state="pending"))
        store.add_decision(Decision(
            title="pending decision beta about the satiation half",
            state="pending"))
        store.add_dead_end(DeadEnd(
            approach="pending dead-end about the variance floor",
            state="pending"))
        store.close()
        out = session_start.run(root)
        assert ("capture-triage debt: 2 decisions + 1 dead-ends pending "
                "-- run claude-mem capture-triage") in out


def test_debt_line_absent_when_nothing_pending():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        store = CaptureStore(db)
        store.add_decision(Decision(
            title="confirmed decision, not debt", state="confirmed"))
        store.close()
        out = session_start.run(root)
        assert "capture-triage debt" not in out


def test_debt_line_placed_after_recent_decisions_section():
    with tempfile.TemporaryDirectory() as tmp:
        root = _init(tmp)
        db = root / ".claude-mem" / "index.db"
        store = CaptureStore(db)
        store.add_decision(Decision(
            title="one pending decision for placement", state="pending"))
        store.close()
        out = session_start.run(root)
        lines = out.splitlines()
        decisions_idx = lines.index("## RECENT DECISIONS")
        debt_idx = next(
            i for i, l in enumerate(lines)
            if l.startswith("capture-triage debt:")
        )
        health_idx = next(
            i for i, l in enumerate(lines) if l.startswith("memory-health")
        )
        assert decisions_idx < debt_idx < health_idx
