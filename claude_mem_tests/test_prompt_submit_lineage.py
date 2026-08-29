from pathlib import Path

from claude_mem.schema import init_db
from claude_mem.capture import CaptureStore, Thread, Decision, DeadEnd
from claude_mem.hooks import prompt_submit


def _seed_thread(root: Path):
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_thread(Thread(name="Inner Dialogue Drives Action", id="idda",
                            summary="dialogue should drive behavior"))
    store.add_decision(Decision(title="FM-head rewire NOT pursued (R3)",
                                date="2026-06-04", state="confirmed",
                                thread_id="idda", mike_approved=True))
    store.add_dead_end(DeadEnd(approach="GW causal-write path never built",
                               date="2026-04-09", state="confirmed",
                               thread_id="idda"))
    store.close()


def test_decision_prompt_injects_thread_lineage_via_fallback(tmp_path):
    root = Path(tmp_path)
    _seed_thread(root)
    # No cached lineage and no Ollama -> structured fallback is injected.
    out = prompt_submit.run("should we double-down on the FM-head rewire?", root)
    assert "GW causal-write path never built" in out
    assert "FM-head rewire NOT pursued" in out


def test_non_decision_unrelated_prompt_no_lineage(tmp_path):
    root = Path(tmp_path)
    _seed_thread(root)
    out = prompt_submit.run("run the tests and show output", root)
    # No decision intent + no subsystem hits -> no lineage block.
    assert "GW causal-write path" not in out


def test_lineage_path_never_raises(tmp_path, monkeypatch):
    root = Path(tmp_path)
    _seed_thread(root)
    # Force the lineage helper to raise; run() must still return safely.
    monkeypatch.setattr(prompt_submit, "_thread_lineage_block",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    out = prompt_submit.run("should we double-down or pivot?", root)
    assert isinstance(out, str)  # did not raise


def test_summary_only_word_does_not_falsefire(tmp_path):
    # A decision prompt whose only overlap is a word present ONLY in the thread
    # summary (not in any decision title / dead-end approach) must NOT inject.
    root = Path(tmp_path)
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_thread(Thread(name="Quantization Plan", id="qz",
                            summary="hinges on the calibration drift question"))
    store.add_decision(Decision(title="use INT8 weights", date="2026-06-04",
                                state="confirmed", thread_id="qz"))
    store.close()
    # "calibration" appears ONLY in the summary, not the name/title.
    out = prompt_submit.run("should we revisit the calibration approach or not?", root)
    assert "Quantization Plan" not in out


def test_single_token_overlap_does_not_falsefire(tmp_path):
    # A decision prompt sharing exactly ONE specific token with a thread's
    # decision text must NOT inject (the gate requires >= 2 overlap).
    root = Path(tmp_path)
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_thread(Thread(name="Build Anti-Recurrence", id="ar", summary="x"))
    store.add_decision(Decision(title="build the capture layer", date="2026-06-04",
                                state="confirmed", thread_id="ar"))
    store.close()
    # Only "build" overlaps; a single common-word match must not fire.
    out = prompt_submit.run("should we build a new parser or buy one?", root)
    assert "Build Anti-Recurrence" not in out


# ---------------------------------------------------------------------------
# Fix #2 — null byte must not crash run()
# ---------------------------------------------------------------------------
def _seed_from_script(root: Path):
    """Seed using the canonical seed_decision_threads data (via the same
    _seed_thread helper above, which has FM-head and GW causal text)."""
    _seed_thread(root)


def test_null_byte_in_prompt_does_not_raise(tmp_path):
    """run() must return a str (never raise) when the prompt contains a null byte."""
    root = Path(tmp_path)
    _seed_from_script(root)
    # Use a build-intent framing so the prompt gets past the early
    # has_existing_subsystem_intent guard and reaches the FTS search path
    # (where the null byte previously caused sqlite3.OperationalError).
    result = prompt_submit.run(
        "what if we build the \x00 FM-head widget from scratch", root
    )
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Fix #3 — build-intent prompts that match a thread must surface lineage
# ---------------------------------------------------------------------------
def test_build_intent_with_thread_overlap_injects_lineage(tmp_path):
    """A 'what if we' build-intent prompt that names thread decision text
    (>= 2 significant-token overlap) must surface the DECISION LINEAGE block."""
    root = Path(tmp_path)
    _seed_thread(root)
    # "FM-head" and "rewire" both appear in the seeded decision titles.
    out = prompt_submit.run(
        "what if we revive the FM-head rewire for coupling dialogue to drives", root
    )
    assert "DECISION LINEAGE" in out


# ---------------------------------------------------------------------------
# Fix #4 — short project abbreviations (MLM, F8) must match thread text
# ---------------------------------------------------------------------------
def test_short_abbreviations_match_thread_text(tmp_path):
    """MLM and F8 appear in seeded decision titles; they must match despite
    being < 4 chars and therefore previously stripped by _significant_tokens."""
    root = Path(tmp_path)
    _seed_thread(root)
    # The seeded decision title: "Two-problem decomposition: keep MLM for F8, build GW"
    # We use the idda fixture which has exactly this text. Our local _seed_thread
    # fixture uses abbreviated titles so we add a richer one here.
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_decision(Decision(
        title="keep MLM for F8 label coupling test",
        date="2026-06-04", state="confirmed",
        thread_id="idda", mike_approved=True,
    ))
    store.close()
    out = prompt_submit.run("should we keep the MLM for F8", root)
    assert "DECISION LINEAGE" in out
