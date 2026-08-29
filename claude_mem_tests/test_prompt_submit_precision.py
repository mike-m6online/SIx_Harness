"""Task B tests: prompt-submit precision + instrumentation.

Covers:
  - B1 calibration: the distinctive-token IDF lineage gate separates
    on-topic from off-topic prompts against a SYNTHETIC corpus (the live
    DB is never touched) -- including the two failure shapes the old
    substring-overlap gate could not block (two generic-token matches;
    an over-threshold sum with no distinctive token).
  - B2 damping: per-session injection caps via delivered_nudges
    (thread <= 2, DNR <= 3, stale <= 3); new session resets; session_id
    None disables damping.
  - B3 intent + per-item precision: removed bare verbs no longer fire;
    only items with >= 2 distinctive-token matches are listed; no
    passing item -> NOTHING emitted.
  - B4 stale scoping: reminder rides only on a fired DNR block and
    cites only filter-passing chunks.
  - B5 instrumentation: every invocation writes a wrapper_invocations
    row with correct flags / thread ids / matched-token evidence, and
    the CLI plumbs session_id end to end.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import List

import pytest
from click.testing import CliRunner

from claude_mem.capture import CaptureStore, DeadEnd, Decision, Thread
from claude_mem.cli import cli
from claude_mem.hooks import prompt_submit
from claude_mem.ingest import Chunk, Ingester
from claude_mem.intent import has_build_intent, has_existing_subsystem_intent
from claude_mem.relevance import (
    DISTINCTIVE_IDF, LINEAGE_IDF_SUM_THRESHOLD, META_DECISION_TOKENS,
    MIN_IDF_CORPUS,
    CorpusIdf, dnr_item_gate, lineage_gate, prompt_relevance_tokens,
    significant_tokens,
)
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> List[float]:
        return [0.1] * 1024


@pytest.fixture(autouse=True)
def _no_ollama(monkeypatch):
    """The hook must never hit a real embedding endpoint in tests."""
    monkeypatch.setattr(
        "claude_mem.hooks.prompt_submit.EmbeddingClient",
        lambda **kw: _ConstEmbedder(),
    )


# ---------------------------------------------------------------------------
# Synthetic corpus (IDF regime): >= MIN_IDF_CORPUS chunks with controlled
# document frequencies. NEVER the live DB.
# ---------------------------------------------------------------------------
GENERIC_WORDS = (
    "build test kit packaging installer database paper review week "
    "plan list settings hooks head"
)

ON_TOPIC_PROMPT = (
    "should we double-down on the FM-head rewire coupling the inner "
    "dialogue to drives"
)
OFF_TOPIC_KIT = (
    "let's build the kit packaging so the installer can vendor the module"
)
OFF_TOPIC_PAPER = (
    "should we double-down on the database paper review section or "
    "restructure the benchmarks table"
)
OFF_TOPIC_INSTALLER = (
    "we need to add an installer that writes the hooks into settings"
)
# Two generic-token matches ("build", "test") -- the shape the old
# substring overlap>=2 gate injected on.
OFF_TOPIC_GENERIC_2MATCH = "should we build the test suite or buy one"
# Three mid-idf matches whose sum clears the threshold but with no
# distinctive (idf >= 3.0) token among them.
OFF_TOPIC_MID_BAND = "should we keep the verifier kernel substrate or pivot"


def _build_corpus_project(root: Path) -> None:
    (root / ".claude-mem").mkdir(parents=True)
    db = root / ".claude-mem" / "index.db"
    init_db(db)
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    # 100 filler docs make the GENERIC_WORDS vocabulary cheap (high df).
    for i in range(100):
        ing.add(Chunk(
            content=f"filler note {i:03d}: {GENERIC_WORDS} routine chores",
            source="doc",
        ))
    # 9 mid-band docs put verifier/kernel/substrate idf between 2 and 3.
    for i in range(9):
        ing.add(Chunk(
            content=f"mid note {i}: verifier kernel substrate pipeline",
            source="doc",
        ))
    # 1 domain doc keeps inner/dialogue/rewire/coupling/drives distinctive
    # (df 2-3 including the vetted/narrative on-topic chunks below).
    ing.add(Chunk(
        content=(
            "domain note: the inner dialogue rewire coupling "
            "drives the gw path"
        ),
        source="doc",
    ))
    # Operator-vetted chunks the DNR search can surface:
    ing.add(Chunk(
        content=(
            "decision: the inner dialogue rewire coupling ships through "
            "the gw path -- do not add parallel infrastructure"
        ),
        source="memory", module="dialogue_gw",
        is_decision=True, signal_weight=80,
    ))
    ing.add(Chunk(
        content="decision: the kit packaging installer uses the build settings",
        source="doc", module="kit_packaging",
        is_decision=True, signal_weight=80,
    ))
    # Narrative symbol-bearing chunks for the stale-claim scoping test:
    # one on-topic (passes the B3 filter), one off-topic (must be dropped
    # even when retrieved).
    ing.add(Chunk(
        content=(
            "checkpoint: use_dialogue_gw flag wired in "
            "src/cuda_engine/modules/dialogue.py; the inner dialogue "
            "rewire coupling verified"
        ),
        source="memory", module="dialogue_gw",
        is_decision=True, signal_weight=80,
    ))
    ing.add(Chunk(
        content=(
            "checkpoint: use_installer_hooks flag added in "
            "scripts/install.py for the kit packaging settings"
        ),
        source="memory", module="kit_packaging",
        is_decision=True, signal_weight=80,
    ))
    ing.close()
    store = CaptureStore(db)
    store.add_thread(Thread(name="Inner Dialogue Drives Action", id="idda",
                            summary="dialogue should drive behavior"))
    store.add_decision(Decision(title="FM-head rewire NOT pursued (R3)",
                                date="2026-06-04", state="confirmed",
                                thread_id="idda", mike_approved=True))
    store.add_decision(Decision(title="build the test harness for action",
                                date="2026-06-05", state="confirmed",
                                thread_id="idda"))
    store.add_decision(Decision(title="verifier kernel substrate review",
                                date="2026-06-06", state="confirmed",
                                thread_id="idda"))
    store.add_decision(Decision(
        title="keep the inner dialogue rewire coupling; drives stay",
        date="2026-06-07", state="confirmed", thread_id="idda"))
    store.add_dead_end(DeadEnd(approach="GW causal-write path never built",
                               date="2026-04-09", state="confirmed",
                               thread_id="idda"))
    store.close()


@pytest.fixture(scope="module")
def corpus_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("corpus_proj")
    _build_corpus_project(root)
    return root


def _idf(root: Path) -> CorpusIdf:
    return CorpusIdf(root / ".claude-mem" / "index.db")


def _haystack(root: Path) -> str:
    db = root / ".claude-mem" / "index.db"
    store = CaptureStore(db)
    try:
        thread = store.get_thread("idda")
        return prompt_submit._thread_haystack(store, thread)
    finally:
        store.close()


# ---------------------------------------------------------------------------
# B1 calibration: gate separation on the synthetic corpus
# ---------------------------------------------------------------------------
def test_corpus_is_in_idf_regime(corpus_root):
    with _idf(corpus_root) as idf:
        assert idf.corpus_size >= MIN_IDF_CORPUS
        assert idf.usable


def test_calibration_bands_hold_on_synthetic_corpus(corpus_root):
    """Self-check the fixture: generic vocab scores low, domain vocab
    distinctive, mid-band vocab between -- the same band structure
    measured on the live corpus (generic 0.9-3.0, distinctive 3.3-8.7)."""
    with _idf(corpus_root) as idf:
        for tok in ("build", "test", "packaging", "installer", "head"):
            assert idf.idf(tok) < 1.0, tok
        for tok in ("dialogue", "rewire", "coupling", "inner", "drives"):
            assert idf.idf(tok) >= DISTINCTIVE_IDF, tok
        for tok in ("verifier", "kernel", "substrate"):
            assert 2.0 < idf.idf(tok) < DISTINCTIVE_IDF, tok


def test_on_topic_prompt_passes_lineage_gate(corpus_root):
    with _idf(corpus_root) as idf:
        res = lineage_gate(
            prompt_relevance_tokens(ON_TOPIC_PROMPT), _haystack(corpus_root), idf,
        )
    assert res.passed
    assert res.idf_sum >= LINEAGE_IDF_SUM_THRESHOLD
    assert any(v >= DISTINCTIVE_IDF for v in res.matched.values())


@pytest.mark.parametrize("prompt", [
    OFF_TOPIC_KIT, OFF_TOPIC_PAPER, OFF_TOPIC_INSTALLER,
])
def test_off_topic_classes_fail_lineage_gate(corpus_root, prompt):
    with _idf(corpus_root) as idf:
        res = lineage_gate(
            prompt_relevance_tokens(prompt), _haystack(corpus_root), idf,
        )
    assert not res.passed
    # These classes match at most one haystack token on word boundaries.
    assert len(res.matched) < 2


def test_two_generic_matches_blocked_by_idf_sum(corpus_root):
    """The old substring gate injected on any two token matches; two
    GENERIC matches must now fail the IDF-sum requirement specifically."""
    with _idf(corpus_root) as idf:
        res = lineage_gate(
            prompt_relevance_tokens(OFF_TOPIC_GENERIC_2MATCH),
            _haystack(corpus_root), idf,
        )
    assert not res.passed
    assert len(res.matched) >= 2          # the old gate would have fired
    assert res.idf_sum < LINEAGE_IDF_SUM_THRESHOLD


def test_mid_band_sum_blocked_by_distinctive_requirement(corpus_root):
    """A matched set can clear the sum threshold on volume alone; the
    >=1-token-above-DISTINCTIVE_IDF requirement must still block it."""
    with _idf(corpus_root) as idf:
        res = lineage_gate(
            prompt_relevance_tokens(OFF_TOPIC_MID_BAND),
            _haystack(corpus_root), idf,
        )
        assert not res.passed
        assert res.idf_sum >= LINEAGE_IDF_SUM_THRESHOLD
        assert all(v < DISTINCTIVE_IDF for v in res.matched.values())


def test_word_boundary_not_substring(corpus_root):
    """'test' must not match 'latest' (the old substring gate did)."""
    with _idf(corpus_root) as idf:
        res = lineage_gate(["test", "build"], "the latest rebuild notes", idf)
    assert res.matched == {}


# ---------------------------------------------------------------------------
# B1+B3 end-to-end through the hook (synthetic corpus)
# ---------------------------------------------------------------------------
def test_on_topic_prompt_injects_lineage_and_dnr(corpus_root):
    out = prompt_submit.run(ON_TOPIC_PROMPT, corpus_root)
    assert "DECISION LINEAGE" in out
    assert "DO NOT REBUILD" in out
    assert "inner dialogue rewire coupling" in out


@pytest.mark.parametrize("prompt", [
    OFF_TOPIC_KIT, OFF_TOPIC_PAPER, OFF_TOPIC_INSTALLER,
    OFF_TOPIC_GENERIC_2MATCH, OFF_TOPIC_MID_BAND,
    "let's build a test suite for the packaging kit",
])
def test_off_topic_prompts_emit_nothing(corpus_root, prompt):
    """B3c: intent fires and the search returns SOMETHING (the corpus has
    vetted kit/packaging chunks), but no item passes the distinctive
    filter and no thread passes the lineage gate -> empty output."""
    assert prompt_submit.run(prompt, corpus_root) == ""


def test_dnr_lists_only_filter_passing_items(corpus_root):
    """The off-topic vetted chunk (kit packaging) must not be listed on
    an on-topic prompt even when retrieval surfaces it."""
    out = prompt_submit.run(ON_TOPIC_PROMPT, corpus_root)
    assert "DO NOT REBUILD" in out
    assert "kit packaging installer uses the build settings" not in out


# ---------------------------------------------------------------------------
# B4 stale-claim scoping
# ---------------------------------------------------------------------------
def test_stale_reminder_cites_only_filtered_chunks(corpus_root):
    out = prompt_submit.run(ON_TOPIC_PROMPT, corpus_root)
    assert "STALE-CLAIM VERIFICATION REMINDER" in out
    assert "use_dialogue_gw" in out          # on-topic narrative chunk
    assert "use_installer_hooks" not in out  # off-topic narrative chunk


def test_stale_reminder_absent_when_dnr_absent(corpus_root):
    """B4: no DNR block -> no stale reminder, even though the corpus is
    full of narrative symbol-bearing chunks."""
    out = prompt_submit.run(OFF_TOPIC_INSTALLER, corpus_root)
    assert "STALE-CLAIM" not in out


# ---------------------------------------------------------------------------
# B3a intent-verb regression
# ---------------------------------------------------------------------------
def test_shopping_list_prompt_has_no_intent():
    prompt = "Make a shopping list for the week and plan the meals"
    assert not has_build_intent(prompt)
    assert not has_existing_subsystem_intent(prompt)


def test_shopping_list_prompt_emits_no_blocks(corpus_root):
    out = prompt_submit.run(
        "Make a shopping list for the week and plan the meals", corpus_root,
    )
    assert out == ""


def test_removed_bare_verbs_no_longer_fire():
    assert not has_build_intent("plan the meals for the week")
    assert not has_build_intent("make it faster")
    assert not has_build_intent("let's grab lunch")
    assert not has_build_intent("we should talk tomorrow")


def test_kept_construction_verbs_still_fire():
    assert has_build_intent("build the gate")
    assert has_build_intent("implement the parser")
    assert has_build_intent("create a config")
    assert has_build_intent("design the schema")
    assert has_build_intent("add a knob")
    assert has_build_intent("write a migration")
    assert has_build_intent("wire the getter into the adapter")
    assert has_build_intent("refactor the loader")
    assert has_build_intent("what if we skip the cache")


# ---------------------------------------------------------------------------
# B2 damping (small fixture -- fallback regime; damping is gate-independent)
# ---------------------------------------------------------------------------
DAMP_PROMPT = "what if we revive the FM-head rewire for coupling dialogue to drives"


def _seed_small_project(root: Path) -> Path:
    (root / ".claude-mem").mkdir(parents=True)
    db = root / ".claude-mem" / "index.db"
    init_db(db)
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content=(
            "checkpoint: fm-head rewire coupling verified via "
            "use_dialogue_gw flag in src/cuda_engine/modules/dialogue.py"
        ),
        source="memory", module="dialogue_gw",
        is_decision=True, signal_weight=80,
    ))
    ing.close()
    store = CaptureStore(db)
    store.add_thread(Thread(name="Inner Dialogue Drives Action", id="idda",
                            summary="dialogue should drive behavior"))
    store.add_decision(Decision(title="FM-head rewire NOT pursued (R3)",
                                date="2026-06-04", state="confirmed",
                                thread_id="idda", mike_approved=True))
    store.add_dead_end(DeadEnd(approach="GW causal-write path never built",
                               date="2026-04-09", state="confirmed",
                               thread_id="idda"))
    store.close()
    return db


def _last_tele_row(root: Path) -> sqlite3.Row:
    conn = sqlite3.connect(root / ".claude-mem" / "telemetry.db")
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM wrapper_invocations ORDER BY id DESC LIMIT 1"
        ).fetchone()
    finally:
        conn.close()


def test_thread_injects_at_most_twice_per_session(tmp_path):
    _seed_small_project(tmp_path)
    out1 = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s1")
    out2 = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s1")
    out3 = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s1")
    assert "DECISION LINEAGE" in out1
    assert "DECISION LINEAGE" in out2
    assert "DECISION LINEAGE" not in out3
    row = _last_tele_row(tmp_path)
    assert "thread:idda" in json.loads(row["suppressed_by_damping"])
    # A NEW session injects again.
    out4 = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s2")
    assert "DECISION LINEAGE" in out4


def test_dnr_block_capped_at_three_per_session(tmp_path):
    _seed_small_project(tmp_path)
    outs = [
        prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s1")
        for _ in range(4)
    ]
    assert all("DO NOT REBUILD" in o for o in outs[:3])
    assert "DO NOT REBUILD" not in outs[3]
    row = _last_tele_row(tmp_path)
    assert "dnr" in json.loads(row["suppressed_by_damping"])
    # New session resets the cap.
    assert "DO NOT REBUILD" in prompt_submit.run(
        DAMP_PROMPT, tmp_path, session_id="s2",
    )


def test_stale_block_capped_independently(tmp_path):
    db = _seed_small_project(tmp_path)
    store = CaptureStore(db)
    try:
        for _ in range(3):
            store.record_delivery("s1", "stale", "prompt_submit")
    finally:
        store.close()
    out = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="s1")
    # DNR still fires (its own counter is fresh) but the stale appendix
    # is withheld at its cap.
    assert "DO NOT REBUILD" in out
    assert "STALE-CLAIM" not in out
    row = _last_tele_row(tmp_path)
    assert "stale" in json.loads(row["suppressed_by_damping"])


def test_no_session_id_disables_damping(tmp_path):
    _seed_small_project(tmp_path)
    for _ in range(5):
        out = prompt_submit.run(DAMP_PROMPT, tmp_path)
        assert "DECISION LINEAGE" in out
        assert "DO NOT REBUILD" in out


# ---------------------------------------------------------------------------
# B2 capture-store counter mechanics (incl. live-DB additive migration)
# ---------------------------------------------------------------------------
def test_record_delivery_counts(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    store = CaptureStore(db)
    try:
        assert store.delivery_count("s1", "thread:idda") == 0
        store.record_delivery("s1", "thread:idda", "prompt_submit")
        store.record_delivery("s1", "thread:idda", "prompt_submit")
        assert store.delivery_count("s1", "thread:idda") == 2
        assert store.delivery_count("s2", "thread:idda") == 0
        # Presence-semantics API is unaffected and interoperable.
        assert store.was_delivered("s1", "thread:idda") is True
    finally:
        store.close()


def test_delivery_count_migrates_pre_column_rows(tmp_path):
    """A delivered_nudges table created before the delivery_count column
    (the live-DB shape) must migrate additively; its existing rows read
    as count=1."""
    db = tmp_path / "index.db"
    init_db(db)
    # delivered_nudges is created by CaptureStore, not init_db -- create
    # the PRE-migration shape directly (no delivery_count column).
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE delivered_nudges ("
        " session_id TEXT, chunk_id TEXT, tool_name TEXT, delivered_at TEXT,"
        " PRIMARY KEY (session_id, chunk_id))"
    )
    conn.execute(
        "INSERT INTO delivered_nudges (session_id, chunk_id, tool_name,"
        " delivered_at) VALUES ('s1', 'dnr', 'prompt_submit', 'x')"
    )
    conn.commit()
    conn.close()
    store = CaptureStore(db)
    try:
        assert store.delivery_count("s1", "dnr") == 1
        store.record_delivery("s1", "dnr", "prompt_submit")
        assert store.delivery_count("s1", "dnr") == 2
    finally:
        store.close()


# ---------------------------------------------------------------------------
# B5 instrumentation
# ---------------------------------------------------------------------------
def test_instrumentation_row_on_firing_prompt(tmp_path):
    _seed_small_project(tmp_path)
    out = prompt_submit.run(DAMP_PROMPT, tmp_path, session_id="sess-x")
    assert out  # blocks emitted
    row = _last_tele_row(tmp_path)
    assert row["session_id"] == "sess-x"
    assert row["build_intent_fired"] == 1
    assert row["investigation_intent_fired"] == 0
    assert row["do_not_rebuild_warning_emitted"] == 1
    assert row["stale_claim_warning_emitted"] == 1
    assert row["lineage_block_emitted"] == 1
    assert json.loads(row["lineage_thread_ids"]) == ["idda"]
    summary = json.loads(row["matched_token_summary"])
    assert "dnr" in summary and "lineage" in summary
    assert "idda" in summary["lineage"]
    assert row["prompt_hash"] == hashlib.sha256(
        DAMP_PROMPT.encode("utf-8")
    ).hexdigest()
    assert row["prompt_truncated"].startswith("what if we revive")
    assert row["retrieved_chunk_count"] >= 1


def test_instrumentation_row_on_non_intent_prompt(tmp_path):
    _seed_small_project(tmp_path)
    out = prompt_submit.run("status?", tmp_path, session_id="sess-y")
    assert out == ""
    row = _last_tele_row(tmp_path)
    assert row["session_id"] == "sess-y"
    assert row["build_intent_fired"] == 0
    assert row["decision_intent_fired"] == 0
    assert row["do_not_rebuild_warning_emitted"] == 0
    assert row["lineage_block_emitted"] == 0


def test_instrumentation_row_counts_every_invocation(tmp_path):
    _seed_small_project(tmp_path)
    prompt_submit.run("status?", tmp_path)
    prompt_submit.run(DAMP_PROMPT, tmp_path)
    conn = sqlite3.connect(tmp_path / ".claude-mem" / "telemetry.db")
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM wrapper_invocations"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n == 2


def test_wrapper_invocations_migrates_old_schema(tmp_path):
    """A telemetry.db created before the B5 columns must gain them
    additively on the next init (CREATE TABLE IF NOT EXISTS + ALTER)."""
    from claude_mem.telemetry import record_wrapper_invocation
    db = tmp_path / "telemetry.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE wrapper_invocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            prompt_hash TEXT,
            prompt_truncated TEXT,
            build_intent_fired INTEGER,
            investigation_intent_fired INTEGER,
            do_not_rebuild_warning_emitted INTEGER,
            stale_claim_warning_emitted INTEGER,
            retrieved_chunk_count INTEGER,
            retrieved_chunk_topics TEXT,
            retrieval_latency_ms INTEGER,
            session_id TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    record_wrapper_invocation(
        db, prompt_truncated="x", lineage_block_emitted=True,
        lineage_thread_ids=["idda"], suppressed_by_damping=["dnr"],
    )
    conn = sqlite3.connect(db)
    try:
        row = conn.execute(
            "SELECT lineage_block_emitted, lineage_thread_ids,"
            " suppressed_by_damping FROM wrapper_invocations"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == 1
    assert json.loads(row[1]) == ["idda"]
    assert json.loads(row[2]) == ["dnr"]


# ---------------------------------------------------------------------------
# CLI session_id plumb (CliRunner)
# ---------------------------------------------------------------------------
def test_cli_plumbs_session_id_from_stdin_json(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["init", "--project-root", str(tmp_path)])
    _seed_small_project_over_init(tmp_path)
    stdin_json = json.dumps({
        "session_id": "cli-sid",
        "cwd": str(tmp_path),
        "hook_event_name": "UserPromptSubmit",
        "prompt": DAMP_PROMPT,
    })
    result = runner.invoke(cli, ["prompt-submit", "--stdin"], input=stdin_json)
    assert result.exit_code == 0, result.output
    assert "DO NOT REBUILD" in result.output
    row = _last_tele_row(tmp_path)
    assert row["session_id"] == "cli-sid"


def test_cli_session_id_damps_across_invocations(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["init", "--project-root", str(tmp_path)])
    _seed_small_project_over_init(tmp_path)
    outputs = []
    for _ in range(3):
        stdin_json = json.dumps({
            "session_id": "cli-sid-2",
            "cwd": str(tmp_path),
            "hook_event_name": "UserPromptSubmit",
            "prompt": DAMP_PROMPT,
        })
        result = runner.invoke(
            cli, ["prompt-submit", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0, result.output
        outputs.append(result.output)
    assert "DECISION LINEAGE" in outputs[0]
    assert "DECISION LINEAGE" in outputs[1]
    assert "DECISION LINEAGE" not in outputs[2]


def test_cli_explicit_session_id_option(tmp_path):
    runner = CliRunner()
    runner.invoke(cli, ["init", "--project-root", str(tmp_path)])
    _seed_small_project_over_init(tmp_path)
    result = runner.invoke(cli, [
        "prompt-submit", "--project-root", str(tmp_path),
        "--session-id", "opt-sid", DAMP_PROMPT,
    ])
    assert result.exit_code == 0, result.output
    row = _last_tele_row(tmp_path)
    assert row["session_id"] == "opt-sid"


def _seed_small_project_over_init(root: Path) -> None:
    """Seed the small fixture into a project already created by
    `claude-mem init` (which made .claude-mem/ and index.db)."""
    db = root / ".claude-mem" / "index.db"
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content=(
            "checkpoint: fm-head rewire coupling verified via "
            "use_dialogue_gw flag in src/cuda_engine/modules/dialogue.py"
        ),
        source="memory", module="dialogue_gw",
        is_decision=True, signal_weight=80,
    ))
    ing.close()
    store = CaptureStore(db)
    store.add_thread(Thread(name="Inner Dialogue Drives Action", id="idda",
                            summary="dialogue should drive behavior"))
    store.add_decision(Decision(title="FM-head rewire NOT pursued (R3)",
                                date="2026-06-04", state="confirmed",
                                thread_id="idda", mike_approved=True))
    store.add_dead_end(DeadEnd(approach="GW causal-write path never built",
                               date="2026-04-09", state="confirmed",
                               thread_id="idda"))
    store.close()


# ---------------------------------------------------------------------------
# Small-corpus fallback regime: word-boundary overlap gate, no IDF
# ---------------------------------------------------------------------------
def test_small_corpus_falls_back_to_overlap_gate(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    with CorpusIdf(db) as idf:
        assert not idf.usable
        res = lineage_gate(
            ["head", "rewire"], "fm-head rewire not pursued", idf,
        )
        assert res.passed
        res1 = lineage_gate(["rewire"], "fm-head rewire not pursued", idf)
        assert not res1.passed


def test_small_corpus_dnr_gate_needs_two_matches(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    with CorpusIdf(db) as idf:
        assert dnr_item_gate(
            ["differential", "rivals"],
            "handles differential rivals", idf,
        ).passed
        assert not dnr_item_gate(
            ["differential", "dispatcher"],
            "handles differential rivals", idf,
        ).passed


def test_significant_tokens_splits_on_all_separators():
    toks = significant_tokens("the FM-head use_dialogue_gw src/scripts path")
    assert "head" in toks
    assert "dialogue" in toks
    assert "scripts" in toks
    # stopword + short-token filtering still applies; dedup preserves order
    assert "the" not in toks
    assert toks.count("head") == 1


def test_decision_meta_vocabulary_excluded_from_relevance_tokens():
    """Words from DECISION_INTENT_VERBS phrases mark THAT the user is
    deciding, not WHAT the topic is; they must not count as topical
    evidence (they leaked 2/200 vetted items on the live corpus via
    'double'+'down' before this exclusion)."""
    for tok in ("double", "down", "pivot", "revisit", "decision", "lineage"):
        assert tok in META_DECISION_TOKENS, tok
    toks = prompt_relevance_tokens(
        "should we double-down or pivot on the FM-head rewire decision"
    )
    assert "double" not in toks
    assert "down" not in toks
    assert "pivot" not in toks
    assert "decision" not in toks
    # Topic words survive.
    assert "head" in toks
    assert "rewire" in toks


def test_meta_tokens_do_not_match_items(corpus_root):
    """A chunk sharing only decision meta-vocabulary with the prompt must
    not pass the DNR item filter, whatever its idf."""
    with _idf(corpus_root) as idf:
        res = dnr_item_gate(
            prompt_relevance_tokens("should we double-down or pivot here"),
            "we chose to double down; do not pivot on this decision", idf,
        )
    assert not res.passed
    assert res.matched == {}
