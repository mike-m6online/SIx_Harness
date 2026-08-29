"""Candidate mining: cue-phrase extraction, watermark offsets, stored-
title quality (whole-word clip + full-sentence rationale), and the
precision deny-list (short titles / status echoes / MEMORY.md-
maintenance records) with per-category skip counts."""
import json
import tempfile
from pathlib import Path

from claude_mem.extract_decisions import (
    ScanSkips,
    extract_decision_candidates,
    extract_dead_end_candidates,
    scan_candidates,
)
from claude_mem.textutil import ELLIPSIS


def _write_jsonl(tmp: str) -> Path:
    rows = [
        {"message": {"role": "user", "content":
            "We decided to keep the MLM head wired for the F8 grounding test."}},
        {"message": {"role": "assistant", "content": [
            {"type": "text", "text": "The FM-head rewire is rejected; it hits the "
                                     "predict-not-cause wall."},
        ]}},
        {"message": {"role": "assistant", "content": "Plain chatter with no cue."}},
        {"message": {"role": "user", "content":
            "Let's go with the GW causal-write path for the dialogue substrate."}},
    ]
    p = Path(tmp) / "session.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return p


def test_decision_candidates_found():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_jsonl(tmp)
        cands = extract_decision_candidates(p)
        titles = [c.title.lower() for c in cands]
        assert any("keep the mlm" in t for t in titles)
        assert any("go with the gw" in t for t in titles)
        assert all(c.state == "pending" for c in cands)


def test_dead_end_candidates_found():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_jsonl(tmp)
        cands = extract_dead_end_candidates(p)
        assert any("rejected" in c.approach.lower() for c in cands)
        assert all(c.state == "pending" for c in cands)


def test_no_cue_no_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "s.jsonl"
        p.write_text(json.dumps(
            {"message": {"role": "user", "content": "just a normal sentence."}}
        ), encoding="utf-8")
        assert extract_decision_candidates(p) == []
        assert extract_dead_end_candidates(p) == []


def test_scan_candidates_offset_reaches_eof():
    with tempfile.TemporaryDirectory() as tmp:
        p = _write_jsonl(tmp)
        _d, _e, off, _sk = scan_candidates(p)
        assert off == p.stat().st_size


def test_scan_candidates_incremental_from_offset():
    with tempfile.TemporaryDirectory() as tmp:
        # First record carries a decision cue with a known byte length.
        first = json.dumps(
            {"message": {"role": "user",
                         "content": "We decided to keep the MLM head wired "
                                    "for the F8 grounding test."}}
        ) + "\n"
        later = json.dumps(
            {"message": {"role": "user",
                         "content": "Let's go with the GW causal-write path "
                                    "for the dialogue substrate."}}
        ) + "\n"
        p = Path(tmp) / "session.jsonl"
        p.write_text(first + later, encoding="utf-8")

        start = len(first.encode("utf-8"))
        decisions, _e, off, _sk = scan_candidates(p, start_offset=start)
        titles = [d.title.lower() for d in decisions]
        assert not any("keep the mlm" in t for t in titles)
        assert any("go with the gw" in t for t in titles)
        assert off == p.stat().st_size


def test_scan_candidates_date_from_timestamp():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        p.write_text(json.dumps({
            "timestamp": "2026-05-01T12:00:00.000Z",
            "message": {"role": "user", "content":
                "We decided to keep the MLM head wired for the F8 test."},
        }), encoding="utf-8")
        decisions, _e, _off, _sk = scan_candidates(p)
        assert len(decisions) >= 1
        assert decisions[0].date == "2026-05-01"


# ---------------------------------------------------------------------------
# A2: stored-title quality -- whole-word clip at 200, full first sentence
# (whitespace-collapsed, capped 600) in rationale instead of a title copy
# ---------------------------------------------------------------------------

def _one_record_jsonl(tmp: str, content: str) -> Path:
    p = Path(tmp) / "session.jsonl"
    p.write_text(json.dumps(
        {"message": {"role": "user", "content": content}}
    ) + "\n", encoding="utf-8")
    return p


def test_long_title_clipped_word_boundary_with_ellipsis():
    long_sentence = (
        "We decided to route the epistemic-metabolism dial through the "
        "flag-gated use_epistemic_metabolism path so the dormancy-with-"
        "reception construction keeps its four discharge proofs intact "
        "while the wake-tick ledger map and the aspiration-satiation half "
        "ride the launch package unchanged for the matched control run."
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = _one_record_jsonl(tmp, long_sentence)
        decisions, _e, _off, _sk = scan_candidates(p)
        assert len(decisions) == 1
        title = decisions[0].title
        assert len(title) <= 200
        assert title.endswith(ELLIPSIS)
        # Word-boundary: the char before the ellipsis ends a whole word
        # present in the source (no mid-word fragment).
        head = title[:-1]
        assert not head.endswith(" ")
        assert head.split()[-1] in long_sentence.split()


def test_rationale_holds_full_sentence_not_title_copy():
    long_sentence = (
        "We decided to route the epistemic-metabolism dial through the "
        "flag-gated use_epistemic_metabolism path so the dormancy-with-"
        "reception construction keeps its four discharge proofs intact "
        "while the wake-tick ledger map and the aspiration-satiation half "
        "ride the launch package unchanged for the matched control run."
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = _one_record_jsonl(tmp, long_sentence)
        decisions, _e, _off, _sk = scan_candidates(p)
        assert len(decisions) == 1
        d = decisions[0]
        # Full sentence (< 600 chars) stored untruncated in rationale.
        assert d.rationale == long_sentence
        assert len(d.rationale) > len(d.title)


def test_rationale_capped_at_600_word_boundary():
    filler = " ".join(f"clause{i} of the ruling chain" for i in range(40))
    huge = f"We decided that {filler} settles the launch ordering."
    assert len(huge) > 600
    with tempfile.TemporaryDirectory() as tmp:
        p = _one_record_jsonl(tmp, huge)
        decisions, _e, _off, _sk = scan_candidates(p)
        assert len(decisions) == 1
        assert len(decisions[0].rationale) <= 600
        assert decisions[0].rationale.endswith(ELLIPSIS)


def test_dead_end_why_shelved_holds_full_sentence():
    sentence = (
        "The variance-floor approach was rejected because Mike ruled an "
        "estimator is not a floor and the felt path needs a live-Welford "
        "normalizer instead of the frozen z-score construction."
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = _one_record_jsonl(tmp, sentence)
        _d, dead_ends, _off, _sk = scan_candidates(p)
        assert len(dead_ends) == 1
        assert dead_ends[0].why_shelved == sentence


def test_multiline_sentence_whitespace_collapsed():
    content = (
        "We decided   to keep\nthe MLM head wired\n\tfor the F8 "
        "grounding test across both engines."
    )
    with tempfile.TemporaryDirectory() as tmp:
        p = _one_record_jsonl(tmp, content)
        decisions, _e, _off, _sk = scan_candidates(p)
        assert len(decisions) == 1
        assert "\n" not in decisions[0].title
        assert "  " not in decisions[0].title


# ---------------------------------------------------------------------------
# A6: precision deny-list -- real junk from the 2026-08-19 triage vs
# genuine candidates; MEMORY.md-maintenance records; skip counting
# ---------------------------------------------------------------------------

# Six junk titles matching the triage's dominant reject shapes: bare
# status-echo fragments and sub-40-char fragments.
_JUNK_MESSAGES = [
    "Locked in.",                                     # triage row, verbatim
    "All four locked in.",                            # triage row, verbatim
    "Everything is locked in.",
    "All 5 levers locked in!",
    "Approved: ship it.",                             # <40 chars, bare fragment
    "We decided: go.",                                # <40 chars, bare fragment
]

# Six genuine candidates (all >= 40 chars, none a bare acknowledgment).
_GENUINE_MESSAGES = [
    "We decided to keep the MLM head wired for the F8 grounding test.",
    "Let's go with the GW causal-write path for the dialogue substrate.",
    "Decision: we will adopt the live-Welford normalizer for felt-path signals.",
    "We decided to route the epistemic-metabolism dial through a flag gate.",
    "Mike approved the matched-control launch at the 300k frozen horizon.",
    "We are going with reception-based pin assignment for the P4 config.",
]


def _jsonl_from_messages(tmp: str, messages) -> Path:
    p = Path(tmp) / "session.jsonl"
    p.write_text("\n".join(
        json.dumps({"message": {"role": "user", "content": m}})
        for m in messages
    ) + "\n", encoding="utf-8")
    return p


def test_junk_vs_genuine_split():
    """The six real junk shapes are dropped; the six genuine ones all
    survive -- the deny-list splits the triage sample exactly."""
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, _JUNK_MESSAGES + _GENUINE_MESSAGES)
        decisions, _de, _off, skips = scan_candidates(p)
        titles = [d.title for d in decisions]
        for junk in _JUNK_MESSAGES:
            assert not any(junk.rstrip(".!") in t for t in titles), (junk, titles)
        for genuine in _GENUINE_MESSAGES:
            assert any(genuine[:30] in t for t in titles), (genuine, titles)
        assert skips.total == len(_JUNK_MESSAGES)


def test_status_echo_shapes_counted_as_status_echo():
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, [
            "Locked in.", "All four locked in.", "Everything is locked in.",
        ])
        decisions, _de, _off, skips = scan_candidates(p)
        assert decisions == []
        assert skips.status_echo == 3
        assert skips.short_title == 0


def test_short_fragment_counted_as_short_title():
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, ["We decided: go."])
        decisions, _de, _off, skips = scan_candidates(p)
        assert decisions == []
        assert skips.short_title == 1
        assert skips.status_echo == 0


def test_memory_md_maintenance_records_skipped():
    """Records OPENING with a maintenance marker are skipped wholesale --
    their 'superseded'/'trimming' narration phrase-matches dead-end cues."""
    maintenance = [
        "MEMORY.md compaction pass: superseded entries moved to the archive.",
        "Compacting the ledger now; the superseded checkpoints get archived.",
        "Trimming superseded anchors from the LATEST block per discipline.",
    ]
    genuine = "The FM-head rewire was rejected because it hits the predict-not-cause wall."
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, maintenance + [genuine])
        _d, dead_ends, _off, skips = scan_candidates(p)
        approaches = [e.approach for e in dead_ends]
        assert not any("superseded entries" in a for a in approaches)
        assert not any("Compacting" in a for a in approaches)
        assert not any("Trimming" in a for a in approaches)
        assert any("FM-head rewire" in a for a in approaches)
        assert skips.maintenance_records == 3


def test_maintenance_marker_mid_text_not_skipped():
    """Prefix discipline: a genuine sentence MENTIONING MEMORY.md
    mid-text is not maintenance narration."""
    msg = ("We decided to move the resume anchor rules out of MEMORY.md "
           "and into the arc ledger for every future closure.")
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, [msg])
        decisions, _de, _off, skips = scan_candidates(p)
        assert len(decisions) == 1
        assert skips.maintenance_records == 0


def test_genuine_sentence_containing_locked_in_survives():
    """The status-echo regex is anchored: a real decision sentence that
    CONTAINS 'locked in' mid-sentence is not a bare echo."""
    msg = ("We locked in the reception-based pin assignment because the "
           "match-gated alternative starves the codebook of updates.")
    with tempfile.TemporaryDirectory() as tmp:
        p = _jsonl_from_messages(tmp, [msg])
        decisions, _de, _off, skips = scan_candidates(p)
        assert len(decisions) == 1
        assert skips.status_echo == 0


def test_scan_skips_addition_and_summary():
    a = ScanSkips(short_title=1, status_echo=2, maintenance_records=3)
    b = ScanSkips(short_title=4, status_echo=0, maintenance_records=1)
    c = a + b
    assert (c.short_title, c.status_echo, c.maintenance_records) == (5, 2, 4)
    assert c.total == 11
    s = c.summary()
    assert "11" in s and "short-title=5" in s and "status-echo=2" in s
    assert "memory-maintenance-records=4" in s
