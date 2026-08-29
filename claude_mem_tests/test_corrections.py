import json
import sqlite3
import tempfile
from pathlib import Path

from claude_mem.corrections import (
    CorrectionEvent, apply_corrections, extract_corrections,
    extract_topic,
)
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _write_session(path: Path, messages: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


def test_extract_corrections_finds_user_correction_after_assistant_msg():
    """Pattern: assistant proposes X; user replies with a correction
    phrase. extract_corrections returns one CorrectionEvent."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            {"type": "assistant",
             "message": {"role": "assistant",
                         "content": "let's build a new apollo loop"},
             "timestamp": "2026-05-24T12:00:00Z"},
            {"type": "user",
             "message": {"role": "user",
                         "content": "no, we already built the apollo loop"},
             "timestamp": "2026-05-24T12:00:30Z"},
        ])
        events = extract_corrections(p)
        assert len(events) == 1
        e = events[0]
        assert isinstance(e, CorrectionEvent)
        assert "we already built" in e.user_correction.lower()
        assert "apollo" in e.user_correction.lower()
        assert "apollo" in e.topic.lower()


def test_extract_corrections_ignores_user_messages_without_correction_phrase():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            {"type": "assistant",
             "message": {"role": "assistant", "content": "proposing X"},
             "timestamp": "2026-05-24T12:00:00Z"},
            {"type": "user",
             "message": {"role": "user", "content": "great, proceed"},
             "timestamp": "2026-05-24T12:00:30Z"},
        ])
        events = extract_corrections(p)
        assert events == []


def test_extract_corrections_ignores_correction_with_no_prior_assistant_msg():
    """A correction phrase as the very first message has no preceding
    assistant message to attribute to; should be skipped."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            {"type": "user",
             "message": {"role": "user",
                         "content": "we already built the X"},
             "timestamp": "2026-05-24T12:00:00Z"},
        ])
        events = extract_corrections(p)
        assert events == []


def test_extract_topic_pulls_capitalized_phrase_or_quoted_string():
    """Topic extraction grabs the most-distinctive nouns from the
    combined correction + prior-assistant text."""
    txt = "no we already built ddx_differential_intent_emit_kernel"
    topic = extract_topic(txt)
    assert "ddx_differential" in topic or "differential" in topic.lower()


def test_apply_corrections_indexes_correction_at_signal_weight_100():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        # Seed a baseline chunk on the topic
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(content="apollo loop discussion", source="claude_code",
                      role="assistant", signal_weight=20))
        ing.close()
        events = [CorrectionEvent(
            session_id="s1", tick=2, topic="apollo",
            user_correction="we already built the apollo loop",
            assistant_message="let's build apollo",
            timestamp="2026-05-24T12:00:30Z",
        )]
        n_applied = apply_corrections(
            events, db, embedder=_ConstEmbedder(),
        )
        assert n_applied == 1
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT signal_weight, is_correction FROM chunks "
                "WHERE source = 'correction_event'"
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == 100
        assert row[1] == 1


def test_apply_corrections_boosts_matching_existing_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="the apollo loop is the master switch for X",
            source="doc", module="apollo", signal_weight=20,
        ))
        ing.close()
        events = [CorrectionEvent(
            session_id="s1", tick=2, topic="apollo",
            user_correction="we already built apollo",
            assistant_message="let's build apollo",
            timestamp="2026-05-24T12:00:30Z",
        )]
        apply_corrections(events, db, embedder=_ConstEmbedder())
        conn = sqlite3.connect(db)
        try:
            sw = conn.execute(
                "SELECT signal_weight FROM chunks WHERE source = 'doc'"
            ).fetchone()[0]
        finally:
            conn.close()
        # Original sw=20; boosted by +20 to 40
        assert sw == 40


# -------- scan_corrections: streaming byte-offset watermark --------
#
# extract_corrections loads the ENTIRE session JSONL into memory; the live
# project transcript is multi-GB, so the SessionEnd hook needs the same
# byte-offset watermark treatment scan_candidates already has. These tests
# pin: parity with extract_corrections on the pair pattern, resume from
# offset, the ends-on-assistant rewind (pairing context survives the
# boundary), and the harness-content guard.

from claude_mem.corrections import scan_corrections
from claude_mem.hooks import session_end as se_mod


def _append_session(path: Path, messages: list[dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for m in messages:
            f.write(json.dumps(m) + "\n")


def _amsg(text: str, ts: str = "2026-07-02T12:00:00Z") -> dict:
    return {"type": "assistant",
            "message": {"role": "assistant", "content": text},
            "timestamp": ts}


def _umsg(text: str, ts: str = "2026-07-02T12:00:30Z") -> dict:
    return {"type": "user",
            "message": {"role": "user", "content": text},
            "timestamp": ts}


def test_scan_corrections_matches_extract_on_pair_pattern():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            _amsg("let's build a new apollo loop"),
            _umsg("no, we already built the apollo loop"),
        ])
        events, new_offset = scan_corrections(p)
        assert len(events) == 1
        assert "apollo" in events[0].topic.lower()
        assert new_offset == p.stat().st_size


def test_scan_corrections_resumes_from_offset():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            _amsg("proposing X"),
            _umsg("no, we already built the X thing"),
        ])
        events1, off1 = scan_corrections(p)
        assert len(events1) == 1
        _append_session(p, [
            _amsg("proposing the verifier_density_probe again"),
            _umsg("no, the verifier_density_probe already exists"),
        ])
        events2, off2 = scan_corrections(p, start_offset=off1)
        assert len(events2) == 1
        assert "verifier_density_probe" in events2[0].topic
        assert off2 == p.stat().st_size


def test_scan_corrections_rewinds_watermark_over_trailing_assistant():
    # Window ends on an assistant message: the returned offset must point
    # at that record's START, so a correction appended in the NEXT window
    # still has its pairing context. Re-reading an assistant record can
    # never duplicate an event (events fire only on user records).
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [_amsg("I will rebuild the gate_firing probe")])
        events1, off1 = scan_corrections(p)
        assert events1 == []
        assert off1 < p.stat().st_size  # rewound to the assistant start
        _append_session(p, [
            _umsg("no, gate_firing_diagnostic_probe already exists"),
        ])
        events2, off2 = scan_corrections(p, start_offset=off1)
        assert len(events2) == 1
        assert off2 == p.stat().st_size


def test_scan_corrections_skips_harness_pseudo_user_content():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        _write_session(p, [
            _amsg("proposing Y"),
            _umsg("<system-reminder>no, we already built the Y "
                  "subsystem</system-reminder>"),
        ])
        events, _ = scan_corrections(p)
        assert events == []


def test_session_end_run_anchors_forward_then_processes(monkeypatch):
    # First observation of a session file anchors the watermark at EOF
    # (multi-GB historical logs are never re-read); the next run applies
    # only newly-appended corrections.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        root.mkdir()
        db = root / ".claude-mem" / "index.db"
        init_db(db)
        home = Path(tmp) / "home"
        slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
        sess_dir = home / ".claude" / "projects" / slug
        sess_dir.mkdir(parents=True)
        jsonl = sess_dir / "sess-1.jsonl"
        _write_session(jsonl, [
            _amsg("historical proposal"),
            _umsg("no, we already built the historical_thing_module"),
        ])
        class _CloseableEmbedder(_ConstEmbedder):
            def close(self) -> None:
                pass

        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(
            se_mod, "EmbeddingClient",
            lambda **kw: _CloseableEmbedder(),
        )
        out1 = se_mod.run("sess-1", root)
        assert out1 == ""  # anchor-forward: nothing applied
        _append_session(jsonl, [
            _amsg("I propose building a fresh dwell_climb analyzer"),
            _umsg("no, the dwell_climb_analyzer already exists"),
        ])
        out2 = se_mod.run("sess-1", root)
        assert "1 correction" in out2 or "detected 1" in out2
        conn = sqlite3.connect(db)
        n = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE source='correction_event'"
        ).fetchone()[0]
        conn.close()
        assert n == 1
