import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from claude_mem.report import build_weekly_summary
from claude_mem.telemetry import (
    init_telemetry_db, record_heartbeat, record_wrapper_invocation,
)


def _seed_week(db: Path):
    init_telemetry_db(db)
    for i in range(5):
        record_wrapper_invocation(
            db, prompt_truncated=f"prompt {i}",
            build_intent_fired=(i % 2 == 0),
            investigation_intent_fired=(i == 1),
            do_not_rebuild_warning_emitted=(i % 2 == 0),
            stale_claim_warning_emitted=(i == 0),
            retrieved_chunk_count=i * 2,
            retrieved_chunk_topics=[f"topic_{i}"],
            retrieval_latency_ms=50 + i * 10,
            session_id=f"s{i}",
        )
    record_heartbeat(db, component="ollama", status="alive")
    record_heartbeat(db, component="index", status="alive")


def test_build_weekly_summary_counts_invocations_and_warnings():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        _seed_week(db)
        summary = build_weekly_summary(db)
        assert "Wrapper invocations: 5" in summary
        assert "Build-intent fires: 3" in summary
        assert "Investigation-intent fires: 1" in summary
        assert "DO NOT REBUILD warnings emitted: 3" in summary
        assert "Stale-claim warnings emitted: 1" in summary


def test_build_weekly_summary_lists_top_topics():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        for _ in range(3):
            record_wrapper_invocation(
                db, retrieved_chunk_topics=["apollo", "ddx"],
            )
        for _ in range(2):
            record_wrapper_invocation(
                db, retrieved_chunk_topics=["kmi"],
            )
        summary = build_weekly_summary(db)
        assert "apollo" in summary
        assert "kmi" in summary


def test_build_weekly_summary_flags_dead_components():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        record_heartbeat(db, component="ollama", status="dead",
                         detail={"error": "connection refused"})
        record_heartbeat(db, component="index", status="alive")
        summary = build_weekly_summary(db)
        assert "ollama" in summary.lower()
        assert "dead" in summary.lower() or "degraded" in summary.lower()


# ---------------------------------------------------------------------------
# Embed vector leg section (2026-08-19 embed-resilience fix): the weekly
# summary is the search-stats surface; it must expose the degradation count.
# ---------------------------------------------------------------------------

def test_build_weekly_summary_exposes_embed_degradation_count():
    from claude_mem.telemetry import record_embed_degradation
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        _seed_week(db)
        record_embed_degradation(db, reason="EmbeddingError: ollama down")
        record_embed_degradation(db, reason="EmbeddingError: read timeout")
        summary = build_weekly_summary(db)
        assert "## Embed vector leg" in summary
        assert (
            "Degradation events (search fell back to BM25-only): 2"
            in summary
        )
        assert "WARN: vector leg degraded in window" in summary
        assert "EmbeddingError: read timeout" in summary  # latest reason


def test_build_weekly_summary_zero_degradations_no_warn():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        _seed_week(db)
        summary = build_weekly_summary(db)
        assert (
            "Degradation events (search fell back to BM25-only): 0"
            in summary
        )
        assert "WARN: vector leg degraded" not in summary


def test_build_weekly_summary_tolerates_legacy_db_without_table():
    """A telemetry.db created before the embed_degradation table existed
    must not break the report; the section is omitted (no instrumentation
    is different from zero events)."""
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE wrapper_invocations ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "prompt_hash TEXT, prompt_truncated TEXT, "
            "build_intent_fired INTEGER, investigation_intent_fired INTEGER, "
            "do_not_rebuild_warning_emitted INTEGER, "
            "stale_claim_warning_emitted INTEGER, "
            "retrieved_chunk_count INTEGER, retrieved_chunk_topics TEXT, "
            "retrieval_latency_ms INTEGER, session_id TEXT)"
        )
        conn.execute(
            "CREATE TABLE heartbeat ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
            "component TEXT NOT NULL, status TEXT NOT NULL, detail TEXT)"
        )
        conn.commit()
        conn.close()
        summary = build_weekly_summary(db)
        assert "## Embed vector leg" not in summary
