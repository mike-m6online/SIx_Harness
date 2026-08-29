import sqlite3
import tempfile
from pathlib import Path

from claude_mem.telemetry import (
    init_telemetry_db, record_heartbeat, record_wrapper_invocation,
    probe_components,
)


def test_init_telemetry_db_creates_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        conn = sqlite3.connect(db)
        try:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            conn.close()
        expected = {"wrapper_invocations", "heartbeat"}
        assert expected.issubset(tables)


def test_record_wrapper_invocation_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        record_wrapper_invocation(
            db, prompt_truncated="let's build X",
            build_intent_fired=True,
            investigation_intent_fired=False,
            do_not_rebuild_warning_emitted=True,
            stale_claim_warning_emitted=False,
            retrieved_chunk_count=5,
            retrieved_chunk_topics=["apollo", "kmi"],
            retrieval_latency_ms=120,
            session_id="abc",
        )
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT build_intent_fired, retrieved_chunk_count "
                "FROM wrapper_invocations"
            ).fetchone()
        finally:
            conn.close()
        assert row == (1, 5)


def test_record_heartbeat_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        record_heartbeat(db, component="ollama", status="alive",
                         detail={"latency_ms": 42})
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT component, status FROM heartbeat"
            ).fetchone()
        finally:
            conn.close()
        assert row == ("ollama", "alive")


def test_probe_components_returns_status_dict(monkeypatch):
    """probe_components hits each component (ollama, index, sessionstart)
    and returns a {component: (status, detail)} dict."""
    with tempfile.TemporaryDirectory() as tmp:
        from claude_mem.config import ProjectConfig
        cfg = ProjectConfig(project_root=Path(tmp))
        cfg.write()
        from claude_mem.schema import init_db
        init_db(cfg.db_path)
        # Substitute the ollama probe to avoid a real network call
        # (keep_alive kwarg: 2026-08-19 embed-resilience fix)
        monkeypatch.setattr(
            "claude_mem.telemetry._probe_ollama",
            lambda endpoint, model, keep_alive="4h": (
                "alive", {"endpoint": endpoint}),
        )
        statuses = probe_components(cfg)
        assert "ollama" in statuses
        assert "index" in statuses
        assert statuses["ollama"][0] == "alive"
        assert statuses["index"][0] == "alive"


# ---------------------------------------------------------------------------
# embed_degradation (2026-08-19 embed-resilience fix)
# ---------------------------------------------------------------------------

def test_init_telemetry_db_creates_embed_degradation_table():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        conn = sqlite3.connect(db)
        try:
            tables = {
                row[0] for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                )
            }
        finally:
            conn.close()
        assert "embed_degradation" in tables


def test_record_embed_degradation_writes_row():
    from claude_mem.telemetry import record_embed_degradation
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        record_embed_degradation(db, reason="EmbeddingError: ollama down")
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT timestamp, reason FROM embed_degradation"
            ).fetchall()
        finally:
            conn.close()
        assert len(rows) == 1
        assert rows[0][1] == "EmbeddingError: ollama down"
        assert rows[0][0]  # ISO timestamp present


def test_record_embed_degradation_never_raises(tmp_path):
    """Same contract as record_hook_heartbeat: degradation bookkeeping
    must never be the reason a retrieval breaks. A directory path (an
    unopenable database) must be swallowed."""
    from claude_mem.telemetry import record_embed_degradation
    record_embed_degradation(tmp_path, reason="x")  # tmp_path is a dir


def test_probe_ollama_carries_keep_alive(monkeypatch):
    """The heartbeat probe is a real Ollama request: omitting keep_alive
    would RESET the residency timer to the 5m server default and silently
    undo the embed path keep_alive."""
    import claude_mem.telemetry as telemetry

    captured: dict = {}

    class _FakeResponse:
        status_code = 200
        text = ""

    class _FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def post(self, url, json=None, **kw):
            captured["json"] = json
            return _FakeResponse()

    monkeypatch.setattr(telemetry.httpx, "Client", _FakeClient)
    status, detail = telemetry._probe_ollama(
        "http://localhost:11434", "qwen3-embedding:0.6b", keep_alive="2h",
    )
    assert status == "alive"
    assert captured["json"]["keep_alive"] == "2h"
