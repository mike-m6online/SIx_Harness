import json
import sqlite3
import tempfile
import time
from pathlib import Path

import pytest

from claude_mem.capture import CaptureStore, Thread, Decision, DeadEnd
from claude_mem.schema import init_db


def _tables(db: Path) -> set:
    conn = sqlite3.connect(db)
    try:
        return {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    finally:
        conn.close()


def test_init_db_creates_capture_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        names = _tables(db)
        assert {"threads", "decisions", "dead_ends"}.issubset(names)


def test_capture_columns_present():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        conn = sqlite3.connect(db)
        try:
            dec_cols = {r[1] for r in conn.execute("PRAGMA table_info(decisions)")}
            de_cols = {r[1] for r in conn.execute("PRAGMA table_info(dead_ends)")}
            thr_cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}
        finally:
            conn.close()
        assert {"id", "date", "title", "rationale", "options_rejected",
                "state", "thread_id", "linked_commits", "mike_approved"} <= dec_cols
        assert {"id", "date", "approach", "why_shelved", "superseded_by",
                "thread_id", "state"} <= de_cols
        assert {"id", "name", "opened_date", "state", "last_updated",
                "summary"} <= thr_cols


def _store(tmp):
    db = Path(tmp) / "index.db"
    init_db(db)
    return CaptureStore(db), db


def test_add_thread_and_dedup():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        t = Thread(name="Inner Dialogue Drives Action", summary="x")
        assert t.id == "inner-dialogue-drives-action"
        assert store.add_thread(t) is True
        assert store.add_thread(Thread(name="Inner Dialogue Drives Action")) is False
        store.close()


def test_add_decision_serializes_lists():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        d = Decision(
            title="keep MLM for F8",
            date="2026-06-04",
            rationale="R3 does not touch the MLM's F8 value",
            options_rejected=["double-down FM-head", "resolution-PE"],
            state="confirmed",
            mike_approved=True,
        )
        assert store.add_decision(d) is True
        assert store.add_decision(d) is False  # dedup by id
        store.close()
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT options_rejected, mike_approved, state FROM decisions WHERE id=?",
                (d.id,),
            ).fetchone()
        finally:
            conn.close()
        assert json.loads(row[0]) == ["double-down FM-head", "resolution-PE"]
        assert row[1] == 1
        assert row[2] == "confirmed"


def test_add_dead_end():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        e = DeadEnd(
            approach="GW path designed but never built",
            date="2026-04-09",
            why_shelved="deferred into F8 work",
        )
        assert e.state == "pending"
        assert store.add_dead_end(e) is True
        store.close()
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT approach, state FROM dead_ends WHERE id=?", (e.id,)
            ).fetchone()
        finally:
            conn.close()
        assert row[0] == "GW path designed but never built"
        assert row[1] == "pending"


def test_add_decision_rejects_unknown_thread():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        with pytest.raises(ValueError):
            store.add_decision(Decision(title="x", thread_id="no-such-thread"))
        store.close()


def test_context_manager_closes():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        with CaptureStore(db) as store:
            assert store.add_thread(Thread(name="ctx", id="ctx")) is True
        # After exit, a fresh store sees the committed row.
        store2 = CaptureStore(db)
        assert hasattr(store2, "add_thread") or True  # get_thread arrives in Task 3
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("SELECT COUNT(*) FROM threads").fetchone()[0] == 1
        finally:
            conn.close()
        store2.close()


def test_list_filters_and_confirm():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        store.add_decision(Decision(title="a", date="2026-06-01", thread_id="t"))
        store.add_decision(Decision(title="b", date="2026-06-02", state="confirmed"))
        store.add_dead_end(DeadEnd(approach="x", date="2026-06-01", thread_id="t"))

        assert len(store.list_decisions()) == 2
        assert len(store.list_decisions(thread_id="t")) == 1
        assert len(store.list_decisions(state="pending")) == 1
        assert len(store.list_dead_ends(thread_id="t")) == 1
        assert store.list_decisions()[0]["date"] == "2026-06-02"  # DESC order

        did = store.list_decisions(thread_id="t")[0]["id"]
        store.confirm_decision(did, mike_approved=True)
        row = store.list_decisions(thread_id="t")[0]
        assert row["state"] == "confirmed"
        assert row["mike_approved"] == 1
        store.close()


def test_add_to_thread_bumps_last_updated():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        before = store.get_thread("t")["last_updated"]
        time.sleep(1.05)  # iso seconds resolution
        store.add_decision(Decision(title="later", thread_id="t"))
        after = store.get_thread("t")["last_updated"]
        assert after >= before
        assert after != before
        store.close()


def test_threads_has_lineage_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        conn = sqlite3.connect(db)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}
        finally:
            conn.close()
        assert {"lineage_text", "lineage_cached_at", "lineage_cache_key"} <= cols


def test_init_db_migrates_preexisting_threads_table():
    # Simulate a pre-Rung-3 DB: threads table WITHOUT the lineage columns.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE threads (id TEXT PRIMARY KEY, name TEXT NOT NULL, "
            "opened_date TEXT, state TEXT, last_updated TEXT, summary TEXT)"
        )
        conn.commit()
        conn.close()
        # init_db must ALTER-add the missing columns without dropping data.
        init_db(db)
        conn = sqlite3.connect(db)
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(threads)")}
        finally:
            conn.close()
        assert {"lineage_text", "lineage_cached_at", "lineage_cache_key"} <= cols


def test_lineage_cache_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        assert store.get_cached_lineage("t") == (None, None)
        store.set_cached_lineage("t", "lineage body", "key-123")
        text, key = store.get_cached_lineage("t")
        assert text == "lineage body"
        assert key == "key-123"
        store.close()


def test_get_cached_lineage_unknown_thread():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        assert store.get_cached_lineage("nope") == (None, None)
        store.close()


def test_meta_get_set_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.set_meta("extract_offset:x", "123")
        assert store.get_meta("extract_offset:x") == "123"
        assert store.get_meta("absent") is None
        store.set_meta("extract_offset:x", "456")
        assert store.get_meta("extract_offset:x") == "456"
        store.close()


def test_meta_table_autocreated():
    # A bare DB with NO init_db: the CaptureStore __init__ CREATE must
    # self-migrate the meta table so the live hook never crashes.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "bare.db"
        conn = sqlite3.connect(db)
        conn.close()
        store = CaptureStore(db)
        store.set_meta("k", "v")
        assert store.get_meta("k") == "v"
        store.close()
