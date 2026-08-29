"""Tests for the capture-triage loop (spec R5/R6, Task 6).

Covers:
  - CaptureStore triage primitives: reject_decision, set_decision_thread
    (attach + optional retitle, bumps thread.last_updated).
  - `capture-triage` CLI: numbered review-sheet render over pending
    decisions/dead-ends; `--apply <json>` batch-applies verdicts.
  - The apply path unfreezes synthesize's cache gate: attaching a
    confirmed decision to a thread bumps last_updated past the cached
    lineage_cache_key, so get_or_regenerate_lineage regenerates.
"""
import json
import sqlite3
import tempfile
import time
from pathlib import Path

from click.testing import CliRunner

from claude_mem.capture import CaptureStore, Thread, Decision, DeadEnd
from claude_mem.cli import cli
from claude_mem.schema import init_db
from claude_mem.synthesize import get_or_regenerate_lineage


def _store(tmp):
    db = Path(tmp) / "index.db"
    init_db(db)
    return CaptureStore(db), db


def _init_project(tmp: str) -> Path:
    root = Path(tmp)
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    return root


# ---- CaptureStore primitives ----

def test_reject_decision_sets_state():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        d = Decision(title="junk fragment", date="2026-06-01")
        store.add_decision(d)
        store.reject_decision(d.id)
        row = store.list_decisions()[0]
        assert row["state"] == "rejected"
        store.close()


def test_set_decision_thread_attaches_and_bumps_last_updated():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        before = store.get_thread("t")["last_updated"]
        d = Decision(title="a real decision", date="2026-06-01")
        store.add_decision(d)
        time.sleep(1.05)  # iso-seconds resolution
        store.set_decision_thread(d.id, "t")
        row = [r for r in store.list_decisions() if r["id"] == d.id][0]
        assert row["thread_id"] == "t"
        after = store.get_thread("t")["last_updated"]
        assert after != before
        store.close()


def test_set_decision_thread_can_edit_title():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        d = Decision(title="raw fragment title", date="2026-06-01")
        store.add_decision(d)
        store.set_decision_thread(d.id, "t", title="cleaned up title")
        row = [r for r in store.list_decisions() if r["id"] == d.id][0]
        assert row["title"] == "cleaned up title"
        store.close()


def test_set_decision_thread_rejects_unknown_thread():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        d = Decision(title="x", date="2026-06-01")
        store.add_decision(d)
        try:
            store.set_decision_thread(d.id, "no-such-thread")
            assert False, "expected ValueError"
        except ValueError:
            pass
        store.close()


def test_reject_dead_end_sets_state():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        e = DeadEnd(approach="junk", date="2026-06-01")
        store.add_dead_end(e)
        store.reject_dead_end(e.id)
        row = store.list_dead_ends()[0]
        assert row["state"] == "rejected"
        store.close()


def test_set_dead_end_thread_attaches_and_bumps_last_updated():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        before = store.get_thread("t")["last_updated"]
        e = DeadEnd(approach="a real dead end", date="2026-06-01")
        store.add_dead_end(e)
        time.sleep(1.05)
        store.set_dead_end_thread(e.id, "t")
        row = [r for r in store.list_dead_ends() if r["id"] == e.id][0]
        assert row["thread_id"] == "t"
        after = store.get_thread("t")["last_updated"]
        assert after != before
        store.close()


# ---- capture-triage CLI: render ----

def test_supersede_dead_end_sets_superseded_by_and_bumps_thread():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t"))
        e = DeadEnd(approach="GW path designed but never built",
                    date="2026-04-09", thread_id="t", state="confirmed")
        store.add_dead_end(e)
        before = store.get_thread("t")["last_updated"]
        time.sleep(1.05)

        store.supersede_dead_end(e.id, "the 2026-06-30 verified-built decision")

        row = [r for r in store.list_dead_ends() if r["id"] == e.id][0]
        assert row["superseded_by"] == "the 2026-06-30 verified-built decision"
        after = store.get_thread("t")["last_updated"]
        assert after != before
        store.close()


def test_supersede_dead_end_unattached_still_works():
    # A dead-end with no thread_id (thread_id=None) must not crash
    # touch_thread -- supersede is a no-op on last_updated in that case.
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        e = DeadEnd(approach="orphan dead end", date="2026-04-09")
        store.add_dead_end(e)
        store.supersede_dead_end(e.id, "something")
        row = [r for r in store.list_dead_ends() if r["id"] == e.id][0]
        assert row["superseded_by"] == "something"
        store.close()


def test_update_thread_summary_bumps_last_updated():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t", summary="stale claim never built"))
        before = store.get_thread("t")["last_updated"]
        time.sleep(1.05)

        store.update_thread_summary("t", "corrected summary: verified built")

        row = store.get_thread("t")
        assert row["summary"] == "corrected summary: verified built"
        assert row["last_updated"] != before
        store.close()


def test_capture_triage_renders_numbered_sheet(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_decision(Decision(title="pending decision one", date="2026-06-10"))
    store.add_dead_end(DeadEnd(approach="pending dead end one", date="2026-06-09"))
    store.close()

    runner = CliRunner()
    res = runner.invoke(cli, ["capture-triage", "--project-root", str(root)])
    assert res.exit_code == 0
    assert "pending decision one" in res.output
    assert "pending dead end one" in res.output
    # numbered / ids present so an operator (or --apply JSON) can reference rows
    assert "1." in res.output or "[1]" in res.output


def test_capture_triage_limit(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    for i in range(5):
        store.add_decision(Decision(title=f"decision {i}", date=f"2026-06-{10+i:02d}"))
    store.close()

    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root), "--limit", "2",
    ])
    assert res.exit_code == 0
    # only the 2 most recent (by date desc) should appear
    assert "decision 4" in res.output
    assert "decision 3" in res.output
    assert "decision 0" not in res.output


# ---- capture-triage CLI: --apply ----

def test_capture_triage_apply_confirms_and_rejects(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    d1 = Decision(title="keep this one", date="2026-06-10")
    d2 = Decision(title="harness junk fragment", date="2026-06-11")
    store.add_decision(d1)
    store.add_decision(d2)
    store.close()

    verdicts = [
        {"id": d1.id, "verdict": "confirm"},
        {"id": d2.id, "verdict": "reject"},
    ]
    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0, res.output

    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    try:
        s1 = conn.execute("SELECT state FROM decisions WHERE id=?", (d1.id,)).fetchone()[0]
        s2 = conn.execute("SELECT state FROM decisions WHERE id=?", (d2.id,)).fetchone()[0]
    finally:
        conn.close()
    assert s1 == "confirmed"
    assert s2 == "rejected"


def test_capture_triage_apply_attaches_to_thread_and_bumps_last_updated(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_thread(Thread(name="Inner Dialogue Drives Action",
                            id="inner-dialogue-drives-action"))
    before = store.get_thread("inner-dialogue-drives-action")["last_updated"]
    d = Decision(title="belongs to the thread", date="2026-06-10")
    store.add_decision(d)
    store.close()

    time.sleep(1.05)
    verdicts = [
        {"id": d.id, "verdict": "confirm", "thread": "inner-dialogue-drives-action"},
    ]
    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0, res.output

    store2 = CaptureStore(root / ".claude-mem" / "index.db")
    row = [r for r in store2.list_decisions() if r["id"] == d.id][0]
    assert row["state"] == "confirmed"
    assert row["thread_id"] == "inner-dialogue-drives-action"
    after = store2.get_thread("inner-dialogue-drives-action")["last_updated"]
    store2.close()
    assert after != before


def test_capture_triage_apply_title_edit(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    d = Decision(title="raw truncated fragment ...", date="2026-06-10")
    store.add_decision(d)
    store.close()

    verdicts = [
        {"id": d.id, "verdict": "confirm", "title_edit": "clean operator title"},
    ]
    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0, res.output

    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    try:
        title = conn.execute("SELECT title FROM decisions WHERE id=?", (d.id,)).fetchone()[0]
    finally:
        conn.close()
    assert title == "clean operator title"


def test_capture_triage_apply_new_thread(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    d = Decision(title="opens a fresh thread", date="2026-06-10")
    store.add_decision(d)
    store.close()

    verdicts = [
        {"id": d.id, "verdict": "confirm", "thread": "new:A Fresh Problem Thread"},
    ]
    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0, res.output

    store2 = CaptureStore(root / ".claude-mem" / "index.db")
    threads = {t["name"]: t for t in store2.list_threads()}
    assert "A Fresh Problem Thread" in threads
    row = [r for r in store2.list_decisions() if r["id"] == d.id][0]
    assert row["thread_id"] == threads["A Fresh Problem Thread"]["id"]
    store2.close()


def test_capture_triage_apply_dead_end_verdicts(tmp_path):
    root = _init_project(str(tmp_path))
    store = CaptureStore(root / ".claude-mem" / "index.db")
    e1 = DeadEnd(approach="keep this dead end", date="2026-06-10")
    e2 = DeadEnd(approach="junk harness fragment", date="2026-06-11")
    store.add_dead_end(e1)
    store.add_dead_end(e2)
    store.close()

    verdicts = [
        {"id": e1.id, "verdict": "confirm"},
        {"id": e2.id, "verdict": "reject"},
    ]
    runner = CliRunner()
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0, res.output

    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    try:
        s1 = conn.execute("SELECT state FROM dead_ends WHERE id=?", (e1.id,)).fetchone()[0]
        s2 = conn.execute("SELECT state FROM dead_ends WHERE id=?", (e2.id,)).fetchone()[0]
    finally:
        conn.close()
    assert s1 == "confirmed"
    assert s2 == "rejected"


def test_capture_triage_apply_unknown_id_reported_not_crashed(tmp_path):
    root = _init_project(str(tmp_path))
    runner = CliRunner()
    verdicts = [{"id": "does-not-exist", "verdict": "confirm"}]
    res = runner.invoke(cli, [
        "capture-triage", "--project-root", str(root),
        "--apply", json.dumps(verdicts),
    ])
    assert res.exit_code == 0
    assert "does-not-exist" in res.output


# ---- synthesize regenerates on stale cache key (apply -> lineage refresh) ----

def test_synthesize_regenerates_after_triage_attach_bumps_thread():
    calls = []

    def _gen(p):
        calls.append(p)
        return "LINEAGE v2 mentions the newly attached decision."

    with tempfile.TemporaryDirectory() as tmp:
        store, db = _store(tmp)
        store.add_thread(Thread(name="T", id="t", summary="s"))
        # Prime a cached lineage keyed on the thread's current last_updated.
        t0 = store.get_thread("t")
        store.set_cached_lineage("t", "LINEAGE v1 (stale)", t0["last_updated"])
        cached_before, key_before = store.get_cached_lineage("t")
        assert cached_before == "LINEAGE v1 (stale)"

        # Simulate the triage --apply path: attach a new confirmed decision,
        # which must bump threads.last_updated past the cached key.
        time.sleep(1.05)
        d = Decision(title="new evidence", date="2026-06-30", state="confirmed")
        store.add_decision(d)
        store.set_decision_thread(d.id, "t")

        out = get_or_regenerate_lineage(store, "t", _gen)
        assert out == "LINEAGE v2 mentions the newly attached decision."
        assert len(calls) == 1
        store.close()
