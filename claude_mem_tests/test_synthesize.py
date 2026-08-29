import tempfile
from pathlib import Path

from claude_mem.schema import init_db
from claude_mem.capture import CaptureStore, Thread, Decision, DeadEnd
from claude_mem.synthesize import (
    build_lineage_prompt,
    synthesize_lineage,
    structured_fallback,
    get_or_regenerate_lineage,
)


def _seeded_store(tmp):
    db = Path(tmp) / "index.db"
    init_db(db)
    store = CaptureStore(db)
    store.add_thread(Thread(name="Dialogue Grounding", id="dg", summary="two problems"))
    store.add_decision(Decision(title="keep MLM for F8", date="2026-06-04",
                                state="confirmed", thread_id="dg",
                                options_rejected=["FM-head"], mike_approved=True))
    store.add_dead_end(DeadEnd(approach="GW path never built", date="2026-04-09",
                               state="confirmed", thread_id="dg"))
    return store, db


def test_build_prompt_includes_dated_rows():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _seeded_store(tmp)
        t = store.get_thread("dg")
        decs = store.list_decisions(thread_id="dg")
        des = store.list_dead_ends(thread_id="dg")
        store.close()
        prompt = build_lineage_prompt(t, decs, des)
        assert "Dialogue Grounding" in prompt
        assert "2026-06-04" in prompt and "keep MLM for F8" in prompt
        assert "2026-04-09" in prompt and "GW path never built" in prompt


def test_synthesize_uses_injected_generator():
    calls = []

    def _gen(p):
        calls.append(p)
        return "2026-04-09 GW design -> 2026-06-04 R3."

    with tempfile.TemporaryDirectory() as tmp:
        store, db = _seeded_store(tmp)
        t = store.get_thread("dg")
        decs = store.list_decisions(thread_id="dg")
        des = store.list_dead_ends(thread_id="dg")
        store.close()
        out = synthesize_lineage(t, decs, des, _gen)
        assert "GW design" in out
        assert len(calls) == 1


def test_structured_fallback_contains_rows():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _seeded_store(tmp)
        t = store.get_thread("dg")
        decs = store.list_decisions(thread_id="dg")
        des = store.list_dead_ends(thread_id="dg")
        store.close()
        block = structured_fallback(t, decs, des)
        assert "Dialogue Grounding" in block
        assert "keep MLM for F8" in block
        assert "GW path never built" in block


def test_get_or_regenerate_caches_on_last_updated():
    calls = []

    def _gen(p):
        calls.append(p)
        return "LINEAGE v1"

    with tempfile.TemporaryDirectory() as tmp:
        store, db = _seeded_store(tmp)
        out1 = get_or_regenerate_lineage(store, "dg", _gen)
        assert out1 == "LINEAGE v1"
        assert len(calls) == 1
        out2 = get_or_regenerate_lineage(store, "dg", _gen)
        assert out2 == "LINEAGE v1"
        assert len(calls) == 1
        import time
        time.sleep(1.05)
        store.touch_thread("dg")
        get_or_regenerate_lineage(store, "dg", _gen)
        assert len(calls) == 2
        store.close()


def test_get_or_regenerate_unknown_thread_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        store, db = _seeded_store(tmp)
        assert get_or_regenerate_lineage(store, "nope", lambda p: "x") == ""
        store.close()
