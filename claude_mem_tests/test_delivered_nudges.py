from claude_mem.capture import CaptureStore
from claude_mem.schema import init_db


def test_delivered_nudges_suppress_and_record(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    store = CaptureStore(db)
    try:
        assert store.was_delivered("sessA", "chunk1") is False
        store.record_delivered("sessA", "chunk1", "Edit")
        assert store.was_delivered("sessA", "chunk1") is True
        # idempotent: recording twice does not raise and stays delivered
        store.record_delivered("sessA", "chunk1", "Edit")
        assert store.was_delivered("sessA", "chunk1") is True
        # scoped per session: a different session has not seen it
        assert store.was_delivered("sessB", "chunk1") is False
    finally:
        store.close()
