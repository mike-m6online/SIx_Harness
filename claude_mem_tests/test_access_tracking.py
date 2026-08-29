"""Tests for the usage-feedback wire: access_count + last_accessed.

These columns are declared in schema.py but were never written before
this change. search.py's Searcher.search() and the session_start hook now
bump them whenever a chunk is surfaced for injection. Critically, the
bookkeeping is failure-safe: an UPDATE error must never break retrieval.
"""
import sqlite3
import tempfile
from pathlib import Path

from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db
from claude_mem.search import Searcher


class _ConstEmbedder:
    def __init__(self, value: float = 0.1, dim: int = 1024) -> None:
        self.value = value
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [self.value] * self.dim


def _embedder():
    return _ConstEmbedder()


def _seed(db: Path) -> None:
    init_db(db)
    ing = Ingester(db_path=db, embedder=_embedder())
    ing.add(Chunk(content="apollo hypothesis loop master switch", source="doc"))
    ing.add(Chunk(content="kmi flag dormant since run018", source="memory"))
    ing.close()


def _access_row(db: Path, content_substr: str):
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT access_count, last_accessed FROM chunks "
            "WHERE content LIKE ?",
            (f"%{content_substr}%",),
        ).fetchone()
    finally:
        conn.close()
    return row


def test_search_increments_access_count_and_sets_last_accessed():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed(db)

        before = _access_row(db, "apollo")
        assert before["access_count"] == 0
        assert before["last_accessed"] is None

        s = Searcher(db_path=db, embedder=_embedder())
        results = s.search("apollo", top_k=3)
        s.close()
        assert any("apollo" in r["content"].lower() for r in results)

        after = _access_row(db, "apollo")
        assert after["access_count"] == 1
        assert after["last_accessed"] is not None


def test_repeated_search_accumulates_access_count():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed(db)
        s = Searcher(db_path=db, embedder=_embedder())
        for _ in range(3):
            s.search("apollo", top_k=3)
        s.close()
        after = _access_row(db, "apollo")
        assert after["access_count"] == 3


def test_only_surfaced_chunks_are_bumped():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed(db)
        s = Searcher(db_path=db, embedder=_embedder())
        # Query matches only the apollo chunk via BM25 (the const embedder
        # makes both equidistant in vector space, but top_k limits how many
        # are surfaced). Force a precise query + top_k=1.
        s.search("apollo", top_k=1)
        s.close()
        apollo = _access_row(db, "apollo")
        kmi = _access_row(db, "kmi")
        assert apollo["access_count"] == 1
        # The kmi chunk was never the top surfaced result for "apollo".
        assert kmi["access_count"] == 0


def test_update_failure_does_not_break_retrieval(tmp_path, monkeypatch):
    """A forced failure inside the bookkeeping write must be swallowed and
    the search results returned intact. We make the access-recording body
    raise (here, by breaking the timestamp call it makes), proving the
    try/except guard never lets usage bookkeeping break retrieval."""
    import claude_mem.search as search_mod

    db = tmp_path / "index.db"
    _seed(db)

    class _ExplodingDatetime:
        @staticmethod
        def now(*_a, **_k):
            raise sqlite3.OperationalError("simulated bookkeeping failure")

    # _record_access calls datetime.now(timezone.utc) first; making that
    # raise drives the except-guard. Any exception type is swallowed.
    monkeypatch.setattr(search_mod, "datetime", _ExplodingDatetime)

    s = Searcher(db_path=db, embedder=_embedder())
    try:
        results = s.search("apollo", top_k=3)
        # Retrieval still succeeded despite the bookkeeping write failing.
        assert any("apollo" in r["content"].lower() for r in results)
    finally:
        s.close()

    # And the access_count was NOT bumped (the write never completed).
    after = _access_row(db, "apollo")
    assert after["access_count"] == 0


def test_record_access_empty_ids_is_noop():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed(db)
        s = Searcher(db_path=db, embedder=_embedder())
        # Empty query returns no results -> _record_access([]) is a no-op.
        results = s.search("", top_k=3)
        s.close()
        assert results == []
        apollo = _access_row(db, "apollo")
        assert apollo["access_count"] == 0
