import tempfile
from pathlib import Path

from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db
from claude_mem.search import Searcher


class _ConstEmbedder:
    """Hand-rolled embedder substitute. Returns the same 1024-d vector
    regardless of input."""
    def __init__(self, value: float = 0.1, dim: int = 1024) -> None:
        self.value = value
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [self.value] * self.dim


def _embedder(value: float = 0.1):
    return _ConstEmbedder(value)


def _seed_index(db: Path) -> None:
    init_db(db)
    ing = Ingester(db_path=db, embedder=_embedder(0.1))
    chunks = [
        Chunk(content="use_apollo master switch for apollo hypothesis loop",
              source="doc", module="apollo", status="PRODUCTION",
              do_not_rebuild=True, signal_weight=50),
        Chunk(content="kmi flag dormant since run018", source="memory",
              module="kmi", status="DORMANT", do_not_rebuild=True,
              signal_weight=70),
        Chunk(content="bistability trace tick 201 critique saturation",
              source="doc", signal_weight=30),
    ]
    for c in chunks:
        ing.add(c)
    ing.close()


def test_bm25_returns_match():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search("apollo", top_k=3)
        s.close()
        assert len(results) >= 1
        assert any("apollo" in r["content"].lower() for r in results)


def test_filter_do_not_rebuild():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search(
            "apollo bistability",
            top_k=10,
            filter_do_not_rebuild=True,
        )
        s.close()
        for r in results:
            assert r["do_not_rebuild"] == 1


def test_signal_weight_boost_increases_final_score():
    """signal_weight=70 chunk has higher final_score than sw=30 chunk
    when both have identical fusion_score (same BM25 + vector position)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder(0.1))
        ing.add(Chunk(content="alpha low", source="doc", signal_weight=10))
        ing.add(Chunk(content="alpha high", source="doc", signal_weight=80))
        ing.close()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search("alpha", top_k=5)
        s.close()
        assert len(results) == 2
        # The 80-weight chunk should rank first
        assert results[0]["signal_weight"] == 80
        assert results[1]["signal_weight"] == 10


def test_empty_query_returns_no_results():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search("", top_k=5)
        s.close()
        assert results == []


def test_filter_operator_vetted_matches_decisions():
    """Production-realistic: corpus has 0 do_not_rebuild=1 chunks but many
    is_decision=1 chunks. filter_operator_vetted should match them."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder(0.1))
        ing.add(Chunk(
            content="decision: phase 4 ships pre-FM head before pre-RM head",
            source="memory", module="phase4",
            is_decision=True, signal_weight=30,
        ))
        ing.add(Chunk(
            content="random low-signal noise chunk about phase 4",
            source="doc", module="phase4", signal_weight=10,
        ))
        ing.close()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search(
            "phase 4 head", top_k=10, filter_operator_vetted=True,
        )
        s.close()
        # The decision chunk passes; the low-signal one is excluded
        assert len(results) == 1
        assert results[0]["is_decision"] == 1


def test_filter_operator_vetted_matches_high_signal_weight():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder(0.1))
        ing.add(Chunk(
            content="alpha component A high-signal", source="doc",
            signal_weight=60,
        ))
        ing.add(Chunk(
            content="alpha component B low-signal", source="doc",
            signal_weight=20,
        ))
        ing.close()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search(
            "alpha", top_k=10, filter_operator_vetted=True,
        )
        s.close()
        assert len(results) == 1
        assert results[0]["signal_weight"] == 60


def test_filter_operator_vetted_matches_corrections():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder(0.1))
        ing.add(Chunk(
            content="correction: bias gradient was wrong, fixed in 0c3fd98f",
            source="memory", is_correction=True, signal_weight=10,
        ))
        ing.add(Chunk(
            content="bias gradient implementation note", source="doc",
            signal_weight=10,
        ))
        ing.close()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search(
            "bias gradient", top_k=10, filter_operator_vetted=True,
        )
        s.close()
        assert len(results) == 1
        assert results[0]["is_correction"] == 1


def test_filter_operator_vetted_excludes_pure_noise():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder(0.1))
        ing.add(Chunk(
            content="apollo orbital mechanics low-signal note",
            source="doc", signal_weight=5,
        ))
        ing.close()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search(
            "apollo", top_k=10, filter_operator_vetted=True,
        )
        s.close()
        # No is_decision / is_correction / DNR / sw>=50 -> excluded
        assert results == []


def test_bm25_pathological_query_is_bounded():
    """Regression for the 2026-07-06 hook outage: an FTS5 MATCH built from
    an unbounded tool payload (tens of thousands of OR terms) allocated
    >12 GB inside sqlite and hung the hook while it held the write lock.
    A machine-generated mega-query must complete promptly, and a relevant
    token in the payload HEAD must still rank (order-preserving cap)."""
    import time
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        # Relevant term in the head, then a flood of distinct junk terms,
        # heavy duplication, and over-long hash/base64-style blobs.
        junk = [f"junktoken{i}" for i in range(30000)]
        blobs = ["x" * 200] * 500
        query = " ".join(["apollo"] + junk + junk + blobs)
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        t0 = time.monotonic()
        results = s.search(query, top_k=3)
        elapsed = time.monotonic() - t0
        s.close()
        assert elapsed < 10.0, f"bounded query took {elapsed:.1f}s"
        assert any("apollo" in r["content"].lower() for r in results)


def test_bm25_all_overlong_tokens_returns_empty():
    """Above the cap, tokens longer than _BM25_MAX_TOKEN_LEN are dropped;
    a payload of only such blobs must degrade to no-match, not explode."""
    from claude_mem.search import _BM25_MAX_TERMS
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        query = " ".join(f"{'y' * 100}{i}" for i in range(_BM25_MAX_TERMS + 50))
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s._bm25_search(query, 5)
        s.close()
        assert results == []


def test_bm25_small_query_path_unchanged():
    """Queries at or below _BM25_MAX_TERMS bypass normalization entirely
    (duplicates and long tokens pass through as before the cap)."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        results = s.search("apollo apollo " + "z" * 100, top_k=3)
        s.close()
        assert any("apollo" in r["content"].lower() for r in results)


# ---------------------------------------------------------------------------
# Vector-leg degradation telemetry (2026-08-19 embed-resilience fix): an
# embed failure must no longer be swallowed silently -- search stays
# BM25-only (unchanged) but ONE embed_degradation telemetry row per process
# (log-once) records that the vector leg died.
# ---------------------------------------------------------------------------

class _FailingEmbedder:
    """Embedder double whose embed() always raises (dead/cold Ollama)."""
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("ollama request failed: connect timeout")


def _degradation_rows(telemetry_db: Path) -> list:
    import sqlite3
    if not telemetry_db.is_file():
        return []
    conn = sqlite3.connect(telemetry_db)
    try:
        return conn.execute(
            "SELECT reason FROM embed_degradation").fetchall()
    finally:
        conn.close()


def test_embed_failure_still_returns_bm25_results_and_logs_once():
    from claude_mem.search import _reset_embed_degradation_logged
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        _reset_embed_degradation_logged()
        s = Searcher(db_path=db, embedder=_FailingEmbedder())
        try:
            results = s.search("apollo", top_k=3)
            # BM25 leg still answers (graceful degradation unchanged)
            assert any("apollo" in r["content"].lower() for r in results)
            # ... but the degradation is RECORDED, exactly once, in the
            # sibling telemetry.db (the ProjectConfig layout).
            rows = _degradation_rows(Path(tmp) / "telemetry.db")
            assert len(rows) == 1
            assert "RuntimeError" in rows[0][0]
            assert "connect timeout" in rows[0][0]
            # Second failing search in the SAME process: still one row.
            s.search("apollo", top_k=3)
            assert len(_degradation_rows(Path(tmp) / "telemetry.db")) == 1
        finally:
            s.close()


def test_embed_failure_logs_again_in_a_fresh_process():
    """The guard is process-lifetime: a new process (simulated via the
    reset seam) records its own degradation row."""
    from claude_mem.search import _reset_embed_degradation_logged
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        _reset_embed_degradation_logged()
        s = Searcher(db_path=db, embedder=_FailingEmbedder())
        s.search("apollo", top_k=3)
        s.close()
        _reset_embed_degradation_logged()  # simulate a new process
        s2 = Searcher(db_path=db, embedder=_FailingEmbedder())
        s2.search("apollo", top_k=3)
        s2.close()
        assert len(_degradation_rows(Path(tmp) / "telemetry.db")) == 2


def test_embed_success_writes_no_degradation_row():
    from claude_mem.search import _reset_embed_degradation_logged
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        _seed_index(db)
        _reset_embed_degradation_logged()
        s = Searcher(db_path=db, embedder=_embedder(0.1))
        s.search("apollo", top_k=3)
        s.close()
        assert _degradation_rows(Path(tmp) / "telemetry.db") == []


def test_explicit_telemetry_path_override_is_honored():
    from claude_mem.search import _reset_embed_degradation_logged
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        custom = Path(tmp) / "elsewhere" / "custom-telemetry.db"
        _seed_index(db)
        _reset_embed_degradation_logged()
        s = Searcher(
            db_path=db, embedder=_FailingEmbedder(), telemetry_path=custom,
        )
        s.search("apollo", top_k=3)
        s.close()
        assert len(_degradation_rows(custom)) == 1
        assert _degradation_rows(Path(tmp) / "telemetry.db") == []
