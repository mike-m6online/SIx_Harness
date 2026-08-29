import sqlite3
import tempfile
from pathlib import Path

import sqlite_vec

from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _StaticEmbedder:
    """Hand-rolled embedder substitute returning a deterministic 1024-d
    vector keyed off the input length."""
    def __init__(self, modulus: int = 7, dim: int = 1024) -> None:
        self.modulus = modulus
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [float(len(text) % self.modulus) / self.modulus] * self.dim


def _embedder():
    return _StaticEmbedder()


class _TransactionAssertingEmbedder:
    """Test-double embedder that pins the fix for the write-lock-held-
    across-embed bug: asserts no SQLite transaction is open on the
    connection at the moment embed() is called. Ingester.add() must call
    embed() BEFORE its first INSERT (which is what opens the implicit
    transaction under sqlite3's default isolation_level) -- if a future
    change reorders this back to embed-inside-the-transaction, this
    assertion fails immediately instead of only showing up as SQLITE_BUSY
    under real concurrency."""

    def __init__(self, conn: sqlite3.Connection, dim: int = 1024) -> None:
        self.conn = conn
        self.dim = dim
        self.calls = 0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        assert self.conn.in_transaction is False, (
            "embed() was called while a write transaction was already "
            "open -- this holds the SQLite write lock across the network "
            "call and starves concurrent writers (Task-4 review Finding 1)"
        )
        return [float(len(text) % 7) / 7] * self.dim


def test_ingest_writes_chunk_to_fts_and_vec():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        chunk = Chunk(
            content="use_apollo is the master switch for the apollo loop.",
            source="doc",
            role="doc",
            file_path="docs/apollo.md",
            module="apollo",
            status="PRODUCTION",
            signal_weight=50,
        )
        ing.add(chunk)
        conn = sqlite3.connect(db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            fts_count = conn.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'apollo'"
            ).fetchone()[0]
        finally:
            conn.close()
        ing.close()
        assert count == 1
        assert fts_count == 1


def test_ingest_dedupes_identical_content():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        chunk = Chunk(content="identical content", source="doc")
        first = ing.add(chunk)
        second = ing.add(chunk)
        conn = sqlite3.connect(db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        ing.close()
        assert first is True
        assert second is False
        assert count == 1


def test_record_ingestion_writes_audit_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.record_ingestion(
            source_path="docs/foo.md",
            chunks_added=3,
            file_mtime=12345.0,
            chunk_ids=["a", "b", "c"],
        )
        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT source_path, chunks_added FROM ingestion_log"
            ).fetchone()
        finally:
            conn.close()
        ing.close()
        assert row == ("docs/foo.md", 3)


def test_chunk_aliases_persisted_and_searchable():
    """aliases are stored in the chunks row and indexed in chunks_fts.

    The chunk content itself does NOT contain the expansion phrase; the
    FTS match comes purely from the aliases column populated by
    derive_aliases().
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        # "CWM" acronym in content -> aliases should contain "causal world model"
        chunk = Chunk(
            content="row about the CWM kernel",
            source="doc",
            aliases="causal world model",
        )
        ing.add(chunk)
        ing.close()  # release before querying so Windows doesn't lock the file
        conn = sqlite3.connect(db)
        try:
            # Verify the aliases column was persisted
            stored_aliases = conn.execute(
                "SELECT aliases FROM chunks WHERE id = ?", (chunk.id,)
            ).fetchone()[0]
            # The chunk is findable via the aliases column through FTS
            # FTS5 external-content tables: must JOIN chunks_fts.rowid = chunks.rowid
            fts_rows = conn.execute(
                """
                SELECT c.id
                FROM chunks_fts f
                JOIN chunks c ON c.rowid = f.rowid
                WHERE chunks_fts MATCH 'causal'
                """
            ).fetchall()
        finally:
            conn.close()
        assert stored_aliases == "causal world model"
        assert len(fts_rows) == 1
        assert fts_rows[0][0] == chunk.id


def test_embed_runs_before_write_transaction_opens():
    """Task-4 review Finding 1 (Important): Ingester.add() must call
    embed() BEFORE it opens a SQLite write transaction. Previously the
    order was INSERT chunks -> embed() (network, up to 60s) -> INSERT
    chunks_vec -> commit(), all inside one implicit transaction -- the
    write lock sat open for the entire embed() round-trip. A concurrent
    writer (e.g. embed-backfill running at the same time as incremental
    ingestion, which is concurrent-by-default since Task 4's SessionEnd
    wiring) hits SQLITE_BUSY once its busy_timeout elapses.

    The test-double embedder asserts conn.in_transaction is False at the
    moment it is called -- pins the ordering directly rather than relying
    on timing/flakiness to expose the bug.
    """
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=None)  # type: ignore[arg-type]
        embedder = _TransactionAssertingEmbedder(ing._conn)
        ing.embedder = embedder
        chunk = Chunk(content="embed ordering must precede the write transaction", source="doc")

        result = ing.add(chunk)
        ing.close()

        assert result is True
        assert embedder.calls == 1


def test_embed_failure_still_inserts_chunk_without_vector():
    """Ordering change must not regress the existing failure semantics:
    a chunk whose embed() raises is still inserted (searchable via BM25)
    and the failure is logged to ingestion_log, exactly as before -- only
    the ORDER of operations moved, not the behavior."""
    class _AlwaysFails:
        def embed(self, text: str) -> list[float]:
            raise RuntimeError("simulated embed failure")

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_AlwaysFails())
        chunk = Chunk(content="content whose embedding will fail", source="doc")
        result = ing.add(chunk)
        ing.close()  # release before querying so Windows doesn't lock the file
        conn = sqlite3.connect(db)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        try:
            chunk_count = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE id = ?", (chunk.id,)
            ).fetchone()[0]
            vec_count = conn.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (chunk.id,)
            ).fetchone()[0]
            fail_count = conn.execute(
                "SELECT COUNT(*) FROM ingestion_log WHERE action = 'embed_fail'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert result is True
        assert chunk_count == 1
        assert vec_count == 0
        assert fail_count == 1


def test_dimension_mismatch_embed_failure_logs_and_skips_vector():
    """A real EmbeddingClient refusing a dimensionally-incompatible
    fallback vector (EmbeddingError) must flow through Ingester.add()'s
    existing generic failure path exactly like any other embed()
    exception: chunk inserted without a vector, failure logged to
    ingestion_log with a detail string that names the incompatibility.
    Root cause pinned here: qwen3-embedding:0.6b (1024-dim) falling back
    to nomic-embed-text (768-dim) against a vec table fixed at 1024 dims
    used to "succeed" at embed() and then fail the sqlite-vec INSERT
    with an opaque 'Dimension mismatch' error far from this call site."""
    from claude_mem.embed import EmbeddingError

    class _WrongDimFallback:
        def embed(self, text: str) -> list[float]:
            raise EmbeddingError(
                "fallback model dimension 768 != 1024 -- refusing "
                "incompatible vector"
            )

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_WrongDimFallback())
        chunk = Chunk(content="content whose fallback embed is wrong-dim", source="doc")
        result = ing.add(chunk)
        ing.close()
        conn = sqlite3.connect(db)
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        try:
            vec_count = conn.execute(
                "SELECT COUNT(*) FROM chunks_vec WHERE chunk_id = ?", (chunk.id,)
            ).fetchone()[0]
            row = conn.execute(
                "SELECT detail FROM ingestion_log WHERE action = 'embed_fail'"
            ).fetchone()
        finally:
            conn.close()
        assert result is True
        assert vec_count == 0
        assert row is not None
        assert "768" in row[0] and "1024" in row[0]


def test_ingester_connection_has_raised_busy_timeout():
    """Belt-and-braces fix alongside the embed-ordering fix: the
    Ingester's connection sets busy_timeout well above the live DB's
    busy_timeout=5000 default, so brief lock contention on the short
    local-write transaction (not the long embed-network-call case, which
    the ordering fix eliminates entirely) waits instead of failing fast
    with SQLITE_BUSY."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        busy_timeout = ing._conn.execute("PRAGMA busy_timeout").fetchone()[0]
        ing.close()
        assert busy_timeout >= 30000
