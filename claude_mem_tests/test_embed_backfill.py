"""Tests for claude_mem.embed_backfill (spec R2b).

Embedding repair + backfill: `chunks_vec` is populated for every chunk
that is missing a vector row, in batches with a per-batch commit
(resumable), and every embedding failure is logged to `ingestion_log`
(action='embed_fail', detail=reason) instead of being silently
swallowed.

Every test builds its own throwaway fixture DB via schema.init_db +
Ingester -- never touches the live .claude-mem/index.db. A hand-rolled
test-double embedder substitutes for Ollama (no real HTTP calls in
this file; the real service is exercised by the live backfill run,
not by the test suite).
"""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_mem.cli import cli
from claude_mem.embed_backfill import backfill
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _NullEmbedder:
    """Ingester-time embedder that always fails, so chunks land in the
    fixture DB with NO chunks_vec row -- exactly the state a stale
    live index is in (embed failures were silently swallowed)."""
    def embed(self, text: str) -> list[float]:
        raise RuntimeError("embedding disabled for fixture setup")


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


class _FlakyEmbedder:
    """Test-double Ollama client: raises a timeout-shaped error for
    content containing 'FAIL', otherwise returns a real vector. Models
    the read-timeout failure mode this task fixes (large chunk body vs
    a too-short read timeout)."""
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if "FAIL" in text:
            raise TimeoutError("ollama read timed out after 2.0s")
        return [0.2] * 1024


def _fixture_db_with_unembedded_chunks(tmp: str, contents: list[str]) -> Path:
    """Build a fixture DB with chunks inserted via a failing embedder,
    so chunks_vec starts empty -- mirrors live-DB state.

    Ingester.add() now logs every embed failure to ingestion_log
    (action='embed_fail') -- exactly the behavior this task adds. Using
    a failing embedder to seed the fixture therefore also seeds
    ingestion_log with setup noise unrelated to the backfill() call
    under test; clear it after fixture setup so each test's assertions
    on ingestion_log reflect only what backfill() itself did.
    """
    db = Path(tmp) / "index.db"
    init_db(db)
    ing = Ingester(db_path=db, embedder=_NullEmbedder())
    for c in contents:
        ing.add(Chunk(content=c, source="doc"))
    ing.close()
    conn = sqlite3.connect(db)
    try:
        conn.execute("DELETE FROM ingestion_log")
        conn.commit()
    finally:
        conn.close()
    return db


def _vec_count(db: Path) -> int:
    import sqlite_vec
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        return conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    finally:
        conn.close()


def _ingestion_log_rows(db: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT * FROM ingestion_log WHERE action = 'embed_fail'"
        ).fetchall()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Step 1: failing tests describing the desired backfill behavior
# ---------------------------------------------------------------------------

def test_success_path_inserts_into_chunks_vec():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(
            tmp, ["alpha content about apollo", "beta content about kmi"],
        )
        assert _vec_count(db) == 0

        summary = backfill(db, embedder=_ConstEmbedder())

        assert _vec_count(db) == 2
        assert summary["embedded"] == 2
        assert summary["failed"] == 0


def test_timeout_error_path_logs_and_continues():
    """A chunk whose embed() call raises must NOT abort the run: it is
    logged to ingestion_log (action='embed_fail') and the backfill
    continues to the next chunk."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(
            tmp, ["good content one", "FAIL this one times out",
                  "good content two"],
        )

        summary = backfill(db, embedder=_FlakyEmbedder())

        assert summary["embedded"] == 2
        assert summary["failed"] == 1
        assert _vec_count(db) == 2

        fail_rows = _ingestion_log_rows(db)
        assert len(fail_rows) == 1
        assert fail_rows[0]["action"] == "embed_fail"
        assert "timed out" in (fail_rows[0]["detail"] or "")


def test_resume_skips_already_embedded_ids():
    """A second backfill run against a DB that already has some
    chunks_vec rows must skip those ids -- only chunks still missing a
    vector are re-attempted."""
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(
            tmp, ["one", "two", "three"],
        )
        first = backfill(db, embedder=_ConstEmbedder())
        assert first["embedded"] == 3
        assert _vec_count(db) == 3

        embedder2 = _FlakyEmbedder()
        second = backfill(db, embedder=embedder2)

        # Nothing left to embed -- the resumed run must not re-call
        # embed() for ids already present in chunks_vec.
        assert second["embedded"] == 0
        assert second["skipped"] == 3
        assert embedder2.calls == []


def test_resume_after_partial_failure_only_retries_failed_ids():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(
            tmp, ["ok one", "FAIL two", "ok three"],
        )
        first = backfill(db, embedder=_FlakyEmbedder())
        assert first["embedded"] == 2
        assert first["failed"] == 1
        assert _vec_count(db) == 2

        # Second pass with a non-flaky embedder should pick up exactly
        # the one chunk still missing a vector.
        embedder2 = _ConstEmbedder()
        second = backfill(db, embedder=embedder2)
        assert second["embedded"] == 1
        assert second["skipped"] == 2
        assert _vec_count(db) == 3


def test_commits_per_chunk_not_per_batch():
    """backfill() commits after EVERY chunk, not on a batch_size
    cadence. Root cause this pins: Python's sqlite3 module opens an
    implicit transaction on the first DML statement and holds the
    write lock until commit(); embed() is a slow network call, so a
    batch-cadence commit would hold the write lock across every
    embed() call in the batch, starving concurrent readers for the
    whole batch (this is what happened during the real 2026-07-02
    backfill run against a contended Ollama and forced a kill + fix).
    Verified by aborting after the 1st of a batch_size=2 run (a
    NON-batch-boundary point) and confirming that one chunk survived
    the interruption -- if commits were batched, 0 would survive."""
    with tempfile.TemporaryDirectory() as tmp:
        contents = [f"chunk number {i}" for i in range(5)]
        db = _fixture_db_with_unembedded_chunks(tmp, contents)

        class _AbortAfterN:
            """Raises SystemExit (uncaught by backfill's per-chunk
            try/except, which only catches Exception) on the Nth call,
            simulating an interruption mid-run."""
            def __init__(self, abort_after: int) -> None:
                self.abort_after = abort_after
                self.calls = 0

            def embed(self, text: str) -> list[float]:
                self.calls += 1
                if self.calls > self.abort_after:
                    raise SystemExit("simulated interruption")
                return [0.3] * 1024

        with pytest.raises(SystemExit):
            backfill(db, embedder=_AbortAfterN(abort_after=1), batch_size=2)

        # Only 1 chunk succeeded before the interruption on chunk 2 (a
        # non-batch-boundary point since batch_size=2) -- per-chunk
        # commit means that 1 chunk is durable even though backfill()
        # never reached a batch boundary or its own final commit.
        assert _vec_count(db) == 1

        # A fresh backfill run resumes from where the interrupted one
        # left off -- the 1 already-embedded chunk is skipped, and the
        # remaining 4 are embedded to completion.
        resumed = backfill(db, embedder=_ConstEmbedder(), batch_size=2)
        assert resumed["skipped"] == 1
        assert resumed["embedded"] == 4
        assert _vec_count(db) == 5


def test_progress_cb_invoked_at_batch_cadence():
    """progress_cb(done, total) fires every batch_size chunks processed
    -- batch_size now gates progress reporting, not commit cadence."""
    with tempfile.TemporaryDirectory() as tmp:
        contents = [f"chunk number {i}" for i in range(5)]
        db = _fixture_db_with_unembedded_chunks(tmp, contents)

        calls: list[tuple[int, int]] = []
        backfill(
            db, embedder=_ConstEmbedder(), batch_size=2,
            progress_cb=lambda done, total: calls.append((done, total)),
        )
        # 5 chunks, batch_size=2 -> callback fires at done=2 and done=4
        # (done=5 does not land on a batch_size boundary, so no 3rd call).
        assert calls == [(2, 5), (4, 5)]


def test_dimension_mismatch_failure_logs_and_leaves_chunk_unvectored():
    """A backfill embedder that refuses a wrong-dimension fallback vector
    (EmbeddingError, the live-defect fix) must behave exactly like any
    other embed_fail: the chunk is left without a chunks_vec row and the
    failure is logged with a detail string naming the incompatibility,
    so a human/embed-backfill --resume can see and retry it instead of
    it silently stalling the run at a fixed offset (the live 8,602/15,157
    stall this task fixes)."""
    from claude_mem.embed import EmbeddingError

    class _DimMismatchEmbedder:
        def embed(self, text: str) -> list[float]:
            raise EmbeddingError(
                "fallback model dimension 768 != 1024 -- refusing "
                "incompatible vector"
            )

    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(tmp, ["alpha", "beta"])

        summary = backfill(db, embedder=_DimMismatchEmbedder())

        assert summary["embedded"] == 0
        assert summary["failed"] == 2
        assert _vec_count(db) == 0

        fail_rows = _ingestion_log_rows(db)
        assert len(fail_rows) == 2
        for row in fail_rows:
            assert "768" in row["detail"] and "1024" in row["detail"]


def test_failure_count_returned_in_summary():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(
            tmp, ["FAIL one", "FAIL two", "ok three"],
        )
        summary = backfill(db, embedder=_FlakyEmbedder())
        assert summary["embedded"] == 1
        assert summary["failed"] == 2
        assert summary["total"] == 3


def test_no_chunks_missing_vectors_is_a_noop():
    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(tmp, ["solo chunk"])
        backfill(db, embedder=_ConstEmbedder())
        summary = backfill(db, embedder=_ConstEmbedder())
        assert summary["embedded"] == 0
        assert summary["skipped"] == 1
        assert summary["total"] == 1


def test_backfill_connection_has_raised_busy_timeout():
    """backfill()'s connection must set PRAGMA busy_timeout well above
    SQLite's own default (5000ms, confirmed via `PRAGMA busy_timeout` on
    a fresh connection in this environment) so a slow-but-legitimate
    collision with a concurrent writer (a live hook's incremental
    ingest, another maintenance pass) waits instead of raising 'database
    is locked' on conn.commit() and killing the whole run. Root cause
    pinned here: a live 2026-07-02 embed-backfill --resume run crashed
    on its very first commit with exactly this OperationalError because
    this connection (unlike Ingester's, which explicitly raises it to
    30000ms) relied on the bare 5000ms default. This test holds a
    colliding write lock for 6s (longer than the 5000ms default, well
    under the 30000ms fix) so it FAILS pre-fix and PASSES post-fix."""
    import threading
    import time

    with tempfile.TemporaryDirectory() as tmp:
        db = _fixture_db_with_unembedded_chunks(tmp, ["alpha", "beta"])

        blocker_ready = threading.Event()
        release_blocker = threading.Event()

        def _hold_write_lock():
            conn = sqlite3.connect(db)
            conn.execute("PRAGMA busy_timeout = 30000")
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "INSERT INTO ingestion_log (timestamp, source_path, "
                "chunks_added) VALUES ('t', 'blocker', 0)"
            )
            blocker_ready.set()
            release_blocker.wait(timeout=10)
            conn.commit()
            conn.close()

        t = threading.Thread(target=_hold_write_lock)
        t.start()
        blocker_ready.wait(timeout=5)

        def _release_after_delay():
            time.sleep(6.0)
            release_blocker.set()

        threading.Thread(target=_release_after_delay).start()

        # backfill() must wait out the ~6s collision rather than crash.
        summary = backfill(db, embedder=_ConstEmbedder())
        t.join(timeout=10)

        assert summary["embedded"] == 2
        assert summary["failed"] == 0


def test_missing_vector_query_scales_past_join_pathology():
    """Root cause pinned here (live 2026-07-02 recon): the original
    `_missing_vector_chunks` used a `LEFT JOIN chunks_vec v ON
    v.chunk_id = c.id` -- correct, but a vec0 auxiliary column has no
    supporting index for equality joins, so SQLite fell back to
    scanning the ENTIRE chunks_vec virtual table for every row of
    chunks (confirmed via `EXPLAIN QUERY PLAN`: `SCAN v VIRTUAL TABLE
    ... LEFT-JOIN`). At the live DB's scale (~15k chunks, ~8.6k
    vectored) this made a fresh embed-backfill --resume run appear to
    hang indefinitely -- CPU-pegged, zero network I/O, zero commits --
    before a single embed() call ever fired. This test uses a fixture
    large enough (2,000 vectored + 500 missing) that an O(n*m) join
    reintroduced here would make the suite itself visibly slow; the
    O(n+m) set-difference fix keeps it fast. Primarily a correctness
    pin (right ids returned) with the scale as the regression guard."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        vectored_ids = set()
        for i in range(2000):
            chunk = Chunk(content=f"vectored chunk number {i}", source="doc")
            ing.add(chunk)
            vectored_ids.add(chunk.id)
        ing.close()

        ing2 = Ingester(db_path=db, embedder=_NullEmbedder())
        missing_ids = set()
        for i in range(500):
            chunk = Chunk(content=f"missing chunk number {i}", source="doc")
            ing2.add(chunk)
            missing_ids.add(chunk.id)
        ing2.close()

        import time
        from claude_mem.embed_backfill import _missing_vector_chunks

        conn = sqlite3.connect(db)
        conn.enable_load_extension(True)
        import sqlite_vec as _sv
        _sv.load(conn)
        t0 = time.time()
        result = _missing_vector_chunks(conn)
        elapsed = time.time() - t0
        conn.close()

        result_ids = {r[0] for r in result}
        assert result_ids == missing_ids
        assert result_ids.isdisjoint(vectored_ids)
        # Generous ceiling: the O(n+m) fix runs in well under a second at
        # this fixture size; an O(n*m) regression would take far longer.
        assert elapsed < 5.0, (
            f"_missing_vector_chunks took {elapsed:.2f}s for a 2500-chunk "
            "fixture -- likely regressed to the O(n*m) virtual-table-scan "
            "join this test guards against"
        )


def test_missing_db_raises_runtime_error():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "does_not_exist.db"
        with pytest.raises(RuntimeError):
            backfill(missing, embedder=_ConstEmbedder())


# ---------------------------------------------------------------------------
# CLI-level smoke tests (`claude-mem embed-backfill`)
# ---------------------------------------------------------------------------

def test_cli_embed_backfill_reports_coverage(monkeypatch):
    """`claude-mem embed-backfill` embeds pending chunks and reports a
    total/embedded/failed/skipped summary line."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        cfg_db = Path(tmp) / ".claude-mem" / "index.db"

        # Seed chunks with a failing embedder so chunks_vec starts empty,
        # matching live-DB state before this task's fix.
        ing = Ingester(db_path=cfg_db, embedder=_NullEmbedder())
        ing.add(Chunk(content="alpha doc content", source="doc"))
        ing.add(Chunk(content="beta doc content", source="doc"))
        ing.close()
        conn = sqlite3.connect(cfg_db)
        conn.execute("DELETE FROM ingestion_log")
        conn.commit()
        conn.close()

        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["embed-backfill", "--project-root", tmp, "--batch", "1"],
        )
        assert result.exit_code == 0, result.output
        assert "2 embedded" in result.output
        assert "0 failed" in result.output
        assert _vec_count(cfg_db) == 2


def test_cli_embed_backfill_resume_flag_accepted(monkeypatch):
    """--resume is accepted and does not change behavior -- a resumed
    run against an already-fully-embedded DB embeds 0 and skips all."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        cfg_db = Path(tmp) / ".claude-mem" / "index.db"
        ing = Ingester(db_path=cfg_db, embedder=_ConstEmbedder())
        ing.add(Chunk(content="already embedded chunk", source="doc"))
        ing.close()

        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["embed-backfill", "--project-root", tmp, "--resume"],
        )
        assert result.exit_code == 0, result.output
        assert "0 embedded" in result.output
        assert "1 already had vectors" in result.output


def test_cli_embed_backfill_missing_index_errors(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["embed-backfill", "--project-root", tmp],
        )
        assert result.exit_code != 0
        assert "no index" in result.output.lower()
