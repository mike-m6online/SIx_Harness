"""Embedding repair + backfill (spec R2b).

Populates chunks_vec for every chunk currently missing a vector row.
Root cause of the pre-existing 0-row chunks_vec state: see the module
docstring in claude_mem.embed (read_timeout_s=2.0 vs num_ctx=8192 vs a
silently-swallowed exception in ingest.py -- fixed alongside this
module).

Design:
  - Resumable: a chunk is "missing a vector" iff its id is absent from
    chunks_vec.chunk_id (LEFT JOIN ... WHERE chunk_id IS NULL). A
    previously-embedded chunk is never re-embedded, and a
    previously-FAILED chunk (still absent from chunks_vec) is retried
    on the next run -- no separate "give up" state.
  - Commit after every chunk (success OR logged failure), not on a
    batch cadence. Root-cause note from the live 2026-07 run: Python's
    sqlite3 module opens an IMPLICIT transaction on the first DML
    statement of a connection (default isolation_level) and holds the
    write lock until commit(). embed()'s network round-trip against a
    contended Ollama daemon (a concurrent generation-model call sharing
    the GPU) can take 60-180+ seconds; committing only every N chunks
    means the write lock sits open for the ENTIRE batch's worth of slow
    network calls, starving every concurrent reader (a human checking
    `SELECT COUNT(*) FROM chunks_vec` mid-run gets "database is
    locked"). A first implementation that batched the commit was caught
    exactly this way during the real backfill and had to be killed and
    fixed. Per-chunk commit keeps the write lock held only for the
    local SQLite write itself (sub-millisecond), never across a network
    call -- strictly more resumable than batching (a kill loses at most
    the in-flight chunk, not up to `batch_size` of them) and keeps the
    DB observable by concurrent readers throughout a multi-hour run.
    `batch_size` is retained in the signature/CLI for interface
    compatibility and gates periodic progress reporting, not commit
    cadence.
  - Every embed() failure is logged to ingestion_log (action=
    'embed_fail', detail=str(exc)) and counted in the summary -- never
    silently dropped (that silence is the exact bug this task fixes).
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Protocol

import sqlite_vec

from claude_mem.schema import init_db


class _Embedder(Protocol):
    def embed(self, text: str) -> list: ...


def _missing_vector_chunks(conn: sqlite3.Connection) -> list:
    """Return [(id, content)] for every chunk with no chunks_vec row.

    Root cause fixed here (live 2026-07-02 recon, second stall found
    while relaunching the backfill after the dimension-mismatch fix): a
    plain `LEFT JOIN chunks_vec v ON v.chunk_id = c.id` is CORRECT (the
    vec0 auxiliary column `+chunk_id TEXT` is directly queryable) but
    catastrophically slow at this table's scale. `EXPLAIN QUERY PLAN`
    shows `SCAN v VIRTUAL TABLE ... LEFT-JOIN` -- a vec0 auxiliary
    column has no supporting index for equality joins, so SQLite probes
    `chunks_vec` with a full virtual-table scan for EVERY row of
    `chunks` (O(n*m): ~15,157 * ~8,602 =~ 130M virtual-table row
    accesses through the sqlite-vec C extension). Measured live: still
    running after minutes with the backfill process pegged on CPU and
    zero network I/O (no embed() calls had even started) -- looked
    identical to a hang from the outside.

    Fix: scan each table ONCE (a `vec0` table has no index needed to
    enumerate all of its own rows) and compute the set difference in
    Python -- O(n+m). Measured live on the same DB: chunks_vec scan
    0.03s, chunks scan 0.05s, Python filter 0.01s (~0.1s total, vs.
    unbounded minutes for the join).
    """
    vectored_ids = {
        row[0] for row in conn.execute("SELECT chunk_id FROM chunks_vec")
    }
    all_chunks = conn.execute(
        "SELECT id, content FROM chunks ORDER BY rowid"
    ).fetchall()
    return [
        (chunk_id, content)
        for chunk_id, content in all_chunks
        if chunk_id not in vectored_ids
    ]


def backfill(
    db_path: Path,
    *,
    embedder: _Embedder,
    batch_size: int = 200,
    progress_cb: Optional[Any] = None,
) -> Dict[str, Any]:
    """Embed every chunk missing a chunks_vec row.

    Returns a summary dict: {total, embedded, failed, skipped}.
    `skipped` counts chunks that already had a vector at start (i.e.
    were not candidates this run) -- reported for resume visibility.

    Raises RuntimeError if db_path does not exist (mirrors
    migrate_regrade's refuse-to-run-against-nothing behavior).

    Runs the idempotent schema migration (init_db) before touching the
    DB: this task adds the ingestion_log.action/detail columns via
    schema.py's _ensure_columns, and init_db is the ONLY code path in
    this package that applies that migration. A live DB that predates
    this task (created by `claude-mem init` before this column existed)
    would otherwise fail with "table ingestion_log has no column named
    action" the first time a failure needs logging -- init_db() is
    all-CREATE-IF-NOT-EXISTS + additive ALTER TABLE, so re-running it
    here is always safe and cheap.

    Commits after EVERY chunk (see module docstring for why: holding
    the write lock across a slow embed() network call starves
    concurrent readers for the whole batch). `batch_size` gates
    `progress_cb` calls (invoked with (done, total) every batch_size
    chunks processed) for long-running CLI progress reporting; it does
    NOT change commit or resumability granularity, which is always
    per-chunk.
    """
    if not db_path.is_file():
        raise RuntimeError(f"embed_backfill: no database at {db_path}")

    init_db(db_path)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    # Same belt-and-braces fix as Ingester.__post_init__ (ingest.py): a
    # concurrent writer (a live hook's incremental ingest, another
    # backfill/maintenance pass) can collide on this connection's brief
    # per-chunk write transaction. Without a raised busy_timeout, SQLite's
    # default (busy_timeout=5000 at best, 0 if never set on this
    # connection) surfaces as `sqlite3.OperationalError: database is
    # locked` on conn.commit() and kills the whole backfill run instead of
    # waiting the ~sub-second a colliding local write actually holds the
    # lock for. Observed live 2026-07-02: a fresh `embed-backfill --resume`
    # run crashed on its very first commit for exactly this reason.
    conn.execute("PRAGMA busy_timeout = 30000")

    total_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    candidates = _missing_vector_chunks(conn)
    already_embedded = total_chunks - len(candidates)

    embedded = 0
    failed = 0
    processed = 0

    try:
        for chunk_id, content in candidates:
            try:
                vec = embedder.embed(content)
                conn.execute(
                    "INSERT INTO chunks_vec (embedding, chunk_id) VALUES (?, ?)",
                    (sqlite_vec.serialize_float32(vec), chunk_id),
                )
                embedded += 1
            except Exception as exc:
                conn.execute(
                    """
                    INSERT INTO ingestion_log
                        (timestamp, source_path, chunks_added, file_mtime,
                         chunk_ids, action, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now(timezone.utc).isoformat(),
                        chunk_id, 0, None, json.dumps([chunk_id]),
                        "embed_fail", str(exc)[:2000],
                    ),
                )
                failed += 1
            # Commit immediately: never hold the write lock across the
            # NEXT chunk's slow network call (see module docstring).
            conn.commit()
            processed += 1
            if progress_cb is not None and processed % batch_size == 0:
                progress_cb(processed, len(candidates))
    finally:
        conn.close()

    return {
        "total": total_chunks,
        "embedded": embedded,
        "failed": failed,
        "skipped": already_embedded,
    }
