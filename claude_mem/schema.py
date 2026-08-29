"""SQLite schema for the per-project claude-mem index.

Four storage surfaces:
  chunks        -- canonical chunk metadata + content
  chunks_fts    -- FTS5 BM25 index (k1=1.2 b=0.75 are FTS5 defaults; same
                   defaults agentmemory uses)
  chunks_vec    -- sqlite-vec virtual table for Qwen3-Embedding embeddings
  ingestion_log -- audit trail of what got indexed when (for canonical refresh)
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import sqlite_vec

_CHUNK_TABLE = """
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    role TEXT,
    session_id TEXT,
    file_path TEXT,
    line_start INTEGER,
    line_end INTEGER,
    date TEXT,
    module TEXT,
    status TEXT,
    aliases TEXT,
    do_not_rebuild INTEGER DEFAULT 0,
    signal_weight INTEGER DEFAULT 10,
    is_correction INTEGER DEFAULT 0,
    is_decision INTEGER DEFAULT 0,
    access_count INTEGER DEFAULT 0,
    last_accessed TEXT,
    ingested_at TEXT NOT NULL,
    file_mtime REAL
);
"""

_FTS_TABLE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content, source, role, module, status, aliases,
    content='chunks', content_rowid='rowid',
    tokenize='porter'
);
"""

_FTS_TRIGGERS = [
    """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, content, source, role, module, status, aliases)
        VALUES (new.rowid, new.content, new.source, new.role, new.module, new.status, new.aliases);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, source, role, module, status, aliases)
        VALUES ('delete', old.rowid, old.content, old.source, old.role, old.module, old.status, old.aliases);
    END;""",
    """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, content, source, role, module, status, aliases)
        VALUES ('delete', old.rowid, old.content, old.source, old.role, old.module, old.status, old.aliases);
        INSERT INTO chunks_fts(rowid, content, source, role, module, status, aliases)
        VALUES (new.rowid, new.content, new.source, new.role, new.module, new.status, new.aliases);
    END;""",
]

_VEC_TABLE_TEMPLATE = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    embedding float[{dim}],
    +chunk_id TEXT
);
"""

_INGESTION_LOG = """
CREATE TABLE IF NOT EXISTS ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source_path TEXT NOT NULL,
    chunks_added INTEGER NOT NULL,
    file_mtime REAL,
    chunk_ids TEXT
);
"""

# Qwen3-Embedding-0.6B produces 1024-dim embeddings; nomic-embed-text-v1.5
# is 768-dim. Set the default to 1024 to match the locked primary model.
# Override via init_db(..., embedding_dim=768) when running with the
# nomic fallback.
DEFAULT_EMBEDDING_DIM = 1024


def _ensure_columns(conn, table: str, columns: dict) -> None:
    """Idempotently ADD COLUMN for any missing column (SQLite lacks
    ADD COLUMN IF NOT EXISTS). columns maps name -> SQL type declaration."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    for name, decl in columns.items():
        if name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def _fts_has_aliases(conn) -> bool:
    cur = conn.execute("SELECT * FROM chunks_fts LIMIT 0")
    return "aliases" in [d[0] for d in cur.description]


def _migrate_fts_aliases(conn) -> None:
    """Add the `aliases` column to the external-content FTS index for an
    existing DB by recreating chunks_fts + its triggers and rebuilding from
    chunks. Idempotent: a no-op once the column is present."""
    if _fts_has_aliases(conn):
        return
    for trig in ("chunks_ai", "chunks_ad", "chunks_au"):
        conn.execute(f"DROP TRIGGER IF EXISTS {trig}")
    conn.execute("DROP TABLE IF EXISTS chunks_fts")
    conn.execute(_FTS_TABLE)
    for trig in _FTS_TRIGGERS:
        conn.execute(trig)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")


def init_db(db_path: Path, *, embedding_dim: int = DEFAULT_EMBEDDING_DIM) -> None:
    """Create the schema in db_path. Idempotent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.execute(_CHUNK_TABLE)
        conn.execute(_FTS_TABLE)
        for trig in _FTS_TRIGGERS:
            conn.execute(trig)
        conn.execute(_VEC_TABLE_TEMPLATE.format(dim=embedding_dim))
        conn.execute(_INGESTION_LOG)
        # Anti-recurrence Rung 2: structured capture (threads / decisions / dead-ends).
        # Plain rows -- not FTS/vector indexed; surfaced by gen_decisions_state and
        # the Rung 3 synthesis path, not by chunk search.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                opened_date TEXT,
                state TEXT DEFAULT 'open',
                last_updated TEXT,
                summary TEXT,
                lineage_text TEXT,
                lineage_cached_at TEXT,
                lineage_cache_key TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS decisions (
                id TEXT PRIMARY KEY,
                date TEXT,
                title TEXT NOT NULL,
                rationale TEXT,
                options_rejected TEXT,
                state TEXT DEFAULT 'pending',
                thread_id TEXT,
                linked_commits TEXT,
                mike_approved INTEGER DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS dead_ends (
                id TEXT PRIMARY KEY,
                date TEXT,
                approach TEXT NOT NULL,
                why_shelved TEXT,
                superseded_by TEXT,
                thread_id TEXT,
                state TEXT DEFAULT 'pending'
            )
            """
        )
        _ensure_columns(conn, "threads", {
            "lineage_text": "TEXT",
            "lineage_cached_at": "TEXT",
            "lineage_cache_key": "TEXT",
        })
        conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        _ensure_columns(conn, "chunks", {"aliases": "TEXT"})
        _migrate_fts_aliases(conn)
        # R2b (embed-backfill): failure-visibility columns on ingestion_log.
        # action distinguishes a normal ingest row (NULL/'ingest') from an
        # embed-repair failure row ('embed_fail'); detail carries the
        # exception text. Nullable + additive so existing rows/readers are
        # unaffected.
        _ensure_columns(conn, "ingestion_log", {
            "action": "TEXT",
            "detail": "TEXT",
        })
        conn.commit()
    finally:
        conn.close()
