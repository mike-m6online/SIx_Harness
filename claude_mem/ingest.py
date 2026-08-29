"""Chunk ingestion: dedup by content hash, FTS5 insert, vector insert.

Content-hash dedup is cannibalized from agentmemory src/functions/dedup.ts
(SHA256 over normalized content). Identical chunks are skipped silently.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Protocol

import sqlite_vec

from claude_mem.schema import init_db


class _Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


@dataclass
class Chunk:
    content: str
    source: str
    role: Optional[str] = None
    session_id: Optional[str] = None
    file_path: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    date: Optional[str] = None
    module: Optional[str] = None
    status: Optional[str] = None
    aliases: str = ""
    do_not_rebuild: bool = False
    signal_weight: int = 10
    is_correction: bool = False
    is_decision: bool = False
    file_mtime: Optional[float] = None
    id: str = field(default="")

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _hash_content(self.content)


def _hash_content(content: str) -> str:
    normalized = " ".join(content.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass
class Ingester:
    db_path: Path
    embedder: _Embedder
    cross_project_cfg: Optional[object] = None  # ProjectConfig or None
    _conn: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        # Idempotent schema migration: this task adds the
        # ingestion_log.action/detail columns via schema.py's
        # _ensure_columns, applied only inside init_db(). A DB created by
        # `claude-mem init` before this column existed would otherwise
        # crash on the first embed_fail log write. init_db() is all
        # CREATE-IF-NOT-EXISTS + additive ALTER TABLE, so re-running it on
        # every Ingester construction is always safe and cheap.
        init_db(self.db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        # Belt-and-braces for any remaining short-lock contention (the
        # long-lock case -- embed() held open across the write -- is
        # eliminated below by embedding BEFORE the transaction opens, not
        # by this timeout). Live DB default is journal_mode=delete with
        # busy_timeout=5000; a concurrent writer (embed-backfill, another
        # incremental run) can still collide on the brief local-write
        # transaction itself, so raise the wait ceiling well above the
        # default rather than fail fast with SQLITE_BUSY.
        self._conn.execute("PRAGMA busy_timeout = 30000")

    def add(self, chunk: Chunk) -> bool:
        """Insert chunk; returns True if newly inserted, False if duplicate.

        Ordering is load-bearing: embed() is a network call (up to
        BULK_READ_TIMEOUT_S = 60s against Ollama) that must run BEFORE any
        SQLite write transaction opens. Python's sqlite3 module opens an
        implicit transaction on the first DML statement (default
        isolation_level) and holds the write lock until commit() -- if the
        INSERT INTO chunks ran first, that lock would sit open for the
        entire embed() round-trip, starving every concurrent writer (the
        SessionEnd incremental path now runs concurrently with a live
        embed-backfill) past the DB's busy_timeout. See
        embed_backfill.py's module docstring for the sibling root-cause
        account of the same class of bug on the backfill side. embed()
        failure semantics are unchanged: the chunk is still inserted
        (without a vector) and the failure is logged to ingestion_log so
        embed-backfill can retry it later -- only the ORDER moved, not the
        behavior.
        """
        existing = self._conn.execute(
            "SELECT 1 FROM chunks WHERE id = ?", (chunk.id,)
        ).fetchone()
        if existing:
            return False

        embedding: Optional[List[float]] = None
        embed_error: Optional[Exception] = None
        try:
            embedding = self.embedder.embed(chunk.content)
        except Exception as exc:  # noqa: BLE001 -- logged below, never silent
            embed_error = exc

        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            """
            INSERT INTO chunks (
                id, content, source, role, session_id, file_path,
                line_start, line_end, date, module, status, aliases,
                do_not_rebuild, signal_weight, is_correction, is_decision,
                ingested_at, file_mtime
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chunk.id, chunk.content, chunk.source, chunk.role,
                chunk.session_id, chunk.file_path, chunk.line_start,
                chunk.line_end, chunk.date, chunk.module, chunk.status,
                chunk.aliases,
                int(chunk.do_not_rebuild), chunk.signal_weight,
                int(chunk.is_correction), int(chunk.is_decision),
                now, chunk.file_mtime,
            ),
        )
        if embedding is not None:
            self._conn.execute(
                "INSERT INTO chunks_vec (embedding, chunk_id) VALUES (?, ?)",
                (sqlite_vec.serialize_float32(embedding), chunk.id),
            )
        else:
            # Vector failure is non-fatal to ingestion: BM25 still works
            # without it. It is NOT silent, though -- a bare `except: pass`
            # here is exactly what let chunks_vec sit at 0 rows through two
            # prior "fixes" with zero operator-visible trace (see embed.py
            # module docstring for the full root-cause account). Every
            # failure is logged to ingestion_log so embed-backfill (and a
            # human reading the table) can see it and retry.
            self._conn.execute(
                """
                INSERT INTO ingestion_log
                    (timestamp, source_path, chunks_added, file_mtime,
                     chunk_ids, action, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(timezone.utc).isoformat(),
                    chunk.id, 0, None, json.dumps([chunk.id]),
                    "embed_fail", str(embed_error)[:2000],
                ),
            )
        self._conn.commit()
        # Mirror to global cross-project index when caller supplied
        # the project config + isolate_from_cross_project is False.
        if self.cross_project_cfg is not None:
            try:
                from claude_mem.cross_project import mirror_chunk_to_global
                mirror_chunk_to_global(chunk, self.cross_project_cfg)
            except Exception:
                pass
        return True

    def record_ingestion(
        self,
        source_path: str,
        chunks_added: int,
        file_mtime: Optional[float] = None,
        chunk_ids: Optional[List[str]] = None,
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO ingestion_log
                (timestamp, source_path, chunks_added, file_mtime, chunk_ids)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                source_path, chunks_added, file_mtime,
                json.dumps(chunk_ids or []),
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
