"""Cross-project shared index + privacy-filtered search.

Mike-locked 2026-05-24: ON by default. All chunks from every project
flow into the global index at ~/.claude-mem/cross-project-index.db
tagged with their origin project. Useful when working on Project B
and asking "didn't I solve this in Project A?".

Per-project opt-out: set isolate_from_cross_project: true in
.claude-mem/config.yaml to keep a project's chunks out of the global
index. Already-indexed chunks stay; new ones do not mirror.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Protocol

import sqlite_vec

from claude_mem.schema import init_db
from claude_mem.search import Searcher


class _Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


def cross_project_db_path() -> Path:
    """Resolve the global cross-project index path. Honors
    CLAUDE_MEM_HOME env var override (used in tests + container setups);
    defaults to ~/.claude-mem/."""
    home = os.environ.get("CLAUDE_MEM_HOME")
    base = Path(home) if home else Path.home() / ".claude-mem"
    base.mkdir(parents=True, exist_ok=True)
    return base / "cross-project-index.db"


def _ensure_cross_project_schema(db_path: Path, embedding_dim: int) -> None:
    """The cross-project index reuses the per-project schema + adds an
    origin_project column to chunks."""
    init_db(db_path, embedding_dim=embedding_dim)
    conn = sqlite3.connect(db_path)
    try:
        # ALTER TABLE ADD COLUMN is idempotent-friendly via try/except
        try:
            conn.execute("ALTER TABLE chunks ADD COLUMN origin_project TEXT")
        except sqlite3.OperationalError:
            pass  # column already exists
        conn.commit()
    finally:
        conn.close()


def mirror_chunk_to_global(chunk, cfg) -> None:
    """Mirror a chunk into the cross-project index unless the project
    is isolate_from_cross_project=true. Tags origin_project as the
    project root's directory name."""
    if cfg.values.get("isolate_from_cross_project"):
        return
    db_path = cross_project_db_path()
    _ensure_cross_project_schema(
        db_path, cfg.values.get("embedding_dim", 1024),
    )
    origin = cfg.project_root.name
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        existing = conn.execute(
            "SELECT 1 FROM chunks WHERE id = ?", (chunk.id,)
        ).fetchone()
        if existing:
            return
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """
            INSERT INTO chunks (
                id, content, source, role, session_id, file_path,
                line_start, line_end, date, module, status,
                do_not_rebuild, signal_weight, is_correction, is_decision,
                ingested_at, file_mtime, origin_project
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                chunk.id, chunk.content, chunk.source, chunk.role,
                chunk.session_id, chunk.file_path, chunk.line_start,
                chunk.line_end, chunk.date, chunk.module, chunk.status,
                int(chunk.do_not_rebuild), chunk.signal_weight,
                int(chunk.is_correction), int(chunk.is_decision),
                now, chunk.file_mtime, origin,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def search_cross_project(
    query: str,
    embedder: _Embedder,
    *,
    top_k: int = 10,
    filter_do_not_rebuild: bool = False,
    filter_is_correction: bool = False,
) -> List[Dict[str, Any]]:
    """Search the global cross-project index. Returns rows tagged with
    origin_project so consumers can show "from project: X" beside each
    hit. Returns [] when the global index does not yet exist."""
    db_path = cross_project_db_path()
    if not db_path.is_file():
        return []
    searcher = Searcher(db_path=db_path, embedder=embedder)
    try:
        results = searcher.search(
            query, top_k=top_k,
            filter_do_not_rebuild=filter_do_not_rebuild,
            filter_is_correction=filter_is_correction,
        )
    finally:
        searcher.close()
    return results
