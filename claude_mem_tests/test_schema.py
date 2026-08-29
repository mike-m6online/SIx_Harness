import sqlite3
import tempfile
from pathlib import Path

import sqlite_vec

from claude_mem.schema import init_db


def test_init_db_creates_all_tables():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "index.db"
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' OR type='view'"
                )
            }
        finally:
            conn.close()
        # FTS5 virtual table chunks_fts implies presence of related shadow tables
        expected = {
            "chunks", "chunks_fts", "chunks_vec", "ingestion_log",
            # Anti-recurrence Rung 2 capture tables.
            "threads", "decisions", "dead_ends",
        }
        missing = expected - tables
        assert not missing, f"missing tables: {missing}"


def test_fresh_db_has_aliases_columns():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "index.db"
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            chunk_cols = {row[1] for row in conn.execute("PRAGMA table_info(chunks)")}
            assert "aliases" in chunk_cols
            fts_cols = [
                d[0]
                for d in conn.execute("SELECT * FROM chunks_fts LIMIT 0").description
            ]
            assert "aliases" in fts_cols
        finally:
            conn.close()


def test_alias_fts_match_finds_chunk():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "index.db"
        init_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO chunks (id, content, source, aliases, ingested_at) "
                "VALUES ('a','the CWM kernel row','doc','causal world model','t')"
            )
            conn.commit()
            # chunks_fts is external-content; resolve its rowid back to chunks.id.
            # The match comes from the aliases column, not the content.
            row = conn.execute(
                "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
                "WHERE chunks_fts MATCH 'causal'"
            ).fetchone()
            assert row is not None
            assert row[0] == "a"
        finally:
            conn.close()


def test_idempotent_second_init():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = Path(tmp) / "index.db"
        init_db(db_path)
        init_db(db_path)  # second init must not error
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO chunks (id, content, source, aliases, ingested_at) "
                "VALUES ('a','the CWM kernel row','doc','causal world model','t')"
            )
            conn.commit()
            row = conn.execute(
                "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
                "WHERE chunks_fts MATCH 'causal'"
            ).fetchone()
            assert row is not None
            assert row[0] == "a"
        finally:
            conn.close()


def test_migration_from_old_schema(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.executescript('''
    CREATE TABLE chunks (id TEXT PRIMARY KEY, content TEXT NOT NULL, source TEXT NOT NULL,
      role TEXT, session_id TEXT, file_path TEXT, line_start INTEGER, line_end INTEGER,
      date TEXT, module TEXT, status TEXT, do_not_rebuild INTEGER DEFAULT 0,
      signal_weight INTEGER DEFAULT 10, is_correction INTEGER DEFAULT 0, is_decision INTEGER DEFAULT 0,
      access_count INTEGER DEFAULT 0, last_accessed TEXT, ingested_at TEXT NOT NULL, file_mtime REAL);
    CREATE VIRTUAL TABLE chunks_fts USING fts5(content, source, role, module, status,
      content='chunks', content_rowid='rowid', tokenize='porter');
    CREATE TRIGGER chunks_ai AFTER INSERT ON chunks BEGIN
      INSERT INTO chunks_fts(rowid, content, source, role, module, status)
      VALUES (new.rowid, new.content, new.source, new.role, new.module, new.status); END;
    ''')
    conn.execute("INSERT INTO chunks (id, content, source, ingested_at) VALUES ('x','plain content here','doc','t')")
    conn.commit()
    conn.close()

    init_db(db)  # runs the migration

    conn = sqlite3.connect(db)
    cols = [d[0] for d in conn.execute("SELECT * FROM chunks_fts LIMIT 0").description]
    assert "aliases" in cols
    # pre-existing row survived the rebuild and is still content-searchable
    # (chunks_fts is external-content; resolve its rowid back to chunks.id).
    assert conn.execute(
        "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
        "WHERE chunks_fts MATCH 'plain'"
    ).fetchone() is not None
    # and the new aliases path works after an UPDATE (the AU trigger syncs FTS)
    conn.execute("UPDATE chunks SET aliases='causal world model' WHERE id='x'")
    conn.commit()
    assert conn.execute(
        "SELECT c.id FROM chunks_fts f JOIN chunks c ON c.rowid = f.rowid "
        "WHERE chunks_fts MATCH 'causal'"
    ).fetchone() is not None
    conn.close()
