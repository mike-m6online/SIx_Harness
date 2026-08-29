"""Tests for claude_mem.maintenance.find_prune_candidates.

A never-surfaced (access_count=0) chunk older than --min-age-days is a
prune candidate; a recently-accessed one is not, and a young never-
surfaced one is not yet. REPORT ONLY -- nothing is deleted.
"""
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from claude_mem.ingest import Chunk, Ingester
from claude_mem.maintenance import find_prune_candidates
from claude_mem.schema import init_db


class _ConstEmbedder:
    def __init__(self, value: float = 0.1, dim: int = 1024) -> None:
        self.value = value
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        return [self.value] * self.dim


def _embedder():
    return _ConstEmbedder()


def _set_ingested_at(db: Path, content_substr: str, iso: str) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE chunks SET ingested_at = ? WHERE content LIKE ?",
            (iso, f"%{content_substr}%"),
        )
        conn.commit()
    finally:
        conn.close()


def _set_access(db: Path, content_substr: str, count: int, last: str) -> None:
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "UPDATE chunks SET access_count = ?, last_accessed = ? "
            "WHERE content LIKE ?",
            (count, last, f"%{content_substr}%"),
        )
        conn.commit()
    finally:
        conn.close()


def test_old_never_accessed_chunk_is_a_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="stale never-surfaced memory note", source="memory"))
        ing.close()
        old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        _set_ingested_at(db, "stale never-surfaced", old)

        cands = find_prune_candidates(db, min_age_days=30, limit=50)
        assert len(cands) == 1
        assert "stale never-surfaced" in cands[0]["content"]
        assert cands[0]["age_days"] >= 90


def test_recently_accessed_chunk_is_not_a_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="old but recently surfaced note", source="memory"))
        ing.close()
        old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat()
        _set_ingested_at(db, "old but recently surfaced", old)
        # It WAS surfaced (access_count>0) -> excluded even though old.
        _set_access(
            db, "old but recently surfaced", 4,
            datetime.now(timezone.utc).isoformat(),
        )

        cands = find_prune_candidates(db, min_age_days=30, limit=50)
        assert cands == []


def test_young_never_accessed_chunk_is_not_yet_a_candidate():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="fresh never-surfaced note", source="memory"))
        ing.close()
        recent = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()
        _set_ingested_at(db, "fresh never-surfaced", recent)

        cands = find_prune_candidates(db, min_age_days=30, limit=50)
        assert cands == []


def test_candidates_ordered_oldest_first_and_limited():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="candidate aaa older", source="memory"))
        ing.add(Chunk(content="candidate bbb newer", source="memory"))
        ing.close()
        older = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        newer = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        _set_ingested_at(db, "candidate aaa older", older)
        _set_ingested_at(db, "candidate bbb newer", newer)

        all_c = find_prune_candidates(db, min_age_days=30, limit=50)
        assert [c["content"][:13] for c in all_c] == [
            "candidate aaa", "candidate bbb",
        ]
        limited = find_prune_candidates(db, min_age_days=30, limit=1)
        assert len(limited) == 1
        assert "candidate aaa" in limited[0]["content"]


def test_injected_now_makes_age_deterministic():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_embedder())
        ing.add(Chunk(content="age-check note", source="memory"))
        ing.close()
        _set_ingested_at(db, "age-check note", "2026-01-01T00:00:00+00:00")
        fixed_now = datetime(2026, 3, 2, tzinfo=timezone.utc)
        cands = find_prune_candidates(
            db, min_age_days=30, limit=50, now=fixed_now,
        )
        assert len(cands) == 1
        assert cands[0]["age_days"] == 60
