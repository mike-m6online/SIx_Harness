"""Hybrid BM25 + vector search with reciprocal-rank-fusion reranking.

Algorithm choices cannibalized from agentmemory src/state/hybrid-search.ts:
  - BM25 via SQLite FTS5 (k1=1.2 b=0.75 are the FTS5 defaults; same as
    agentmemory).
  - Vector via sqlite-vec cosine similarity.
  - Fusion via RRF: score = 1/(k+rank_bm25) + 1/(k+rank_vec), k=60
    (Cormack-2009 default; the value agentmemory uses).
  - Signal-weight boost: final = fusion * (1 + signal_weight / 100).
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set

import sqlite_vec

from claude_mem.telemetry import record_embed_degradation


class _Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


# Telemetry DB filename convention (matches config.ProjectConfig: index.db
# and telemetry.db are siblings inside .claude-mem/). Searcher derives its
# telemetry path from db_path when no explicit one is given, so every
# existing construction site gets degradation telemetry without churn.
_TELEMETRY_DB_NAME = "telemetry.db"

# Process-lifetime log-once guard for vector-leg degradation (2026-08-19
# embed-resilience fix). A dead/cold Ollama fails EVERY embed in a hook
# process that may run several searches (prompt_submit runs three); one
# telemetry row per process per telemetry DB is signal, N identical rows
# are noise. Keyed by resolved telemetry path so a process that searches
# two projects records the degradation in each project's telemetry.
_EMBED_DEGRADATION_LOGGED: Set[str] = set()

# Process-lifetime embed circuit breaker (2026-08-25 shakedown fix),
# keyed by embedder endpoint. Once embed fails in this process, later
# searches skip the embed attempt entirely instead of re-burning the
# timeout chain: a hook runs several searches inside a 5s budget, a
# down/cold Ollama cannot recover between them, and the observed cost
# of retrying was an 8.1s prompt_submit that blew its timeout and lost
# the turn's injections (telemetry 2026-08-23 00:35Z). Hook processes
# are short-lived, so the next invocation retries fresh — recovery is
# automatic at the next prompt, never within the doomed window.
_EMBED_CIRCUIT_OPEN: Set[str] = set()


def _embed_circuit_key(embedder: Any) -> str:
    """Circuit key: the embedder's endpoint (test doubles without an
    endpoint share one process-wide key, which the reset seam clears)."""
    return str(getattr(embedder, "endpoint", "") or "unknown-endpoint")


def _reset_embed_degradation_logged() -> None:
    """Clear the process-lifetime guards (log-once + circuit breaker).
    Test seam only: simulates a FRESH process, which starts with both an
    empty log-once set and a closed circuit."""
    _EMBED_DEGRADATION_LOGGED.clear()
    _EMBED_CIRCUIT_OPEN.clear()


# RRF k = 60 (Cormack-2009; agentmemory uses the same constant)
_RRF_K = 60

# FTS5 query-size bound. _bm25_search serves two caller classes: the
# interactive CLI (short human questions) and the PreToolUse/PostToolUse
# nudge hooks, which pass whole tool payloads (file contents, command
# output). SQLite's FTS5 MATCH memory grows with the number of OR terms:
# an unbounded query built from a multi-MB tool output allocated >12 GB
# inside sqlite and hung the hook process while it held index.db's write
# lock — the root cause of the 3.5-day hook-heartbeat outage found
# 2026-07-06. Queries above the cap are normalized (over-long tokens
# dropped, duplicates removed, then truncated); queries at or below it
# pass through byte-identical to the pre-fix behavior, so interactive
# ranking is unchanged. 256 distinct terms is far beyond any real
# question while keeping the sqlite query tree trivial.
_BM25_MAX_TERMS = 256
# Tokens longer than this are hashes / base64 / minified blobs, not
# vocabulary the FTS index contains; they bloat the MATCH string with no
# ranking value. Only enforced on the normalization path above the cap.
_BM25_MAX_TOKEN_LEN = 64


@dataclass
class Searcher:
    db_path: Path
    embedder: _Embedder
    # Where vector-leg degradation telemetry is written. None (the
    # default) derives the sibling telemetry.db next to db_path -- the
    # ProjectConfig layout every per-project index lives in -- so all
    # existing construction sites report degradation without changes.
    telemetry_path: Optional[Path] = None
    _conn: sqlite3.Connection = field(init=False)

    def __post_init__(self) -> None:
        if self.telemetry_path is None:
            self.telemetry_path = Path(self.db_path).parent / _TELEMETRY_DB_NAME
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        filter_do_not_rebuild: bool = False,
        filter_is_correction: bool = False,
        filter_operator_vetted: bool = False,
    ) -> List[Dict[str, Any]]:
        """Run hybrid BM25 + vector + RRF search with optional filters.

        Filters (any combination):
          - filter_do_not_rebuild: strict opt-in flag; matches only
            chunks explicitly tagged do_not_rebuild=1 during ingestion.
            Narrow; most corpora bulk-backfilled from history have
            very few such chunks.
          - filter_is_correction: matches only chunks tagged
            is_correction=1 (user corrections extracted from chat
            history).
          - filter_operator_vetted: union of do_not_rebuild OR
            is_correction OR is_decision OR signal_weight>=50. The
            broad "operator-vetted high-signal" gate intended for
            nudge / DO-NOT-REBUILD style consumers that want to surface
            real-world historical signal without requiring explicit
            do_not_rebuild tagging on every chunk."""
        bm25_rows = self._bm25_search(query, top_k * 3)
        vec_rows = self._vec_search(query, top_k * 3)
        fused = self._fuse(bm25_rows, vec_rows)
        ranked = self._apply_filters_and_boost(
            fused,
            filter_do_not_rebuild,
            filter_is_correction,
            filter_operator_vetted,
        )
        surfaced = ranked[:top_k]
        # Record that these chunks were surfaced for injection. This is the
        # usage-feedback wire that makes access_count / last_accessed real
        # (they are declared in schema.py but were never written before).
        # The prune-candidates command reads them to flag never-surfaced
        # memories. Failure here MUST NOT break retrieval.
        self._record_access([r["id"] for r in surfaced])
        return surfaced

    def _record_access(self, chunk_ids: List[str]) -> None:
        """Bump access_count + stamp last_accessed for the surfaced chunks.

        Wrapped in try/except: retrieval correctness strictly outranks
        usage bookkeeping. A failed UPDATE (locked db, schema drift, read-
        only mount) is swallowed so the search result is always returned."""
        if not chunk_ids:
            return
        try:
            now = datetime.now(timezone.utc).isoformat()
            placeholders = ",".join("?" for _ in chunk_ids)
            self._conn.execute(
                f"""
                UPDATE chunks
                SET access_count = access_count + 1,
                    last_accessed = ?
                WHERE id IN ({placeholders})
                """,
                (now, *chunk_ids),
            )
            self._conn.commit()
        except Exception:
            # Swallow: never let usage bookkeeping break a retrieval.
            pass

    def _bm25_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        # FTS5 MATCH with bm25() ranking. Each token is wrapped in double
        # quotes so FTS5 treats apostrophes / punctuation as literal text
        # rather than syntax. Empty queries short-circuit.
        tokens = [t.strip(".,!?;:()[]{}<>") for t in query.split() if t]
        tokens = [t for t in tokens if t]
        if len(tokens) > _BM25_MAX_TERMS:
            # Pathological (machine-generated) query: normalize before
            # the MATCH build. See _BM25_MAX_TERMS for the incident this
            # bounds. Order-preserving so the head of the payload (paths,
            # symbols, identifiers) wins the budget.
            seen: set[str] = set()
            capped: List[str] = []
            for t in tokens:
                if len(t) > _BM25_MAX_TOKEN_LEN or t in seen:
                    continue
                seen.add(t)
                capped.append(t)
                if len(capped) == _BM25_MAX_TERMS:
                    break
            tokens = capped
        if not tokens:
            return []
        # Replace embedded double-quotes with double-double-quotes (FTS5
        # escaping) and wrap each token.
        quoted = [f'"{t.replace(chr(34), chr(34) + chr(34))}"' for t in tokens]
        match = " OR ".join(quoted)
        rows = self._conn.execute(
            """
            SELECT c.*, bm25(chunks_fts) AS bm25_score
            FROM chunks_fts
            JOIN chunks c ON c.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY bm25_score
            LIMIT ?
            """,
            (match, k),
        ).fetchall()
        return [dict(r) for r in rows]

    def _record_embed_degradation(self, exc: Exception) -> None:
        """Log-once telemetry for a dead vector leg (2026-08-19 fix).

        The pre-fix behavior swallowed EVERY embed failure with zero
        trace: search degraded to BM25-only and nothing recorded it --
        the qwen3-embedding model was missing from Ollama for an
        unknown period precisely because this leg died silently. Now
        the first failure per process (per telemetry DB) writes one
        embed_degradation row via the existing telemetry helpers; the
        weekly report and the memory-health gate's embedding_path
        check surface it. Guarded so bookkeeping can never break the
        retrieval (same contract as _record_access)."""
        try:
            key = str(self.telemetry_path)
            if key in _EMBED_DEGRADATION_LOGGED:
                return
            _EMBED_DEGRADATION_LOGGED.add(key)
            record_embed_degradation(
                Path(self.telemetry_path),
                reason=f"{type(exc).__name__}: {exc}",
            )
        except Exception:
            # Never let degradation bookkeeping break a retrieval.
            pass

    def _vec_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        if _embed_circuit_key(self.embedder) in _EMBED_CIRCUIT_OPEN:
            # Embed already failed in this process: skip the doomed
            # retry (see _EMBED_CIRCUIT_OPEN) and stay BM25-only.
            return []
        try:
            embedding = self.embedder.embed(query)
        except Exception as exc:
            # Vector path is best-effort; degrade gracefully to
            # BM25-only -- but RECORD the degradation (log-once per
            # process) instead of dying silently, and open the circuit
            # so sibling searches in this process fail fast.
            _EMBED_CIRCUIT_OPEN.add(_embed_circuit_key(self.embedder))
            self._record_embed_degradation(exc)
            return []
        try:
            rows = self._conn.execute(
                """
                SELECT c.*, v.distance AS vec_distance
                FROM chunks_vec v
                JOIN chunks c ON c.id = v.chunk_id
                WHERE v.embedding MATCH ?
                ORDER BY v.distance
                LIMIT ?
                """,
                (sqlite_vec.serialize_float32(embedding), k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [dict(r) for r in rows]

    def _fuse(
        self,
        bm25_rows: List[Dict[str, Any]],
        vec_rows: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        scores: Dict[str, float] = {}
        chunks: Dict[str, Dict[str, Any]] = {}
        for rank, row in enumerate(bm25_rows):
            cid = row["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunks[cid] = row
        for rank, row in enumerate(vec_rows):
            cid = row["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunks.setdefault(cid, row)
        merged = []
        for cid, score in scores.items():
            row = dict(chunks[cid])
            row["fusion_score"] = score
            merged.append(row)
        merged.sort(key=lambda r: r["fusion_score"], reverse=True)
        return merged

    def _apply_filters_and_boost(
        self,
        rows: List[Dict[str, Any]],
        filter_do_not_rebuild: bool,
        filter_is_correction: bool,
        filter_operator_vetted: bool,
    ) -> List[Dict[str, Any]]:
        out = []
        for row in rows:
            if filter_do_not_rebuild and not row.get("do_not_rebuild"):
                continue
            if filter_is_correction and not row.get("is_correction"):
                continue
            if filter_operator_vetted and not _is_operator_vetted(row):
                continue
            sw = row.get("signal_weight") or 10
            row["final_score"] = row["fusion_score"] * (1.0 + sw / 100.0)
            out.append(row)
        out.sort(key=lambda r: r["final_score"], reverse=True)
        return out

    def close(self) -> None:
        self._conn.close()


def _is_operator_vetted(row: Dict[str, Any]) -> bool:
    """Predicate used by filter_operator_vetted -- union of operator-
    tagged high-signal markers.

    A chunk is operator-vetted when ANY of:
      - do_not_rebuild=1 (explicit ingestion tag)
      - is_correction=1 (user correction extracted from session jsonl)
      - is_decision=1   (decision marker extracted from session jsonl)
      - signal_weight >= 50 (SessionStart "important context" floor)

    Empirical: the origin project's 2793-chunk corpus has 0 do_not_rebuild chunks
    but 956 is_decision, 146 is_correction, 1215 signal_weight>=50.
    The narrow do_not_rebuild-only filter excluded the entire vetted
    pool; this predicate captures real-world operator signal."""
    if row.get("do_not_rebuild"):
        return True
    if row.get("is_correction"):
        return True
    if row.get("is_decision"):
        return True
    sw = row.get("signal_weight")
    if sw is not None and int(sw) >= 50:
        return True
    return False
