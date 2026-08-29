"""Telemetry tables + heartbeat for claude-mem.

Per spec section 4.6.6 + Mike-locked fail-loud discipline: the system
silently degrading is the worst failure mode. Heartbeat records
component status every 5 min (via cron); weekly summary aggregates +
auto-commits to a project doc so degradation is visible at human
cadence.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import httpx

from claude_mem.schema import _ensure_columns


_WRAPPER_INVOCATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS wrapper_invocations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT,
    prompt_truncated TEXT,
    build_intent_fired INTEGER,
    investigation_intent_fired INTEGER,
    do_not_rebuild_warning_emitted INTEGER,
    stale_claim_warning_emitted INTEGER,
    retrieved_chunk_count INTEGER,
    retrieved_chunk_topics TEXT,
    retrieval_latency_ms INTEGER,
    session_id TEXT
);
"""

_HEARTBEAT_TABLE = """
CREATE TABLE IF NOT EXISTS heartbeat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT
);
"""

# Distinct from `heartbeat` above: that table records periodic component
# probes (ollama/index liveness, driven by the `claude-mem heartbeat`
# cron command). `hook_heartbeat` records one row per HOOK INVOCATION
# (session_start, prompt_submit, tool_use, tool_use_post, session_end,
# capture_extract, capture_synthesize, ...), success AND failure, so
# Task 7's memory-health gate (check #3: "last success > 2 days or last
# run errored") can detect a hook that silently died -- the cp1252
# gen_decisions_state crash class of bug that went unnoticed for a week.
_HOOK_HEARTBEAT_TABLE = """
CREATE TABLE IF NOT EXISTS hook_heartbeat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    hook TEXT NOT NULL,
    ok INTEGER NOT NULL,
    detail TEXT
);
"""

# 2026-08-19 embed-resilience fix: one row per PROCESS in which the
# hybrid search's vector leg failed to embed and search degraded to
# BM25-only (search.py's _vec_search previously swallowed the failure
# with zero telemetry -- the qwen3-embedding model was missing from
# Ollama for an unknown period precisely because nothing recorded the
# degradation). search.py writes at most one row per process (log-once)
# so a chatty hook cannot flood the table; the weekly report surfaces
# the count.
_EMBED_DEGRADATION_TABLE = """
CREATE TABLE IF NOT EXISTS embed_degradation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    reason TEXT
);
"""


def init_telemetry_db(db_path: Path) -> None:
    """Idempotent telemetry schema init."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(_WRAPPER_INVOCATIONS_TABLE)
        conn.execute(_HEARTBEAT_TABLE)
        conn.execute(_HOOK_HEARTBEAT_TABLE)
        conn.execute(_EMBED_DEGRADATION_TABLE)
        # Additive migration (Task B5): prompt_submit's injection-history
        # instrumentation. decision_intent_fired completes the intent-flag
        # trio; lineage_* record which threads were injected and why
        # (matched-token/IDF evidence); suppressed_by_damping records the
        # per-session damping (B2) verdicts so the shakedown can measure
        # both what fired and what was withheld.
        _ensure_columns(conn, "wrapper_invocations", {
            "decision_intent_fired": "INTEGER",
            "lineage_block_emitted": "INTEGER",
            "lineage_thread_ids": "TEXT",
            "matched_token_summary": "TEXT",
            "suppressed_by_damping": "TEXT",
        })
        conn.commit()
    finally:
        conn.close()


def record_hook_heartbeat(
    db_path: Path, *, hook: str, ok: bool, detail: str = "",
) -> None:
    """Write one hook-invocation heartbeat row. Called from a try/finally
    in EVERY hook entry point (cli.py), success and failure both.

    Must be cheap (<10ms; a single-row INSERT) and must NEVER raise --
    a heartbeat write failure must never break the hook it is observing.
    Callers therefore do not need to wrap this in their own try/except,
    but MUST still call it from a try/finally so a hook exception still
    gets an ok=False row recorded before re-raising / being swallowed.
    """
    try:
        init_telemetry_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO hook_heartbeat (timestamp, hook, ok, detail) "
                "VALUES (?, ?, ?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    hook, int(bool(ok)), (detail or "")[:500],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Heartbeat bookkeeping must never be the reason a hook breaks.
        pass


def record_wrapper_invocation(
    db_path: Path,
    *,
    prompt_hash: Optional[str] = None,
    prompt_truncated: str = "",
    build_intent_fired: bool = False,
    investigation_intent_fired: bool = False,
    do_not_rebuild_warning_emitted: bool = False,
    stale_claim_warning_emitted: bool = False,
    retrieved_chunk_count: int = 0,
    retrieved_chunk_topics: Optional[List[str]] = None,
    retrieval_latency_ms: int = 0,
    session_id: Optional[str] = None,
    decision_intent_fired: bool = False,
    lineage_block_emitted: bool = False,
    lineage_thread_ids: Optional[List[str]] = None,
    matched_token_summary: Optional[Dict] = None,
    suppressed_by_damping: Optional[List[str]] = None,
) -> None:
    """Record one prompt_submit hook invocation (the injection history).

    Column semantics (Task B5 shakedown instrument):
      - retrieved_chunk_count: raw retrieved (pre-filter) DNR+correction
        hit count -- the denominator for the per-item filter's precision.
      - retrieved_chunk_topics: modules of the items that PASSED the
        B3 relevance filter (what was actually listed).
      - lineage_thread_ids: thread ids whose lineage blocks were emitted.
      - matched_token_summary: JSON evidence of WHY blocks fired --
        matched token -> idf maps per block/thread.
      - suppressed_by_damping: item keys ('thread:<id>' / 'dnr' /
        'stale') withheld by the per-session damping caps (B2).
    """
    init_telemetry_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO wrapper_invocations (
                timestamp, prompt_hash, prompt_truncated,
                build_intent_fired, investigation_intent_fired,
                do_not_rebuild_warning_emitted, stale_claim_warning_emitted,
                retrieved_chunk_count, retrieved_chunk_topics,
                retrieval_latency_ms, session_id,
                decision_intent_fired, lineage_block_emitted,
                lineage_thread_ids, matched_token_summary,
                suppressed_by_damping
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                prompt_hash,
                prompt_truncated[:200],
                int(build_intent_fired),
                int(investigation_intent_fired),
                int(do_not_rebuild_warning_emitted),
                int(stale_claim_warning_emitted),
                retrieved_chunk_count,
                json.dumps(retrieved_chunk_topics or []),
                retrieval_latency_ms,
                session_id,
                int(decision_intent_fired),
                int(lineage_block_emitted),
                json.dumps(lineage_thread_ids or []),
                json.dumps(matched_token_summary or {}),
                json.dumps(suppressed_by_damping or []),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def record_embed_degradation(db_path: Path, *, reason: str) -> None:
    """Record ONE vector-leg degradation event: the hybrid search's embed
    call failed and the search fell back to BM25-only.

    Written by search.Searcher at most once per process (log-once) so a
    single dead/cold Ollama cannot flood the table from a chatty hook.
    Like record_hook_heartbeat this must be cheap and must NEVER raise:
    degradation bookkeeping must never be the reason a retrieval breaks.
    """
    try:
        init_telemetry_db(db_path)
        conn = sqlite3.connect(db_path)
        try:
            conn.execute(
                "INSERT INTO embed_degradation (timestamp, reason) "
                "VALUES (?, ?)",
                (
                    datetime.now(timezone.utc).isoformat(),
                    (reason or "")[:500],
                ),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        # Degradation bookkeeping must never break a retrieval.
        pass


def record_heartbeat(
    db_path: Path,
    *,
    component: str,
    status: str,
    detail: Optional[Dict] = None,
) -> None:
    init_telemetry_db(db_path)
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO heartbeat (timestamp, component, status, detail)
            VALUES (?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                component, status, json.dumps(detail or {}),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def _probe_ollama(
    endpoint: str, model: str, keep_alive: str = "4h",
) -> Tuple[str, Dict]:
    """Probe Ollama embedding endpoint; return (status, detail).

    Carries keep_alive because every Ollama request RE-ARMS the model's
    residency timer: a probe that omits it would reset the timer to the
    server default (5m) and silently undo the embed-resilience
    keep_alive the real embed path passes (see embed.EmbeddingClient).
    """
    try:
        with httpx.Client(timeout=5.0) as client:
            resp = client.post(
                f"{endpoint}/api/embeddings",
                json={"model": model, "prompt": "heartbeat",
                      "keep_alive": keep_alive},
            )
            if resp.status_code == 200:
                return ("alive", {"endpoint": endpoint, "model": model})
            return ("degraded", {
                "endpoint": endpoint, "model": model,
                "status_code": resp.status_code,
                "body": resp.text[:200],
            })
    except httpx.HTTPError as exc:
        return ("dead", {"endpoint": endpoint, "error": str(exc)})


def _probe_index(db_path: Path) -> Tuple[str, Dict]:
    if not db_path.is_file():
        return ("dead", {"reason": "no index.db"})
    try:
        conn = sqlite3.connect(db_path)
        try:
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        return ("alive", {"chunk_count": int(n)})
    except sqlite3.Error as exc:
        return ("dead", {"error": str(exc)})


def probe_components(cfg) -> Dict[str, Tuple[str, Dict]]:
    """Probe each component; record a heartbeat row + return statuses.

    Returns: {component: (status, detail_dict)}
    """
    statuses: Dict[str, Tuple[str, Dict]] = {}
    statuses["ollama"] = _probe_ollama(
        cfg.values["ollama_endpoint"], cfg.values["embedding_model"],
        keep_alive=cfg.values.get("embedding_keep_alive", "4h"),
    )
    statuses["index"] = _probe_index(cfg.db_path)
    for component, (status, detail) in statuses.items():
        record_heartbeat(
            cfg.telemetry_path, component=component, status=status,
            detail=detail,
        )
    return statuses
