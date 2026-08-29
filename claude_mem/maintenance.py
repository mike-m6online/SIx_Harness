"""Memory maintenance signals derived from the usage-feedback wire.

The chunks table declares access_count + last_accessed; search.py and the
session_start hook now write them every time a chunk is surfaced for
injection. This module reads those columns to surface the actionable
maintenance signal: chunks that have NEVER been retrieved since ingestion
and are old enough to be confident about are prune candidates.

find_prune_candidates is REPORT ONLY. retag_corrections is the one
repair pass here: an UPDATE-only flag retag (no rows deleted -- Rule #2:
archive, never delete memory/experiment data) that applies the
ingestion-time curated-correction rule (bulk.is_curated_correction_file)
to chunks ingested BEFORE the rule existed.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec

from claude_mem.bulk import is_curated_correction_file


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def find_prune_candidates(
    db_path: Path,
    *,
    min_age_days: int = 30,
    limit: int = 50,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """Return chunks with access_count=0 ingested more than min_age_days
    ago, oldest first, capped at `limit`. Each row carries an age_days
    field computed from ingested_at.

    `now` is injectable for deterministic tests; defaults to utcnow.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        rows = conn.execute(
            """
            SELECT id, content, source, module, ingested_at,
                   access_count, last_accessed
            FROM chunks
            WHERE access_count = 0
            ORDER BY ingested_at ASC
            """
        ).fetchall()
    finally:
        conn.close()

    out: List[Dict[str, Any]] = []
    for r in rows:
        ingested = _parse_iso(r["ingested_at"])
        if ingested is None:
            # No usable ingest timestamp: cannot judge age, skip from the
            # age-gated candidate list rather than guess.
            continue
        age_days = (now - ingested).days
        if age_days < min_age_days:
            continue
        d = dict(r)
        d["age_days"] = age_days
        out.append(d)
        if len(out) >= limit:
            break
    return out


def retag_corrections(db_path: Path) -> Dict[str, int]:
    """One-shot repair pass: apply the ingestion-time curated-correction
    rule (bulk.is_curated_correction_file -- memory files whose basename
    starts with ``feedback_`` or ``invariant_``) to ALREADY-ingested
    memory chunks, setting is_correction=1 where the rule matches.

    Same rule, second surface: the ingestion paths (cli.bulk step 3 and
    incremental._scan_memory) now tag these at write time; this pass
    retags the historical rows those paths wrote before the rule existed
    so the operator runs it ONCE on a live index. Idempotent -- a second
    run finds everything already tagged. UPDATE-only; never touches
    content, never deletes.

    Returns counts: ``scanned`` (memory chunks examined), ``matched``
    (chunks whose file basename matches the rule), ``retagged`` (flag
    actually flipped this run), ``already_tagged`` (matched but already
    is_correction=1).
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT id, file_path, is_correction
            FROM chunks
            WHERE source = 'memory'
            """
        ).fetchall()
        matched_ids: List[str] = []
        already_tagged = 0
        for r in rows:
            fp = r["file_path"]
            if not fp or not is_curated_correction_file(fp):
                continue
            if r["is_correction"]:
                already_tagged += 1
            else:
                matched_ids.append(r["id"])
        if matched_ids:
            placeholders = ",".join("?" for _ in matched_ids)
            conn.execute(
                f"UPDATE chunks SET is_correction = 1 "
                f"WHERE id IN ({placeholders})",
                matched_ids,
            )
            conn.commit()
        return {
            "scanned": len(rows),
            "matched": len(matched_ids) + already_tagged,
            "retagged": len(matched_ids),
            "already_tagged": already_tagged,
        }
    finally:
        conn.close()
