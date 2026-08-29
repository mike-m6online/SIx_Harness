"""Structured capture store for the anti-recurrence system (Rung 2).

Threads / decisions / dead-ends live as plain rows in the claude-mem DB
(no FTS, no vector embedding -- these are surfaced by gen_decisions_state at
session start and by the Rung 3 synthesis path, not by chunk search). The
store mirrors the Ingester insert+content-SHA-dedup pattern in ingest.py."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from claude_mem.schema import _ensure_columns


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in text.strip())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")[:64]


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


@dataclass
class Thread:
    name: str
    summary: str = ""
    state: str = "open"
    opened_date: Optional[str] = None
    last_updated: Optional[str] = None
    id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = _slug(self.name)
        if not self.opened_date:
            self.opened_date = _now()
        if not self.last_updated:
            self.last_updated = self.opened_date


@dataclass
class Decision:
    title: str
    date: Optional[str] = None
    rationale: str = ""
    options_rejected: List[str] = field(default_factory=list)
    state: str = "pending"
    thread_id: Optional[str] = None
    linked_commits: List[str] = field(default_factory=list)
    mike_approved: bool = False
    id: str = ""

    def __post_init__(self) -> None:
        if not self.date:
            self.date = _now()
        if not self.id:
            self.id = _sha(f"{self.date}|{self.title}")


@dataclass
class DeadEnd:
    approach: str
    date: Optional[str] = None
    why_shelved: str = ""
    superseded_by: str = ""
    thread_id: Optional[str] = None
    state: str = "pending"
    id: str = ""

    def __post_init__(self) -> None:
        if not self.date:
            self.date = _now()
        if not self.id:
            self.id = _sha(f"{self.date}|{self.approach}")


class CaptureStore:
    """Insert / query / update helpers for the capture tables."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        self._conn.commit()
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS delivered_nudges ("
            " session_id TEXT, chunk_id TEXT, tool_name TEXT, delivered_at TEXT,"
            " PRIMARY KEY (session_id, chunk_id))"
        )
        # Additive migration (Task B2): delivery_count upgrades the
        # presence-only dedup table into a per-session delivery COUNTER so
        # prompt_submit can damp repeat injections ("at most N per
        # session") through the SAME mechanism the tool-use nudge path
        # already uses, instead of a parallel table. Existing rows read as
        # count=1 (the value their single INSERT represented).
        _ensure_columns(self._conn, "delivered_nudges", {
            "delivery_count": "INTEGER DEFAULT 1",
        })
        self._conn.commit()

    def close(self) -> None:
        self._conn.commit()
        self._conn.close()

    def __enter__(self) -> "CaptureStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _thread_exists(self, thread_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM threads WHERE id=?", (thread_id,)
        ).fetchone() is not None

    # ---- threads ----
    def add_thread(self, t: Thread) -> bool:
        if self._conn.execute("SELECT 1 FROM threads WHERE id=?", (t.id,)).fetchone():
            return False
        self._conn.execute(
            "INSERT INTO threads (id, name, opened_date, state, last_updated, summary)"
            " VALUES (?,?,?,?,?,?)",
            (t.id, t.name, t.opened_date, t.state, t.last_updated, t.summary),
        )
        self._conn.commit()
        return True

    # ---- decisions ----
    def add_decision(self, d: Decision) -> bool:
        if self._conn.execute("SELECT 1 FROM decisions WHERE id=?", (d.id,)).fetchone():
            return False
        if d.thread_id and not self._thread_exists(d.thread_id):
            raise ValueError(f"add_decision: no thread with id={d.thread_id!r}")
        self._conn.execute(
            "INSERT INTO decisions (id, date, title, rationale, options_rejected,"
            " state, thread_id, linked_commits, mike_approved)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                d.id, d.date, d.title, d.rationale,
                json.dumps(d.options_rejected), d.state, d.thread_id,
                json.dumps(d.linked_commits), int(d.mike_approved),
            ),
        )
        self._conn.commit()
        if d.thread_id:
            self.touch_thread(d.thread_id)
        return True

    # ---- dead ends ----
    def add_dead_end(self, e: DeadEnd) -> bool:
        if self._conn.execute("SELECT 1 FROM dead_ends WHERE id=?", (e.id,)).fetchone():
            return False
        if e.thread_id and not self._thread_exists(e.thread_id):
            raise ValueError(f"add_dead_end: no thread with id={e.thread_id!r}")
        self._conn.execute(
            "INSERT INTO dead_ends (id, date, approach, why_shelved, superseded_by,"
            " thread_id, state) VALUES (?,?,?,?,?,?,?)",
            (e.id, e.date, e.approach, e.why_shelved, e.superseded_by,
             e.thread_id, e.state),
        )
        self._conn.commit()
        if e.thread_id:
            self.touch_thread(e.thread_id)
        return True

    def update_thread_summary(self, thread_id: str, summary: str) -> None:
        """Rewrite a thread's summary (spec R6 falsified-lineage correction:
        the SUMMARY line is rendered into build_lineage_prompt ahead of the
        dated decisions/dead-ends and anchors the synthesis LLM, so a
        falsified claim living only in the dead-end row's superseded_by
        field is not sufficient -- the summary itself must be corrected
        too). Bumps last_updated so the lineage cache gate regenerates."""
        self._conn.execute(
            "UPDATE threads SET summary=? WHERE id=?", (summary, thread_id)
        )
        self._conn.commit()
        self.touch_thread(thread_id)

    def touch_thread(self, thread_id: str) -> None:
        self._conn.execute(
            "UPDATE threads SET last_updated=? WHERE id=?", (_now(), thread_id)
        )
        self._conn.commit()

    def get_thread(self, thread_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM threads WHERE id=?", (thread_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_threads(self, state: Optional[str] = None) -> List[dict]:
        if state:
            rows = self._conn.execute(
                "SELECT * FROM threads WHERE state=? ORDER BY last_updated DESC",
                (state,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM threads ORDER BY last_updated DESC"
            ).fetchall()
        return [dict(r) for r in rows]

    def list_decisions(
        self, *, thread_id: Optional[str] = None, state: Optional[str] = None
    ) -> List[dict]:
        clauses, params = [], []
        if thread_id:
            clauses.append("thread_id=?")
            params.append(thread_id)
        if state:
            clauses.append("state=?")
            params.append(state)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM decisions{where} ORDER BY date DESC", params
        ).fetchall()
        return [dict(r) for r in rows]

    def list_dead_ends(
        self, *, thread_id: Optional[str] = None, state: Optional[str] = None
    ) -> List[dict]:
        clauses, params = [], []
        if thread_id:
            clauses.append("thread_id=?")
            params.append(thread_id)
        if state:
            clauses.append("state=?")
            params.append(state)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM dead_ends{where} ORDER BY date DESC", params
        ).fetchall()
        return [dict(r) for r in rows]

    def confirm_decision(
        self, decision_id: str, *, mike_approved: Optional[bool] = None
    ) -> None:
        if mike_approved is None:
            self._conn.execute(
                "UPDATE decisions SET state='confirmed' WHERE id=?", (decision_id,)
            )
        else:
            self._conn.execute(
                "UPDATE decisions SET state='confirmed', mike_approved=? WHERE id=?",
                (int(mike_approved), decision_id),
            )
        self._conn.commit()

    def reject_decision(self, decision_id: str) -> None:
        """Mark a pending decision rejected (capture-triage verdict)."""
        self._conn.execute(
            "UPDATE decisions SET state='rejected' WHERE id=?", (decision_id,)
        )
        self._conn.commit()

    def set_decision_thread(
        self, decision_id: str, thread_id: str, *, title: Optional[str] = None
    ) -> None:
        """Attach a decision to a thread (capture-triage verdict), optionally
        rewriting its title (auto-extracted titles are frequently truncated
        sentence fragments an operator wants to clean up at triage time).
        Bumps the thread's last_updated so synthesize's cache gate
        (get_or_regenerate_lineage) regenerates the lineage on next call.
        Raises ValueError if the thread does not exist (mirrors
        add_decision's guard)."""
        if not self._thread_exists(thread_id):
            raise ValueError(f"set_decision_thread: no thread with id={thread_id!r}")
        if title is not None:
            self._conn.execute(
                "UPDATE decisions SET thread_id=?, title=? WHERE id=?",
                (thread_id, title, decision_id),
            )
        else:
            self._conn.execute(
                "UPDATE decisions SET thread_id=? WHERE id=?",
                (thread_id, decision_id),
            )
        self._conn.commit()
        self.touch_thread(thread_id)

    def reject_dead_end(self, dead_end_id: str) -> None:
        """Mark a pending dead-end rejected (capture-triage verdict)."""
        self._conn.execute(
            "UPDATE dead_ends SET state='rejected' WHERE id=?", (dead_end_id,)
        )
        self._conn.commit()

    def confirm_dead_end(self, dead_end_id: str) -> None:
        """Mark a pending dead-end confirmed (capture-triage verdict)."""
        self._conn.execute(
            "UPDATE dead_ends SET state='confirmed' WHERE id=?", (dead_end_id,)
        )
        self._conn.commit()

    def supersede_dead_end(self, dead_end_id: str, superseded_by: str) -> None:
        """Set a dead-end's superseded_by text -- the mechanism for
        correcting a falsified dead-end framing in place (spec R6):
        build_lineage_prompt renders `-> superseded by: {superseded_by}`
        next to the dead-end row, which is what actually steers the
        synthesis LLM away from re-asserting the falsified claim (merely
        adding a new confirmed decision to the same thread is not
        sufficient -- the dead-end row itself has to carry the correction).
        Bumps the owning thread's last_updated when the dead-end has one
        (no-op on last_updated for an unattached dead-end)."""
        row = self._conn.execute(
            "SELECT thread_id FROM dead_ends WHERE id=?", (dead_end_id,)
        ).fetchone()
        self._conn.execute(
            "UPDATE dead_ends SET superseded_by=? WHERE id=?",
            (superseded_by, dead_end_id),
        )
        self._conn.commit()
        if row and row["thread_id"]:
            self.touch_thread(row["thread_id"])

    def set_dead_end_thread(
        self, dead_end_id: str, thread_id: str, *, approach: Optional[str] = None
    ) -> None:
        """Attach a dead-end to a thread (capture-triage verdict), optionally
        rewriting its approach text. Bumps the thread's last_updated (see
        set_decision_thread). Raises ValueError for an unknown thread."""
        if not self._thread_exists(thread_id):
            raise ValueError(f"set_dead_end_thread: no thread with id={thread_id!r}")
        if approach is not None:
            self._conn.execute(
                "UPDATE dead_ends SET thread_id=?, approach=? WHERE id=?",
                (thread_id, approach, dead_end_id),
            )
        else:
            self._conn.execute(
                "UPDATE dead_ends SET thread_id=? WHERE id=?",
                (thread_id, dead_end_id),
            )
        self._conn.commit()
        self.touch_thread(thread_id)

    def get_cached_lineage(self, thread_id: str):
        """Return (lineage_text, lineage_cache_key) for a thread, or
        (None, None) if absent or thread unknown."""
        row = self._conn.execute(
            "SELECT lineage_text, lineage_cache_key FROM threads WHERE id=?",
            (thread_id,),
        ).fetchone()
        if not row:
            return (None, None)
        return (row["lineage_text"], row["lineage_cache_key"])

    def set_cached_lineage(self, thread_id: str, text: str, key: str) -> None:
        """Write lineage text + cache key for a thread and record the timestamp."""
        self._conn.execute(
            "UPDATE threads SET lineage_text=?, lineage_cached_at=?, "
            "lineage_cache_key=? WHERE id=?",
            (text, _now(), key, thread_id),
        )
        self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM meta WHERE key=?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def set_meta(self, key: str, value: str) -> None:
        self._conn.execute(
            "INSERT INTO meta (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        self._conn.commit()

    def was_delivered(self, session_id: str, chunk_id: str) -> bool:
        return self._conn.execute(
            "SELECT 1 FROM delivered_nudges WHERE session_id=? AND chunk_id=?",
            (session_id, chunk_id),
        ).fetchone() is not None

    def record_delivered(
        self, session_id: str, chunk_id: str, tool_name: str
    ) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO delivered_nudges"
            " (session_id, chunk_id, tool_name, delivered_at) VALUES (?,?,?,?)",
            (session_id, chunk_id, tool_name, _now()),
        )
        self._conn.commit()

    def delivery_count(self, session_id: str, item_key: str) -> int:
        """How many times ``item_key`` was delivered in ``session_id``.

        ``item_key`` shares the delivered_nudges.chunk_id column with the
        tool-use nudge path's chunk ids; prompt_submit's keys are
        namespaced ('thread:<id>', 'dnr', 'stale') so they cannot collide
        with sha256-hex chunk ids. Returns 0 when never delivered.
        """
        row = self._conn.execute(
            "SELECT delivery_count FROM delivered_nudges"
            " WHERE session_id=? AND chunk_id=?",
            (session_id, item_key),
        ).fetchone()
        if row is None:
            return 0
        # Pre-migration rows can hold NULL (column added after insert).
        return int(row["delivery_count"]) if row["delivery_count"] is not None else 1

    def record_delivery(
        self, session_id: str, item_key: str, tool_name: str
    ) -> None:
        """Record one delivery of ``item_key`` in ``session_id``,
        incrementing delivery_count on repeat deliveries (contrast
        :meth:`record_delivered`, which keeps first-delivery presence
        semantics for the tool-use nudge dedup path).
        """
        self._conn.execute(
            "INSERT INTO delivered_nudges"
            " (session_id, chunk_id, tool_name, delivered_at, delivery_count)"
            " VALUES (?,?,?,?,1)"
            " ON CONFLICT(session_id, chunk_id) DO UPDATE SET"
            "  delivery_count=COALESCE(delivery_count, 1) + 1,"
            "  delivered_at=excluded.delivered_at,"
            "  tool_name=excluded.tool_name",
            (session_id, item_key, tool_name, _now()),
        )
        self._conn.commit()
