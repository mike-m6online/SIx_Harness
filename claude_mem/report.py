"""Weekly summary report.

Aggregates telemetry.db over the last 7 days; emits a markdown summary.
Output goes to docs/marathon/memory_system_weekly/ when that directory
exists; falls back to .claude-mem/reports/ otherwise. Mike-locked:
auto-commit so degradation is visible at human cadence.
"""
from __future__ import annotations

import collections
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def build_weekly_summary(
    telemetry_db: Path, *, days: int = 7,
) -> str:
    """Return a markdown summary of telemetry.db over the last `days`."""
    if not telemetry_db.is_file():
        return "# claude-mem weekly summary\n\n(no telemetry.db yet)"
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    conn = sqlite3.connect(telemetry_db)
    try:
        invs = conn.execute(
            "SELECT * FROM wrapper_invocations WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall()
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM wrapper_invocations LIMIT 0"
        ).description]
        hbs = conn.execute(
            "SELECT component, status, detail FROM heartbeat "
            "WHERE timestamp >= ?",
            (cutoff,),
        ).fetchall()
        # Vector-leg degradation events (2026-08-19 embed-resilience fix:
        # search.py logs once per process when the embed leg fails and the
        # search falls back to BM25-only). Legacy telemetry DBs predate the
        # table; treat a missing table as "no instrumentation", not zero.
        try:
            degradations = conn.execute(
                "SELECT timestamp, reason FROM embed_degradation "
                "WHERE timestamp >= ? ORDER BY id DESC",
                (cutoff,),
            ).fetchall()
        except sqlite3.OperationalError:
            degradations = None
    finally:
        conn.close()

    rows = [dict(zip(cols, r)) for r in invs]
    lines: list[str] = []
    lines.append("# claude-mem weekly summary")
    lines.append(
        f"Window: last {days} days  |  Generated: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    lines.append("")
    lines.append("## Wrapper")
    lines.append(f"- Wrapper invocations: {len(rows)}")
    lines.append(
        f"- Build-intent fires: "
        f"{sum(1 for r in rows if r['build_intent_fired'])}"
    )
    lines.append(
        f"- Investigation-intent fires: "
        f"{sum(1 for r in rows if r['investigation_intent_fired'])}"
    )
    lines.append(
        f"- DO NOT REBUILD warnings emitted: "
        f"{sum(1 for r in rows if r['do_not_rebuild_warning_emitted'])}"
    )
    lines.append(
        f"- Stale-claim warnings emitted: "
        f"{sum(1 for r in rows if r['stale_claim_warning_emitted'])}"
    )
    latencies = [r["retrieval_latency_ms"] for r in rows
                 if r["retrieval_latency_ms"]]
    if latencies:
        latencies.sort()
        p50 = latencies[len(latencies) // 2]
        p95 = latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))]
        lines.append(f"- Retrieval latency p50/p95: {p50} / {p95} ms")
    lines.append("")

    # Embed vector leg (one log-once row per process that degraded)
    if degradations is not None:
        lines.append("## Embed vector leg")
        lines.append(
            f"- Degradation events (search fell back to BM25-only): "
            f"{len(degradations)}"
        )
        if degradations:
            last_ts, last_reason = degradations[0]
            lines.append(
                f"- WARN: vector leg degraded in window; latest "
                f"{last_ts}: {last_reason}"
            )
        lines.append("")

    # Top topics
    topic_counts: collections.Counter = collections.Counter()
    for r in rows:
        try:
            topics = json.loads(r["retrieved_chunk_topics"] or "[]")
        except json.JSONDecodeError:
            continue
        for t in topics:
            topic_counts[t] += 1
    if topic_counts:
        lines.append("## Top topics triggering warnings")
        for t, c in topic_counts.most_common(10):
            lines.append(f"- {t}: {c}")
        lines.append("")

    # Component health
    if hbs:
        statuses: dict[str, list[str]] = {}
        for component, status, _ in hbs:
            statuses.setdefault(component, []).append(status)
        lines.append("## Component health (heartbeat samples)")
        for component, hist in statuses.items():
            n = len(hist)
            alive = sum(1 for s in hist if s == "alive")
            degraded = sum(1 for s in hist if s == "degraded")
            dead = sum(1 for s in hist if s == "dead")
            note = ""
            if dead > 0 or degraded > 0:
                note = "  WARN: degraded/dead samples in window"
            lines.append(
                f"- {component}: alive={alive}/{n} degraded={degraded} "
                f"dead={dead}{note}"
            )
        lines.append("")

    if not rows and not hbs:
        lines.append("(no telemetry rows in the window)")

    return "\n".join(lines)


def write_weekly_summary(
    telemetry_db: Path,
    project_root: Path,
    *,
    days: int = 7,
) -> Path:
    """Build the summary; write it to docs/marathon/memory_system_weekly/
    when that dir exists, else .claude-mem/reports/. Returns the path
    written."""
    summary = build_weekly_summary(telemetry_db, days=days)
    canonical = project_root / "docs" / "marathon" / "memory_system_weekly"
    if canonical.parent.is_dir():
        canonical.mkdir(parents=True, exist_ok=True)
        out_dir = canonical
    else:
        out_dir = project_root / ".claude-mem" / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%W")
    out = out_dir / f"{ts}.md"
    out.write_text(summary, encoding="utf-8")
    return out
