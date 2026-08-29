#!/usr/bin/env python3
"""Generate DECISIONS_STATE -- the auto-derived session-start view of the
anti-recurrence capture layer (Rung 2). Mirrors the project's project-state
generator: reads the threads / decisions / dead_ends tables from the
claude-mem DB and renders a capped markdown block to stdout (--print) for
SessionStart injection. Edit this generator, never the emitted block.

Harness-kit parameterization (the ONLY deltas from the origin-project original):
  - `--project-root` and `--memory-dir` are REQUIRED flags; omitting either
    exits 2 with argparse's standard clear message. The harness init tool
    bakes concrete absolute paths into the installed hook command, so
    runtime discovery is unnecessary and no default exists (the original
    defaulted --project-root from its own file location).
  - `--memory-dir` is accepted for the uniform harness hook interface
    (every installed hook receives the same flag pair); this generator
    reads only <project-root>/.claude-mem/index.db and writes only
    <project-root>/DECISIONS_STATE.md (when --print is not given).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sqlite3
import sys
from pathlib import Path
from typing import List

# Windows console defaults to cp1252; decision/dead-end titles routinely
# carry non-cp1252 characters (e.g. U+2192 '->'). Reconfigure stdout to
# UTF-8 (with lossy fallback rather than a crash) at entry, before any
# write. This exact class of bug silently killed the SessionStart
# DECISIONS_STATE injection for ~a week (see memory-health-and-repair
# spec R5) -- `sys.stdout.write(out)` raised UnicodeEncodeError and the
# hook's stdout (and therefore the injected context) was simply empty.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_NAME = "DECISIONS_STATE.md"
LINE_CAP = 150


def _db_path(project_root: Path) -> Path:
    return project_root / ".claude-mem" / "index.db"


def _has_tables(conn: sqlite3.Connection) -> bool:
    names = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    return {"threads", "decisions", "dead_ends"}.issubset(names)


def _fmt_rejected(raw) -> str:
    try:
        items = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        items = []
    return "; ".join(items) if items else ""


def render(conn: sqlite3.Connection, *, line_cap: int = LINE_CAP) -> str:
    conn.row_factory = sqlite3.Row
    now = _dt.datetime.now().isoformat(timespec="seconds")
    threads = [dict(r) for r in conn.execute(
        "SELECT * FROM threads ORDER BY last_updated DESC").fetchall()]
    decisions = [dict(r) for r in conn.execute(
        "SELECT * FROM decisions ORDER BY date DESC").fetchall()]
    dead_ends = [dict(r) for r in conn.execute(
        "SELECT * FROM dead_ends ORDER BY date DESC").fetchall()]

    open_threads = [t for t in threads if t["state"] == "open"]
    pending_dec = [d for d in decisions if d["state"] == "pending"]
    pending_de = [e for e in dead_ends if e["state"] == "pending"]

    by_thread_dec = {}
    for d in decisions:
        by_thread_dec.setdefault(d["thread_id"], []).append(d)
    by_thread_de = {}
    for e in dead_ends:
        by_thread_de.setdefault(e["thread_id"], []).append(e)

    lines: List[str] = []
    lines.append(f"=== GENERATED {now} -- DO NOT HAND-EDIT ===")
    lines.append("Source: hooks/gen_decisions_state.py (anti-recurrence Rung 2)")
    lines.append(
        f"threads: {len(threads)} (open: {len(open_threads)}); "
        f"decisions: {len(decisions)} (pending: {len(pending_dec)}); "
        f"dead-ends: {len(dead_ends)} (pending: {len(pending_de)})"
    )
    lines.append("")

    if open_threads:
        lines.append("## OPEN THREADS")
        for t in open_threads:
            lines.append(f"### {t['name']} [{t['state']}] (updated {t['last_updated']})")
            if t.get("summary"):
                lines.append(f"  {t['summary']}")
            for d in by_thread_dec.get(t["id"], []):
                tag = " (Mike-approved)" if d["mike_approved"] else ""
                st = "" if d["state"] == "confirmed" else f" [{d['state']}]"
                lines.append(f"  - DECISION{st}{tag} [{d['date']}] {d['title']}")
                rej = _fmt_rejected(d["options_rejected"])
                if rej:
                    lines.append(f"      rejected: {rej}")
            for e in by_thread_de.get(t["id"], []):
                sup = f" (superseded_by: {e['superseded_by']})" if e["superseded_by"] else ""
                lines.append(f"  - DEAD-END [{e['date']}] {e['approach']}{sup}")
            lines.append("")

    confirmed_dec = [d for d in decisions if d["state"] == "confirmed"][:20]
    if confirmed_dec:
        lines.append("## RECENT CONFIRMED DECISIONS")
        for d in confirmed_dec:
            tag = " (Mike-approved)" if d["mike_approved"] else ""
            lines.append(f"- [{d['date']}] {d['title']}{tag}")
        lines.append("")

    if pending_dec or pending_de:
        lines.append("## PENDING CAPTURE -- agent: review + formalize")
        for d in pending_dec[:15]:
            lines.append(f"- decision [{d['date']}] {d['title']}")
        for e in pending_de[:15]:
            lines.append(f"- dead-end [{e['date']}] {e['approach']}")
        lines.append("")

    if len(lines) > line_cap:
        tail = (
            f"... [TRUNCATED at {line_cap} lines; full content would be "
            f"{len(lines)} lines. Run `claude-mem capture-list` for the full set.]"
        )
        lines = lines[: line_cap - 1] + [tail]
    return "\n".join(lines) + "\n"


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="gen_decisions_state",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the target project root. REQUIRED. The "
             "claude-mem DB is read from <root>/.claude-mem/index.db and "
             "DECISIONS_STATE.md is written to <root>/ (without --print).",
    )
    ap.add_argument(
        "--memory-dir",
        required=True,
        help="Absolute path to the memory directory. REQUIRED. Accepted for "
             "the uniform harness hook interface; this generator reads only "
             "the claude-mem DB under --project-root.",
    )
    ap.add_argument("--print", action="store_true", dest="print_stdout",
                    help="print to stdout instead of writing DECISIONS_STATE.md")
    args = ap.parse_args(argv)

    project_root = Path(args.project_root).resolve()
    output_file = project_root / OUTPUT_NAME
    db = _db_path(project_root)
    if not db.is_file():
        return 0
    conn = sqlite3.connect(db)
    try:
        if not _has_tables(conn):
            return 0
        out = render(conn)
    finally:
        conn.close()
    if args.print_stdout:
        sys.stdout.write(out)
    else:
        output_file.write_text(out, encoding="utf-8")
        print(f"wrote {output_file.name} ({out.count(chr(10))} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
