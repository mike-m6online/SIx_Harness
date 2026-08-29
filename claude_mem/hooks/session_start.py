"""SessionStart hook (spec R4). Curated high-signal cross-session render:

  ## INVARIANTS          -- titles from memory files with `type: invariant`
                             (none exist yet as of Task 5; Task 8 creates
                             them -- the "none recorded" fallback renders
                             until then).
  ## RECENT CORRECTIONS  -- 5 genuine (non-harness) corrections, rotation-
                             aware: least-recently-shown first (v3
                             2026-08-19; the old ingested_at DESC order
                             served the same 5 forever).
  ## RECENT DECISIONS    -- 5 most recently CONFIRMED decisions.
  capture-triage debt    -- one line when pending decisions/dead-ends
                             await triage (absent at zero).
  one memory-health line -- the latest gate verdict, mirrored from the
                             memory_health hook-heartbeat row (the gate
                             itself runs as its own SessionStart hook).

Hard cap: 25 non-blank lines. Replaces the old static
`sw>=50 ORDER BY` query, which surfaced the identical 15 chunks every
session, 14 of them harness boilerplate (skill bodies / task
notifications ingested with role="user" and phrase-graded sw=100).

Novelty guard (v2 2026-07-03; v3 2026-08-19 hashes only the STABLE
content -- the memory-health line's embedded gate timestamp made every
digest unique, permanently disarming the check): the SHA256 of the
rendered block's stable lines is
stored in the index DB's `meta` table under `last_sessionstart_hash`,
alongside `last_sessionstart_session` (the session that received the
render) and `last_sessionstart_corpus_sig` -- the coherent triple the
health gate's check 9 reads. The frozen-corpus flag is appended only
when an identical render is served to a DIFFERENT session after the
corpus moved; identical re-renders inside one session (compact
boundaries -- ingestion is a SessionEnd event) and unchanged renders
over an idle corpus are expected and never flagged (v1 flagged bare
consecutive repetition and false-fired at every quiet compact).

The project's own gen_project_state.py hook (if present) handles
PROJECT_STATE.md; we add only the curated augmentation, not a
duplicate of it.
"""
from __future__ import annotations

import hashlib
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import sqlite_vec
import yaml

from claude_mem.capture import CaptureStore
from claude_mem.config import ProjectConfig
from claude_mem.filters import is_harness_content
from claude_mem.textutil import clip

MAX_RENDER_LINES = 25
_N_CORRECTIONS = 5
_N_DECISIONS = 5

# Per-correction line budget. Whole-word clip with ellipsis via
# textutil.clip -- the old bare [:180] slice cut mid-word with no
# truncation marker. Budget raised 180 -> 240 (2026-08-19): the
# corrections are the operator's own words and the extra 60 chars keep
# most one-sentence corrections intact.
_CORRECTION_HEAD_CHARS = 240

_UNCHANGED_FLAG = (
    "(unchanged since last session -- possible frozen corpus)"
)


def _memory_dir_for_project(project_root: Path) -> Path:
    """Per-project Claude Code memory dir: ~/.claude/projects/<slug>/memory.
    A separate function (not inlined) so tests can monkeypatch the
    lookup without needing a real ~/.claude/projects/ tree on disk."""
    proj_slug = (
        str(project_root).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    return Path.home() / ".claude" / "projects" / proj_slug / "memory"


def _parse_frontmatter(text: str) -> Optional[dict]:
    """Cheap frontmatter parse: the leading `---\\n...\\n---` YAML block,
    if present. Returns None for a file with no frontmatter or malformed
    YAML (never raises -- a bad memory file must not break SessionStart)."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    block = text[3:end]
    try:
        data = yaml.safe_load(block)
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _frontmatter_type(fm: dict) -> Optional[str]:
    """The frontmatter `type`, read from EITHER the top level OR a nested
    `metadata:` block. Memory files land in two legitimate shapes: authored
    files carry a top-level `type:`, but the memory-dir normalizer folds it
    into `metadata: {node_type, type, originSessionId}` (and slugifies
    `name`). The render must recognise both, so an invariant stays visible
    after normalization."""
    top = fm.get("type")
    if isinstance(top, str):
        return top
    meta = fm.get("metadata")
    if isinstance(meta, dict):
        nested = meta.get("type")
        if isinstance(nested, str):
            return nested
    return None


def _body_h1(text: str) -> Optional[str]:
    """The first `# ` heading in the body (after the frontmatter fence).
    This is the human-readable invariant title; the frontmatter `name` is
    slugified by the memory-dir normalizer (e.g. `invariant-pe-only`) and
    reads poorly, so the H1 is preferred when present."""
    lines = text.split("\n")
    start = 0
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            start = text.count("\n", 0, end) + 2  # first body line after fence
    for line in lines[start:]:
        s = line.strip()
        if s.startswith("# "):
            return s[2:].strip()
    return None


def _invariant_titles(memory_dir: Path) -> List[str]:
    """Titles of every memory file tagged `type: invariant` (top-level or
    under `metadata:`). The title is the body H1 when present (readable),
    else the frontmatter `name`, else the file stem. Malformed or unreadable
    files are skipped, never fatal."""
    if not memory_dir.is_dir():
        return []
    titles: List[str] = []
    for md in sorted(memory_dir.glob("*.md")):
        try:
            text = md.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        fm = _parse_frontmatter(text)
        if not fm or _frontmatter_type(fm) != "invariant":
            continue
        title = _body_h1(text) or str(fm.get("name") or md.stem)
        titles.append(title)
    return titles


def _recent_genuine_corrections(db_path: Path, limit: int) -> List[dict]:
    """The next `limit` genuine correction chunks, ROTATION-AWARE.

    Selection order is least-recently-shown first: `last_accessed ASC
    NULLS FIRST` (never-surfaced corrections lead), tie-broken by
    `ingested_at DESC` (freshest first within a rotation cohort). The
    existing _record_access bump below stamps last_accessed on exactly
    the surfaced rows, so the pool self-rotates render to render -- the
    pre-2026-08-19 `ingested_at DESC` order re-served the same 5 chunks
    forever (live evidence: access_count 60-84 on the top 5 while 3
    pool-mates sat at 0).

    No SQL LIMIT: the harness-content filter runs in Python (a content-
    origin check, not expressible as SQL -- pre-Task-1-migration chunks
    may still carry a stale is_correction flag), and harness rows are
    never access-bumped, so under a bounded over-fetch they would sit
    permanently at the NULLS-FIRST head and starve genuine rows behind
    them. Corrections are a small fraction of the corpus, so the full
    fetch is cheap.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        rows = conn.execute(
            """
            SELECT * FROM chunks
            WHERE is_correction = 1
            ORDER BY last_accessed ASC NULLS FIRST, ingested_at DESC
            """
        ).fetchall()
        out = []
        surfaced_ids = []
        for r in rows:
            d = dict(r)
            if is_harness_content(d["content"]):
                continue
            out.append(d)
            surfaced_ids.append(d["id"])
            if len(out) >= limit:
                break
        _record_access(conn, surfaced_ids)
        return out
    finally:
        conn.close()


def _record_access(conn: sqlite3.Connection, chunk_ids: List[str]) -> None:
    """Bump access_count + stamp last_accessed for the surfaced chunks.
    Swallows any error: injection correctness outranks usage bookkeeping."""
    if not chunk_ids:
        return
    try:
        now = datetime.now(timezone.utc).isoformat()
        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(
            f"""
            UPDATE chunks
            SET access_count = access_count + 1,
                last_accessed = ?
            WHERE id IN ({placeholders})
            """,
            (now, *chunk_ids),
        )
        conn.commit()
    except Exception:
        pass


def _recent_confirmed_decisions(db_path: Path, limit: int) -> List[dict]:
    store = CaptureStore(db_path)
    try:
        decisions = store.list_decisions(state="confirmed")
    finally:
        store.close()
    # list_decisions already orders by date DESC.
    return decisions[:limit]


def _pending_triage_counts(db_path: Path) -> Tuple[int, int]:
    """(pending decisions, pending dead-ends) awaiting capture-triage.
    Rendered as the capture-triage debt line so accumulating untriaged
    candidates are visible at session start instead of only via a manual
    capture-list."""
    store = CaptureStore(db_path)
    try:
        n_dec = len(store.list_decisions(state="pending"))
        n_de = len(store.list_dead_ends(state="pending"))
    finally:
        store.close()
    return n_dec, n_de


# The render's half of the mutual watch: the gate monitors every hook's
# heartbeat, and the render (an independent hook process) flags a gate
# whose own heartbeat has gone stale -- e.g. its SessionStart entry was
# removed from settings. Threshold mirrors the gate's own hook-heartbeat
# staleness bound (memory_health.HOOK_HEARTBEAT_MAX_AGE_DAYS), duplicated
# here because scripts/memory_health.py is not an importable package
# member and the render must never depend on it existing.
_GATE_STALE_DAYS = 2.0


def _health_line(telemetry_path: Path) -> str:
    """The latest gate result from the memory_health hook heartbeat.
    The gate itself runs as its own SessionStart hook (scripts/
    memory_health.py); this line mirrors its most recent verdict so the
    injected block is self-contained, and flags a gate that has stopped
    running entirely (see _GATE_STALE_DAYS). Never raises -- a missing/
    broken telemetry DB degrades to an explicit no-gate-run line, not a
    crash."""
    try:
        conn = sqlite3.connect(telemetry_path)
        try:
            row = conn.execute(
                "SELECT timestamp, ok, detail FROM hook_heartbeat "
                "WHERE hook = 'memory_health' ORDER BY id DESC LIMIT 1"
            ).fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        row = None
    if not row:
        return (
            "memory-health: no gate run recorded yet "
            "(scripts/memory_health.py runs at SessionStart)"
        )
    ts, ok, detail = row
    try:
        age_days = (
            datetime.now(timezone.utc) - datetime.fromisoformat(ts)
        ).total_seconds() / 86400.0
    except ValueError:
        age_days = None
    if age_days is not None and age_days > _GATE_STALE_DAYS:
        return (
            f"memory-health: GATE HAS NOT RUN for {age_days:.1f}d "
            f"(last {ts}: {detail}) -- check the SessionStart hook entry "
            f"for scripts/memory_health.py"
        )
    if not ok:
        return f"memory-health: LAST GATE RUN ERRORED at {ts} -- {detail}"
    return f"memory-health (last gate run {ts}): {detail}"


def _render_block(
    invariants: List[str], corrections: List[dict], decisions: List[dict],
    health_line: str, pending_decisions: int = 0, pending_dead_ends: int = 0,
) -> str:
    out = ["# claude-mem -- curated session-start context"]
    out.append("")
    out.append("## INVARIANTS")
    if invariants:
        for title in invariants:
            out.append(f"- {title}")
    else:
        out.append("- (none recorded)")
    out.append("")
    out.append("## RECENT CORRECTIONS")
    if corrections:
        for c in corrections:
            out.append(f"- {clip(c['content'], _CORRECTION_HEAD_CHARS)}")
    else:
        out.append("- (none recorded)")
    out.append("")
    out.append("## RECENT DECISIONS")
    if decisions:
        for d in decisions:
            out.append(f"- [{d['date']}] {d['title']}")
    else:
        out.append("- (none recorded)")
    if pending_decisions > 0 or pending_dead_ends > 0:
        # Capture-triage debt: untriaged candidates otherwise accumulate
        # invisibly (the 2026-08-19 triage found 119 rows pending).
        # Absent entirely at zero -- no line is the healthy state.
        out.append("")
        out.append(
            f"capture-triage debt: {pending_decisions} decisions + "
            f"{pending_dead_ends} dead-ends pending -- run "
            f"claude-mem capture-triage"
        )
    out.append("")
    out.append(health_line)
    return "\n".join(out)


def run(project_root: Path, session_id: str = "") -> str:
    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""

    invariants = _invariant_titles(_memory_dir_for_project(project_root))
    corrections = _recent_genuine_corrections(cfg.db_path, _N_CORRECTIONS)
    decisions = _recent_confirmed_decisions(cfg.db_path, _N_DECISIONS)
    pending_dec, pending_de = _pending_triage_counts(cfg.db_path)

    block = _render_block(
        invariants, corrections, decisions,
        _health_line(cfg.telemetry_path),
        pending_decisions=pending_dec, pending_dead_ends=pending_de,
    )

    # Hard cap: 25 non-blank lines. Truncate the LOWEST-priority content
    # first (decisions, then corrections) if seed data ever exceeds the
    # cap in a single section despite the per-section limits above (e.g.
    # a future higher N_CORRECTIONS/N_DECISIONS) -- belt-and-suspenders.
    lines = block.splitlines()
    non_blank = [l for l in lines if l.strip()]
    if len(non_blank) > MAX_RENDER_LINES:
        kept: List[str] = []
        non_blank_kept = 0
        for line in lines:
            if line.strip():
                if non_blank_kept >= MAX_RENDER_LINES:
                    continue
                non_blank_kept += 1
            kept.append(line)
        block = "\n".join(kept)

    block = _apply_novelty_guard(cfg.db_path, block, session_id)
    return block


def _corpus_sig(db_path: Path) -> str:
    """Corpus signature: SHA256 over the sorted incr:* watermark pairs plus
    the chunks and decisions row counts -- the row counts make the signature
    move under EVERY ingestion path, not only the incremental pipeline that
    rewrites incr:*. scripts/memory_health.py computes the same concept
    independently for its check 9 (the two signatures are never compared to
    each other; each store only ever compares its own history). Missing
    tables degrade to sentinel components -- always returns a hex string,
    never raises."""
    parts: List[str] = []
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.Error:
        conn = None
    if conn is None:
        parts.append("no-index")
    else:
        try:
            try:
                rows = conn.execute(
                    "SELECT key, value FROM meta WHERE key LIKE 'incr:%' "
                    "ORDER BY key"
                ).fetchall()
                parts.extend(f"{k}={v}" for k, v in rows)
            except sqlite3.Error:
                parts.append("no-meta")
            for table in ("chunks", "decisions"):
                try:
                    n = conn.execute(
                        f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.Error:
                    n = -1
                parts.append(f"{table}:{n}")
        finally:
            conn.close()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# The memory-health line is the ONE render line that varies per run
# without the curated content changing: three of its four variants embed
# the gate heartbeat's timestamp ("memory-health (last gate run
# 2026-...): ..."), which is fresh every SessionStart once the gate is
# installed. Hashing it made the novelty digest unique on EVERY render,
# so the frozen-render check could never fire on a live install (v3
# 2026-08-19 finding). Every variant opens with the literal
# "memory-health", so a single anchored prefix matches all of them.
_HEALTH_LINE_RE = re.compile(r"^memory-health\b")


def _stable_digest_source(block: str) -> str:
    """The render minus timestamp-bearing lines -- the input the novelty
    digest hashes. Currently only the memory-health line qualifies: the
    curated sections (invariants / corrections / decisions / triage-debt)
    contain dates only as stable per-row content, which is exactly what
    the digest SHOULD move with."""
    return "\n".join(
        line for line in block.splitlines()
        if not _HEALTH_LINE_RE.match(line)
    )


def _apply_novelty_guard(db_path: Path, block: str, session_id: str) -> str:
    """SHA256 the rendered block's STABLE content (the memory-health line
    is excluded -- see _stable_digest_source: its embedded gate timestamp
    changes every run, which kept the digest permanently novel and the
    frozen-render check permanently disarmed); store the (hash,
    receiving-session, corpus-signature) meta triple that the health
    gate's check 9 reads.

    The unchanged-flag is appended (AFTER hashing the original block, so the
    stored hash always reflects the underlying content) only for the genuine
    frozen-corpus symptom: an identical render served to a DIFFERENT session
    after the corpus moved. Identical re-renders inside one session are the
    normal compact-boundary case (ingestion is a SessionEnd event), and an
    unchanged render over an idle corpus is legitimate -- neither is flagged
    (v2 2026-07-03; v1 flagged bare consecutive repetition and false-fired
    at every quiet compact). Unknown session identity on either side means
    cross-session repetition cannot be asserted -- no flag."""
    digest = hashlib.sha256(
        _stable_digest_source(block).encode("utf-8")
    ).hexdigest()
    corpus = _corpus_sig(db_path)
    store = CaptureStore(db_path)
    try:
        prev = store.get_meta("last_sessionstart_hash")
        prev_session = store.get_meta("last_sessionstart_session") or ""
        prev_corpus = store.get_meta("last_sessionstart_corpus_sig") or ""
        store.set_meta("last_sessionstart_hash", digest)
        store.set_meta("last_sessionstart_session", session_id)
        store.set_meta("last_sessionstart_corpus_sig", corpus)
    finally:
        store.close()
    if (
        prev == digest
        and session_id and prev_session
        and prev_session != session_id
        and prev_corpus and prev_corpus != corpus
    ):
        return block + "\n\n" + _UNCHANGED_FLAG
    return block
