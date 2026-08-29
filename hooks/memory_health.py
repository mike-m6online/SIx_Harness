#!/usr/bin/env python3
"""The Memory-Health Gate (spec 2026-07-02 section 4, Task 7).

The watchdog that makes silent memory-system death structurally impossible.
Four generations of memory tooling on the source project (Virgil's watcher,
agent-army, the coop supervisor, gen_decisions_state) died UNMONITORED -- a
cp1252 crash went unnoticed for a week, an ingestion pipeline silently
froze, a hook stopped firing and nobody knew until context rot showed up in
a session. This gate ends that pattern: a fixed roster of pre-registered
checks (11 today), each a named invariant with a threshold, run every
SessionStart.

Design contract (all binding):
  * REPORT-ONLY. Always exits 0 once the flags parse -- it NEVER blocks
    Claude Code's turn. A gate that could crash the harness would itself
    become the silent-death it was built to prevent (spec risk section 9).
  * READ-ONLY on every OBSERVED store (index.db, MEMORY.md, module_states,
    graphify, CLAUDE.md). The ONLY writes it performs are to its OWN
    bookkeeping in telemetry.db: (a) its own hook_heartbeat row -- so check
    #3 covers the gate itself next session -- and (b) a memhealth_novelty
    row recording the observed session-start render hash, which is how
    check #9 detects a frozen corpus across sessions without mutating
    index.db. Neither write touches any store the checks observe.
  * Pure stdlib + sqlite3 (+ sqlite_vec loaded DEFENSIVELY for check #2).
    An sqlite_vec load failure is a RED check, never a crash.
  * Every check returns (name, ok, detail, fix_hint). An unknown or missing
    data source renders RED, never an exception.
  * All thresholds are module-level constants with the spec's exact values.

Output:
  * All pass:  one line  `MEMORY-HEALTH: N/N GREEN` (N = total checks,
    recalculated from the registered roster, never hardcoded)
  * Otherwise: a red block: `MEMORY-HEALTH: G/N GREEN (M red)` followed by
    one `RED <name>: <detail>  FIX: <hint>` line per failing check.

The checks (spec section 4 table -- Red when ...):
  1  ingest watermark age               > WATERMARK_MAX_AGE_DAYS (3)
  2  vector coverage chunks_vec/chunks  < VECTOR_COVERAGE_MIN (0.99)
  3  hook heartbeats                    boundary-consistency contradiction
                                        (v2 2026-08-19: per-turn hooks keep
                                        the 2-day wall clock; boundary hooks
                                        are judged against boundary EVIDENCE,
                                        never wall clock -- see
                                        check_hook_heartbeats)
  4  pending capture queue depth        > PENDING_QUEUE_MAX (40)
  5  lineage cache age vs thread        cache older than thread last_updated
  6  MODULE.state auto_derived max age  > MODULE_STATE_MAX_AGE_DAYS (14)
  7  graphify corpus age                > GRAPHIFY_MAX_AGE_DAYS (21) / missing
  8  MEMORY.md budget + anchor age      over budget, or anchor >
                                        LATEST_ANCHOR_MAX_AGE_DAYS (7) old
  9  session-start injection novelty    identical render served to
                                        NOVELTY_DISTINCT_SESSION_REPEATS+1
                                        distinct sessions while the corpus
                                        advanced (v2 2026-07-03; within-
                                        session compact repeats are expected)
  10 CLAUDE.md deprecated-marker scan   any registered deprecated tool name
                                        still present
  11 embedding path end-to-end probe    Ollama unreachable / embed model
                                        missing from /api/tags / embed
                                        probe timeout (2026-08-19; each RED
                                        carries its matching fix hint)

Harness-kit parameterization (the ONLY deltas from the origin-project
original):
  * `--project-root` and `--memory-dir` are REQUIRED flags; omitting either
    exits 2 with argparse's standard clear message. The harness init tool
    bakes concrete absolute paths into the installed hook command, so
    runtime discovery is unnecessary and no default exists.
  * index.db / telemetry.db resolve to `<project-root>/.claude-mem/`,
    CLAUDE.md to `<project-root>/CLAUDE.md`, and the module-states dir to
    `<project-root>/docs/marathon/module_states` (overridable via
    `--module-states-dir`) -- the same project-relative layout the
    original derived from its own repo location.
  * graphify corpus dirs were absolute user-profile paths in the original;
    they are now supplied via repeatable `--graphify-dir` flags (first
    EXISTING dir wins, preserving the original preference-order semantics).
    With no flag given, check 7 renders its original "no graphify out-dir
    found" RED.
  * check 10's deprecated-token list named the origin project's retired
    tooling; tokens are now registered via repeatable
    `--deprecated-token` flags. With none registered the scan finds
    nothing and the check is green -- exactly the original's behavior once
    its CLAUDE.md was decontaminated.
  * fix-hint strings that embedded the origin project's absolute path are
    built from the `--project-root` value; hints that named origin-only
    artifacts are rephrased generically. Check logic, thresholds,
    ordering, and the render format are byte-identical to the original.
  * two documented micro-deltas: check 1's redundant in-loop `import json`
    is dropped (json is module-level in both versions; behavior identical),
    and `_record_novelty_hash` also catches OSError from its mkdir -- the
    original's stated "never raises" contract for the novelty bookkeeping
    leaked OSError on an unwritable telemetry dir, which run_checks would
    have converted into a spurious check-9 "check raised" RED.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import socket
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Thresholds -- spec section 4 table, exact values, module-level constants.
# --------------------------------------------------------------------------
WATERMARK_MAX_AGE_DAYS = 3          # check 1
VECTOR_COVERAGE_MIN = 0.99          # check 2 (99%)
HOOK_HEARTBEAT_MAX_AGE_DAYS = 2     # check 3 (per-turn wall clock + the
                                    # turn-after-start evidence span)
PENDING_QUEUE_MAX = 40              # check 4 (> 40 -> TRIAGE OWED)
# check 3 (v2 2026-08-19): SessionEnd-family consistency window. The
# SessionEnd stack (session_end -> capture_extract -> capture_synthesize)
# runs as one hook array within seconds; a member that has not fired
# within 1h of the family's latest boundary event missed the boundary.
# The same window doubles as the "opportunity" grace: a boundary younger
# than 1h has not yet PROVEN a member dead (pending, not RED).
SESSION_END_FAMILY_WINDOW_S = 3600
# check 3 (v2 2026-08-19): same-boundary tolerance. Two uses, one cause
# (events of ONE session boundary interleave within minutes):
#   * the gate is ordered before claude-mem session-start in the
#     SessionStart array, so at the boundary it runs on, the session_start
#     heartbeat for THIS boundary does not exist yet -- a boundary-cycle
#     contradiction whose session_end is younger than this grace is
#     PENDING, not stale;
#   * the previous session's SessionEnd stack and the new session's
#     SessionStart interleave (observed live 2026-08-19: session_start
#     01:35:09 vs session_end 01:35:11), so a session_end heartbeat this
#     close to the last session_start is the SAME boundary, not evidence
#     that a later boundary went unstarted.
SAME_BOUNDARY_GRACE_S = 600
MODULE_STATE_MAX_AGE_DAYS = 14      # check 6
GRAPHIFY_MAX_AGE_DAYS = 21          # check 7
MEMORY_MAX_LINES = 280             # check 8 -- mirrors memory_decay.DEFAULT_MAX_LINES
MEMORY_MAX_BYTES = 90000           # check 8 -- mirrors memory_decay.DEFAULT_MAX_BYTES
LATEST_ANCHOR_MAX_AGE_DAYS = 7     # check 8
# check 9 (v2): distinct PRIOR receiving-sessions that must have been served
# the identical render (with corpus movement) before RED. Calibration: one
# cross-session repetition can be a trivial session that ingested nothing
# render-visible; the frozen-render pathology is persistent, so requiring a
# second distinct session costs one boundary of latency and removes the
# dominant benign trigger.
NOVELTY_DISTINCT_SESSION_REPEATS = 2
# check 9 (2026-07-04): consecutive all-empty-session observations before the
# frozen-render guard counts as structurally DISARMED. session_start derives a
# session id from session_id or transcript_path; if the guard's own history
# shows this many recent observations that ALL failed to carry one, the
# session-start plumbing is broken and the guard can never fire -- RED, not the
# silent 'session unknown' green. Below this count is warm-up (bootstrap-safe).
NOVELTY_DISARM_MIN = 3
CLAUDE_MD_DEPRECATED_MAX_AGE_DAYS = 30  # check 10 (documented; no dated markers today)
# check 11 (2026-08-19): end-to-end embedding-path probe. The embed vector
# leg of hybrid search died silently for an unknown period (model missing
# from Ollama entirely; search.py degraded to BM25-only with zero
# telemetry). This check makes that class of death visible at every
# SessionStart, discriminating the three live failure modes:
#   * Ollama server unreachable          -> "start Ollama"
#   * embedding model missing (/api/tags) -> "ollama pull <model>"
#   * probe timeout (model cold/loading)  -> "retry -- model loading"
# The probe carries the same keep_alive + num_ctx the real embed path
# sends: keep_alive because every request re-arms Ollama's residency
# timer (a probe omitting it would reset residency to the 5m server
# default), num_ctx because Ollama RELOADS a model whose requested
# context size differs from the loaded instance -- a mismatched probe
# would thrash model loads instead of observing them.
EMBED_PROBE_TAGS_TIMEOUT_S = 2.0    # /api/tags inventory query
EMBED_PROBE_READ_TIMEOUT_S = 5.0    # end-to-end embed (warm ~0.4s, cold ~2s)
# One-shot cold-start warm-up window (2026-08-29): after an idle gap the
# FIRST embed pays the whole model load and can exceed the probe budget;
# a RED whose only fix hint is "retry" is alert fatigue, not signal. On a
# probe timeout the check retries ONCE with this window before its
# verdict -- a genuinely dead server still fails both and stays RED.
EMBED_WARM_RETRY_TIMEOUT_S = 30.0
EMBED_PROBE_NUM_CTX = 8192          # = claude_mem.embed.EmbeddingClient.embed_num_ctx
# 127.0.0.1 literal, NOT "localhost" -- mirrors config.py DEFAULT_CONFIG:
# Windows getaddrinfo orders ::1 before 127.0.0.1 while Ollama binds
# IPv4-only, and the doomed IPv6 connect attempt eats the probe budget
# (measured 2026-08-19 on the origin project).
DEFAULT_OLLAMA_ENDPOINT = "http://127.0.0.1:11434"   # config.py default
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"         # config.py default
DEFAULT_EMBED_KEEP_ALIVE = "4h"                      # config.py default

# check 3: every hook that MUST leave a success heartbeat, partitioned by
# WHEN it can legitimately fire (v2 2026-08-19 boundary-consistency
# semantics; see check_hook_heartbeats):
#   * PER_TURN_HOOKS fire on every prompt/tool turn -- wall-clock
#     staleness is valid evidence for them and they anchor "the session
#     is alive" for the boundary rules.
#   * session_start fires only at SessionStart boundaries (start / resume
#     / compact); under marathon usage none may occur for days.
#   * SESSION_END_FAMILY fires only at SessionEnd boundaries.
EXPECTED_HOOKS = (
    "session_start",
    "prompt_submit",
    "tool_use",
    "tool_use_post",
    "capture_extract",
    "capture_synthesize",
    "session_end",
    # ingestion runs inside capture_extract's heartbeat; covered there
)
PER_TURN_HOOKS = ("prompt_submit", "tool_use", "tool_use_post")
SESSION_START_HOOK = "session_start"
SESSION_END_FAMILY = ("capture_extract", "capture_synthesize", "session_end")

# check 8: the LATEST-anchor is the first memory-file link target under the
# `## ⏯ LATEST` header. Its mtime is the freshness signal for "is the resume
# state current."
_LATEST_HEADER_RE = re.compile(r"^#{1,4}\s*.*LATEST", re.IGNORECASE)
_MD_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")

# The gate's own heartbeat identity in telemetry.db.
SELF_HOOK_NAME = "memory_health"

# Project-relative locations of the stores the gate observes. The harness
# convention places the claude-mem DBs and CLAUDE.md at the project root;
# the module-states dir default preserves the source project's layout and
# is overridable via --module-states-dir.
CLAUDE_MEM_DIRNAME = ".claude-mem"
INDEX_DB_NAME = "index.db"
TELEMETRY_DB_NAME = "telemetry.db"
CLAUDE_MEM_CONFIG_NAME = "config.yaml"
CLAUDE_MD_NAME = "CLAUDE.md"
DEFAULT_MODULE_STATES_RELPATH = Path("docs") / "marathon" / "module_states"


@dataclass(frozen=True)
class HealthPaths:
    """Every data source the gate reads, injected so tests point it at
    isolated fixtures. graphify_dirs is ordered: the first that EXISTS is
    used (preference order preserved from the CLI). project_root is carried
    so fix hints can name the concrete `claude-mem ... --project-root`
    command for THIS project; deprecated_tokens is the operator-registered
    check-10 scan list."""

    project_root: Path
    index_db: Path
    telemetry_db: Path
    memory_dir: Path
    module_states_dir: Path
    graphify_dirs: Sequence[Path]
    claude_md: Path
    deprecated_tokens: Sequence[str] = ()
    # check 11 overrides (--ollama-endpoint / --embed-model). None means
    # "resolve from <project-root>/.claude-mem/config.yaml, else the
    # claude-mem defaults" -- see _resolve_embed_settings.
    ollama_endpoint: Optional[str] = None
    embed_model: Optional[str] = None

    @staticmethod
    def from_cli(
        project_root: Path,
        memory_dir: Path,
        module_states_dir: Optional[Path] = None,
        graphify_dirs: Sequence[Path] = (),
        deprecated_tokens: Sequence[str] = (),
        ollama_endpoint: Optional[str] = None,
        embed_model: Optional[str] = None,
    ) -> "HealthPaths":
        """Build the gate's path set from the baked hook flags. Everything
        except memory_dir / graphify_dirs resolves project-root-relative."""
        return HealthPaths(
            project_root=project_root,
            index_db=project_root / CLAUDE_MEM_DIRNAME / INDEX_DB_NAME,
            telemetry_db=project_root / CLAUDE_MEM_DIRNAME / TELEMETRY_DB_NAME,
            memory_dir=memory_dir,
            module_states_dir=(
                module_states_dir
                if module_states_dir is not None
                else project_root / DEFAULT_MODULE_STATES_RELPATH
            ),
            graphify_dirs=tuple(graphify_dirs),
            claude_md=project_root / CLAUDE_MD_NAME,
            deprecated_tokens=tuple(deprecated_tokens),
            ollama_endpoint=ollama_endpoint,
            embed_model=embed_model,
        )


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str
    fix_hint: str


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------
def _now_ts() -> float:
    return time.time()


def _root_disp(paths: HealthPaths) -> str:
    """The project root as a forward-slash string for fix-hint commands
    (the form `claude-mem --project-root <root>` expects on any OS)."""
    return paths.project_root.as_posix()


def _parse_iso(ts: str) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp (with or without tz). Returns None on
    failure -- callers treat None as 'unknown', which drives the check RED."""
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _age_days_from_iso(ts: str) -> Optional[float]:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0


def _connect_ro(db_path: Path) -> Optional[sqlite3.Connection]:
    """Open a read-only connection to an existing sqlite file. Returns None
    if the file is absent or unopenable -- callers render RED."""
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error:
        return None


def _read_text(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    try:
        with io.open(path, "r", encoding="utf-8") as f:
            text = f.read()
    except OSError:
        return None
    if text.startswith("﻿"):
        text = text[1:]
    return text


def _ensure_sqlite_vec_on_path() -> None:
    """Best-effort: add the user site-packages to sys.path so `import
    sqlite_vec` resolves even when the hook launches under a bare python.
    Never raises."""
    try:
        import site
        for p in (site.getusersitepackages(),):
            if p and p not in sys.path:
                sys.path.append(p)
    except Exception:
        pass


# --------------------------------------------------------------------------
# The 10 checks. Each: (paths) -> CheckResult, never raises.
# --------------------------------------------------------------------------
def check_ingest_watermark(paths: HealthPaths) -> CheckResult:
    """Check 1: the freshest ingest watermark must be <= 3 days old. The
    incr:* meta keys store the newest ingested FILE mtime per source; the
    MAX across sources is 'how recently ingestion processed anything.' If the
    freshest watermark is stale, ingestion has silently stopped (or nothing
    new exists -- either way the operator should look). A missing-index or
    no-watermarks state is RED (ingestion never ran)."""
    name = "ingest_watermark_age"
    fix = (f"run `claude-mem ingest-incremental --project-root "
           f"{_root_disp(paths)}`; "
           "if it errors, the SessionEnd hook stack is broken (see check 3)")
    conn = _connect_ro(paths.index_db)
    if conn is None:
        return CheckResult(name, False, "index.db missing/unreadable", fix)
    try:
        rows = conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'incr:%'"
        ).fetchall()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"meta read failed: {exc}", fix)
    finally:
        conn.close()
    mtimes: List[float] = []
    for r in rows:
        val = r["value"]
        mt: Optional[float] = None
        # value is either a JSON object {"mtime": ...} (docs/memory/ledgers)
        # or a bare byte-offset string (sessions). Only the JSON form carries
        # a wall-clock mtime; byte offsets are not ages.
        try:
            obj = json.loads(val)
            if isinstance(obj, dict) and "mtime" in obj:
                mt = float(obj["mtime"])
        except (ValueError, TypeError):
            mt = None
        if mt is not None:
            mtimes.append(mt)
    if not mtimes:
        return CheckResult(
            name, False,
            "no incr:* watermarks with an mtime (ingestion never ran)", fix,
        )
    freshest = max(mtimes)
    age = (_now_ts() - freshest) / 86400.0
    ok = age <= WATERMARK_MAX_AGE_DAYS
    detail = (f"freshest watermark {age:.1f}d old "
              f"(max {WATERMARK_MAX_AGE_DAYS}d; {len(mtimes)} source(s))")
    return CheckResult(name, ok, detail, fix)


def check_vector_coverage(paths: HealthPaths) -> CheckResult:
    """Check 2: chunks_vec/chunks coverage must be >= 99%. A chunk without a
    vector is invisible to semantic search. Requires the sqlite_vec extension
    to read the vec0 virtual table; an extension-load failure is itself a
    health violation (RED), not a crash."""
    name = "vector_coverage"
    fix = (f"run `claude-mem embed-backfill --project-root "
           f"{_root_disp(paths)}`; "
           "if sqlite_vec fails to load, reinstall it into the hook's python")
    if not paths.index_db.is_file():
        return CheckResult(name, False, "index.db missing", fix)
    _ensure_sqlite_vec_on_path()
    try:
        import sqlite_vec  # noqa: F401  (imported for side-effect availability)
    except Exception as exc:
        return CheckResult(
            name, False, f"sqlite_vec unavailable: {exc}", fix,
        )
    try:
        conn = sqlite3.connect(f"file:{paths.index_db}?mode=ro", uri=True)
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"open failed: {exc}", fix)
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        n_chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        n_vec = conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
    except (sqlite3.Error, AttributeError) as exc:
        return CheckResult(name, False, f"vec read failed: {exc}", fix)
    finally:
        conn.close()
    if n_chunks == 0:
        return CheckResult(name, False, "0 chunks (empty index)", fix)
    coverage = n_vec / n_chunks
    ok = coverage >= VECTOR_COVERAGE_MIN
    detail = (f"{n_vec}/{n_chunks} = {coverage*100:.1f}% "
              f"(min {VECTOR_COVERAGE_MIN*100:.0f}%)")
    return CheckResult(name, ok, detail, fix)


def _fmt_age(delta: timedelta) -> str:
    """Compact age rendering for heartbeat evidence: '3.2d' / '5.1h' / '4m'."""
    s = delta.total_seconds()
    if s >= 86400:
        return f"{s / 86400.0:.1f}d"
    if s >= 3600:
        return f"{s / 3600.0:.1f}h"
    return f"{max(s, 0) / 60.0:.0f}m"


def _gather_heartbeats(
    conn: sqlite3.Connection,
) -> Dict[str, Tuple[Optional[int], Optional[datetime]]]:
    """Per expected hook: (latest row's ok flag or None if no rows,
    datetime of the newest SUCCESS row or None). Raises sqlite3.Error to
    the caller (which renders RED, never crashes)."""
    out: Dict[str, Tuple[Optional[int], Optional[datetime]]] = {}
    for hook in EXPECTED_HOOKS:
        latest = conn.execute(
            "SELECT ok FROM hook_heartbeat WHERE hook=? "
            "ORDER BY id DESC LIMIT 1",
            (hook,),
        ).fetchone()
        latest_ok = None if latest is None else int(latest["ok"])
        last_ok_ts = conn.execute(
            "SELECT MAX(timestamp) FROM hook_heartbeat WHERE hook=? AND ok=1",
            (hook,),
        ).fetchone()[0]
        out[hook] = (latest_ok, _parse_iso(last_ok_ts) if last_ok_ts else None)
    return out


def check_hook_heartbeats(paths: HealthPaths) -> CheckResult:
    """Check 3 (the centerpiece): a silently-dead hook must become visible
    the NEXT session instead of never (the cp1252 crash class).

    v2 SEMANTICS (2026-08-19, boundary-consistency): v1 required EVERY
    hook to carry a success heartbeat <= 2 days old, which misfires BY
    DESIGN under marathon usage -- one Claude Code session can run for
    days without a single SessionStart/SessionEnd boundary, so the
    boundary hooks (session_start + the SessionEnd family) go "stale" on
    the wall clock while the per-turn hooks prove the stack is alive.
    Staleness of a hook that only fires at boundaries is NOT evidence of
    death when no boundary occurred. v2 keeps wall-clock staleness only
    where it is valid evidence and judges boundary hooks against boundary
    EVIDENCE:

      * PER_TURN_HOOKS (prompt_submit / tool_use / tool_use_post):
        unchanged v1 rules -- no rows, latest row errored, or last
        success > HOOK_HEARTBEAT_MAX_AGE_DAYS -> RED. These fire every
        turn, so wall clock is valid, and they anchor "the session is
        alive" for the rules below.
      * session_start: RED only on a boundary-cycle contradiction --
        per-turn activity exists more than the max-age NEWER than the
        last session_start success AND a session_end success exists
        meaningfully after that session_start (> SAME_BOUNDARY_GRACE_S,
        because one boundary's SessionEnd/SessionStart events interleave
        within seconds) AND per-turn activity continued AFTER that
        session_end (a new session actually ran). Same-boundary
        tolerance: the gate is ordered BEFORE claude-mem session-start
        in the SessionStart array, so a contradiction whose session_end
        is younger than SAME_BOUNDARY_GRACE_S is PENDING (this very
        boundary's session_start heartbeat has not been written yet),
        not stale. An old-but-uncontradicted session_start with ongoing
        per-turn activity is GREEN with the informational note
        'long-lived session; boundary hooks idle by design'.
      * SESSION_END_FAMILY (session_end / capture_extract /
        capture_synthesize): compared against the family's OWN boundary
        events, never wall clock. capture_* is RED only if session_end
        fired without it within SESSION_END_FAMILY_WINDOW_S (1h);
        session_end is RED only if capture_* fired without it within the
        same window (both directions of the consistency). A boundary
        younger than the window has not proven a member dead -> pending.
        A family that has NEVER fired leaves no contradicting evidence
        and renders GREEN with a 'no session boundary observed yet' note
        (visible in the detail, not RED -- fresh installs would
        otherwise phantom-red until their first SessionEnd).

    A latest-row error stays RED for every hook (that detector is
    boundary-independent). GREEN details carry the informational notes."""
    name = "hook_heartbeats"
    fix = ("a hook stopped firing or crashed -- inspect telemetry.db "
           "hook_heartbeat + re-run `claude-mem install-hooks`")
    conn = _connect_ro(paths.telemetry_db)
    if conn is None:
        return CheckResult(name, False, "telemetry.db missing/unreadable", fix)
    try:
        beats = _gather_heartbeats(conn)
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"heartbeat read failed: {exc}", fix)
    finally:
        conn.close()

    now = datetime.now(timezone.utc)
    epoch = datetime.min.replace(tzinfo=timezone.utc)
    max_age = timedelta(days=HOOK_HEARTBEAT_MAX_AGE_DAYS)
    grace = timedelta(seconds=SAME_BOUNDARY_GRACE_S)
    window = timedelta(seconds=SESSION_END_FAMILY_WINDOW_S)
    problems: List[str] = []
    notes: List[str] = []

    # --- per-turn hooks: v1 wall-clock rules, unchanged -------------------
    for hook in PER_TURN_HOOKS:
        latest_ok, last_ok = beats[hook]
        if latest_ok is None:
            problems.append(f"{hook}:no-rows")
            continue
        if latest_ok != 1:
            problems.append(f"{hook}:latest-errored")
            continue
        if last_ok is None:
            problems.append(f"{hook}:bad-ts")
        elif (now - last_ok) > max_age:
            problems.append(f"{hook}:{_fmt_age(now - last_ok)}-stale")

    turn_successes = [beats[h][1] for h in PER_TURN_HOOKS if beats[h][1]]
    newest_turn = max(turn_successes) if turn_successes else None

    # --- session_start: boundary-cycle contradiction only -----------------
    start_latest_ok, last_start = beats[SESSION_START_HOOK]
    _end_latest_ok, last_end = beats["session_end"]
    if start_latest_ok is not None and start_latest_ok != 1:
        problems.append(f"{SESSION_START_HOOK}:latest-errored")
    else:
        start_floor = last_start or epoch
        contradiction = (
            newest_turn is not None
            and (newest_turn - start_floor) > max_age
            and last_end is not None
            and (last_end - start_floor) > grace
            and newest_turn > last_end
        )
        if contradiction:
            if (now - last_end) <= grace:
                notes.append(
                    "session_start pending (boundary in progress; the gate "
                    "runs before claude-mem session-start in the "
                    "SessionStart array)"
                )
            else:
                problems.append(
                    f"{SESSION_START_HOOK}:boundary-cycle-without-start "
                    f"(session_end {_fmt_age(now - last_end)} ago, turn "
                    f"activity {_fmt_age(now - newest_turn)} ago, last "
                    "start "
                    + ("never" if last_start is None
                       else _fmt_age(now - last_start) + " ago") + ")"
                )
        elif (last_start is not None and newest_turn is not None
                and (newest_turn - last_start) > max_age):
            notes.append("long-lived session; boundary hooks idle by design")
        elif last_start is None:
            notes.append(
                "session_start has no heartbeat yet (no contradicting "
                "boundary evidence; pending)"
            )

    # --- SessionEnd family: judged against its own boundary events --------
    capture_hooks = tuple(h for h in SESSION_END_FAMILY if h != "session_end")
    capture_successes = [beats[h][1] for h in capture_hooks if beats[h][1]]
    last_capture = max(capture_successes) if capture_successes else None

    for hook in capture_hooks:
        latest_ok, last_ok = beats[hook]
        if latest_ok is not None and latest_ok != 1:
            problems.append(f"{hook}:latest-errored")
            continue
        if last_end is None:
            continue  # no session_end boundary to contradict
        missed = (last_end - (last_ok or epoch)) > window
        if not missed:
            continue
        if (now - last_end) <= window:
            notes.append(
                f"{hook} pending (session_end fired "
                f"{_fmt_age(now - last_end)} ago; within the "
                f"{SESSION_END_FAMILY_WINDOW_S // 3600}h window)"
            )
        else:
            problems.append(
                f"{hook}:missed-boundary (session_end "
                f"{_fmt_age(now - last_end)} ago without {hook} within "
                f"{SESSION_END_FAMILY_WINDOW_S // 3600}h)"
            )

    end_latest_ok, _ = beats["session_end"]
    if end_latest_ok is not None and end_latest_ok != 1:
        problems.append("session_end:latest-errored")
    elif last_capture is not None:
        missed = (last_capture - (last_end or epoch)) > window
        if missed:
            if (now - last_capture) <= window:
                notes.append(
                    f"session_end pending (capture_* fired "
                    f"{_fmt_age(now - last_capture)} ago; within the "
                    f"{SESSION_END_FAMILY_WINDOW_S // 3600}h window)"
                )
            else:
                problems.append(
                    "session_end:missed-boundary (capture_* fired "
                    f"{_fmt_age(now - last_capture)} ago without "
                    "session_end within "
                    f"{SESSION_END_FAMILY_WINDOW_S // 3600}h)"
                )
    if last_end is None and last_capture is None:
        notes.append("no session boundary observed yet")

    ok = not problems
    if ok:
        detail = f"all {len(EXPECTED_HOOKS)} hooks consistent"
        if notes:
            detail += "; " + "; ".join(notes)
    else:
        detail = f"{len(problems)} problem(s): {', '.join(problems)}"
    return CheckResult(name, ok, detail, fix)


def check_pending_capture_queue(paths: HealthPaths) -> CheckResult:
    """Check 4: pending decisions must be <= 40. A deep pending queue means
    capture is mining candidates but nobody is triaging them (TRIAGE OWED)."""
    name = "pending_capture_queue"
    fix = (f"run `claude-mem capture-triage --project-root "
           f"{_root_disp(paths)}` and "
           "confirm/reject the backlog")
    conn = _connect_ro(paths.index_db)
    if conn is None:
        return CheckResult(name, False, "index.db missing/unreadable", fix)
    try:
        n = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE state='pending'"
        ).fetchone()[0]
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"decisions read failed: {exc}", fix)
    finally:
        conn.close()
    ok = n <= PENDING_QUEUE_MAX
    return CheckResult(
        name, ok, f"{n} pending (max {PENDING_QUEUE_MAX})", fix,
    )


def check_lineage_cache_age(paths: HealthPaths) -> CheckResult:
    """Check 5: for every thread with a cached lineage, the cache key (the
    thread last_updated the lineage was built against) must match the current
    last_updated. A cache older than its thread means synthesize has not
    caught up -> the injected lineage is stale."""
    name = "lineage_cache_age"
    fix = (f"run `claude-mem capture-synthesize --project-root "
           f"{_root_disp(paths)}` "
           "to regenerate stale thread lineages")
    conn = _connect_ro(paths.index_db)
    if conn is None:
        return CheckResult(name, False, "index.db missing/unreadable", fix)
    try:
        rows = conn.execute(
            "SELECT id, last_updated, lineage_cache_key FROM threads "
            "WHERE lineage_text IS NOT NULL"
        ).fetchall()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"threads read failed: {exc}", fix)
    finally:
        conn.close()
    stale = [r["id"] for r in rows
             if (r["lineage_cache_key"] or "") != (r["last_updated"] or "")]
    ok = not stale
    detail = (f"{len(rows)} cached lineage(s), all fresh" if ok else
              f"{len(stale)} stale: {', '.join(stale[:5])}")
    return CheckResult(name, ok, detail, fix)


def check_module_state_age(paths: HealthPaths) -> CheckResult:
    """Check 6: the auto_derived facts must have been (re)verified against
    the code within 14 days. The age is read from the generator's
    LAST_REGENERATED.json stamp (written on every full or
    --changed-only-verified pass), NOT from per-file mtimes: the generator
    deliberately skips rewriting unchanged state files, so a file whose
    facts were just re-verified can carry an arbitrarily old mtime -- an
    mtime-based check is a phantom red no regeneration can clear."""
    name = "module_state_age"
    fix = ("re-run the project's module-state generator (full or "
           "--changed-only) to re-verify auto_derived facts and refresh "
           f"{paths.module_states_dir.as_posix()}/LAST_REGENERATED.json")
    d = paths.module_states_dir
    if not d.is_dir():
        return CheckResult(name, False, f"{d} missing", fix)
    files = list(d.glob("*.state.yaml"))
    if not files:
        return CheckResult(name, False, "no *.state.yaml files", fix)
    stamp = d / "LAST_REGENERATED.json"
    try:
        payload = json.loads(stamp.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(str(payload["generated_at"]))
    except (OSError, ValueError, KeyError, TypeError,
            json.JSONDecodeError):
        return CheckResult(
            name, False,
            f"{len(files)} states but no readable LAST_REGENERATED.json "
            "stamp (generator has not completed a full pass since "
            "stamping landed)", fix,
        )
    age = (_now_ts() - ts.timestamp()) / 86400.0
    ok = age <= MODULE_STATE_MAX_AGE_DAYS
    return CheckResult(
        name, ok,
        f"last full verification {age:.1f}d ago ({len(files)} states; "
        f"max {MODULE_STATE_MAX_AGE_DAYS}d)", fix,
    )


def check_graphify_corpus_age(paths: HealthPaths) -> CheckResult:
    """Check 7: the graphify corpus (newest file under the first existing
    graphify out-dir) must be <= 21 days old. Missing entirely -> RED with a
    fix hint pointing at the --graphify-dir flag."""
    name = "graphify_corpus_age"
    missing_fix = ("graphify corpus absent -- point --graphify-dir at an "
                   "extracted corpus (or run the graphify extract for this "
                   "project)")
    stale_fix = ("re-run the graphify extract into a fresh out-dir and "
                 "update --graphify-dir")
    chosen: Optional[Path] = None
    for cand in paths.graphify_dirs:
        if cand.is_dir():
            chosen = cand
            break
    if chosen is None:
        return CheckResult(name, False, "no graphify out-dir found", missing_fix)
    files = [f for f in chosen.rglob("*") if f.is_file()]
    if not files:
        return CheckResult(name, False, f"{chosen} empty", missing_fix)
    newest = max(f.stat().st_mtime for f in files)
    age = (_now_ts() - newest) / 86400.0
    ok = age <= GRAPHIFY_MAX_AGE_DAYS
    return CheckResult(
        name, ok,
        f"{chosen.name} newest file {age:.1f}d old "
        f"(max {GRAPHIFY_MAX_AGE_DAYS}d)", stale_fix,
    )


def check_memory_md_health(paths: HealthPaths) -> CheckResult:
    """Check 8: MEMORY.md must be under the load budget (280 lines / 90KB,
    matching memory_decay's constants) AND the LATEST-anchor (the first
    memory-file link under the `## ⏯ LATEST` header) must have an mtime <= 7
    days old. Over budget OR a stale anchor -> RED."""
    name = "memory_md_health"
    fix = ("run the memory_decay hook with --print (budget) and update "
           "the `## ⏯ LATEST` anchor to the current checkpoint")
    mem_path = paths.memory_dir / "MEMORY.md"
    text = _read_text(mem_path)
    if text is None:
        return CheckResult(name, False, f"{mem_path} missing/unreadable", fix)
    n_lines = text.count("\n") + (0 if text.endswith("\n") or text == "" else 1)
    n_bytes = len(text.encode("utf-8"))
    over_lines = n_lines > MEMORY_MAX_LINES
    over_bytes = n_bytes > MEMORY_MAX_BYTES
    problems: List[str] = []
    if over_lines:
        problems.append(f"lines {n_lines}>{MEMORY_MAX_LINES}")
    if over_bytes:
        problems.append(f"bytes {n_bytes}>{MEMORY_MAX_BYTES}")
    # LATEST-anchor age
    anchor_target = _first_latest_anchor_link(text)
    if anchor_target is None:
        problems.append("no LATEST anchor link")
    else:
        anchor_path = paths.memory_dir / Path(anchor_target).name
        if not anchor_path.is_file():
            problems.append(f"anchor {anchor_target} missing")
        else:
            age = (_now_ts() - anchor_path.stat().st_mtime) / 86400.0
            if age > LATEST_ANCHOR_MAX_AGE_DAYS:
                problems.append(
                    f"anchor {age:.1f}d>{LATEST_ANCHOR_MAX_AGE_DAYS}d"
                )
    ok = not problems
    detail = (f"{n_lines}L/{n_bytes}B, anchor fresh" if ok
              else "; ".join(problems))
    return CheckResult(name, ok, detail, fix)


def _first_latest_anchor_link(memory_text: str) -> Optional[str]:
    """The first `](file.md)` link target appearing at/after the `## ⏯
    LATEST` header. Returns None if no LATEST header or no link follows."""
    lines = memory_text.split("\n")
    start = None
    for i, line in enumerate(lines):
        if _LATEST_HEADER_RE.match(line):
            start = i
            break
    if start is None:
        return None
    m = _MD_LINK_RE.search("\n".join(lines[start:]))
    return m.group(1) if m else None


def check_sessionstart_novelty(paths: HealthPaths) -> CheckResult:
    """Check 9 (v2, 2026-07-03): the session-start injection must not be
    served identically to DISTINCT sessions while the corpus advances (the
    frozen-RENDER symptom: curation stopped tracking a living corpus).

    v1 compared consecutive GATE RUNS, but the gate fires at every compact
    boundary inside one session, where ingestion (a SessionEnd event) cannot
    have advanced -- so v1 fired a false RED at every quiet compact
    (observed 2026-07-03; telemetry rows 10-17 show within-session hash
    runs). The spec's row for this check ("identical to previous SESSION")
    always meant per-session semantics; v2 conforms the implementation.

    The session_start hook writes a coherent meta pair every render:
    last_sessionstart_hash + last_sessionstart_session (the session that
    RECEIVED that render). Because the gate is ordered BEFORE session_start
    in the hook stack, it observes the pair one boundary in arrears --
    equality chains and session attribution are unaffected; only the firing
    boundary shifts by one. Do NOT "fix" the arrears by reading the gate's
    own stdin: the hash must stay paired with the session that actually
    received it. The gate stays read-only on index.db and keeps its own
    history in telemetry.db (memhealth_novelty), now with session_id and
    corpus_sig columns (legacy tables are ALTERed in place; pre-v2 rows
    carry no session evidence and never qualify).

    RED requires ALL of:
      * the current receiving-session is known, AND
      * the most recent observation from each of
        NOVELTY_DISTINCT_SESSION_REPEATS other distinct sessions carries the
        SAME render hash (same render served to >= 3 distinct sessions --
        a single repetition can be a trivial session that ingested nothing
        render-visible), AND
      * the corpus signature moved across the compared span (an idle corpus
        legitimately re-serves the same render; a DEAD pipeline is check 1's
        finding, not this one's).
    Missing index.db / missing hash (session-start never ran) stay RED."""
    name = "sessionstart_novelty"
    fix = ("session-start render is frozen while the corpus advances -- "
           "curation is stuck (see checks 1+3 first to rule out "
           "ingestion/hook death)")
    conn = _connect_ro(paths.index_db)
    if conn is None:
        return CheckResult(name, False, "index.db missing/unreadable", fix)
    try:
        row = conn.execute(
            "SELECT value FROM meta WHERE key='last_sessionstart_hash'"
        ).fetchone()
        srow = conn.execute(
            "SELECT value FROM meta WHERE key='last_sessionstart_session'"
        ).fetchone()
    except sqlite3.Error as exc:
        return CheckResult(name, False, f"meta read failed: {exc}", fix)
    finally:
        conn.close()
    if row is None or not row["value"]:
        return CheckResult(
            name, False, "no last_sessionstart_hash (session-start never ran)",
            fix,
        )
    current = row["value"]
    session = (srow["value"] or "") if srow is not None else ""
    corpus = _corpus_sig(paths.index_db)
    history = _novelty_rows(paths.telemetry_db)
    # record AFTER reading history so this observation is available next run
    _record_novelty_hash(
        paths.telemetry_db, current, session_id=session, corpus_sig=corpus)

    if not history:
        return CheckResult(name, True, "first observation", fix)
    if not session:
        # session_start ran (hash present) but wrote no session identity this
        # run. If it has ALSO failed to write identity across the most recent
        # NOVELTY_DISARM_MIN observations, the frozen-render guard is
        # structurally DISARMED (session-start plumbing broken -- e.g. a
        # SessionStart source carrying neither session_id nor transcript_path)
        # and must flag loudly rather than sit silently green (the 2026-07-04
        # dormant-guard gap). Fewer than that many all-empty observations is
        # warm-up; a recent armed observation means the guard works and this
        # run merely lacked identity -> GREEN.
        recent = history[:NOVELTY_DISARM_MIN]
        if (len(history) >= NOVELTY_DISARM_MIN
                and all(not s for _h, s, _sig in recent)):
            return CheckResult(
                name, False,
                "frozen-render guard disarmed -- session-start wrote no "
                f"session identity across the last {NOVELTY_DISARM_MIN} "
                "observations; the novelty check can never fire",
                "session-start is not recording session identity: verify the "
                "SessionStart hook passes session_id (or transcript_path) in "
                "its stdin JSON -- see claude_mem.cli.session_start",
            )
        return CheckResult(
            name, True,
            "receiving session unknown -- cross-session repetition not "
            "assertable (warming up)", fix,
        )
    latest_hash, latest_session, _latest_sig = history[0]
    same_session_note = (
        " (unchanged within same session -- expected at compact boundaries)"
        if latest_session == session and latest_hash == current else ""
    )
    # most recent observation per distinct OTHER receiving-session,
    # newest first; pre-v2 rows (empty session) carry no session evidence
    distinct: List[Tuple[str, str, str]] = []
    seen_sessions = set()
    for h, s, sig in history:
        if not s or s == session or s in seen_sessions:
            continue
        seen_sessions.add(s)
        distinct.append((h, s, sig))
        if len(distinct) == NOVELTY_DISTINCT_SESSION_REPEATS:
            break
    if len(distinct) < NOVELTY_DISTINCT_SESSION_REPEATS:
        return CheckResult(
            name, True,
            "insufficient cross-session history "
            f"({len(distinct)}/{NOVELTY_DISTINCT_SESSION_REPEATS} distinct "
            f"prior sessions){same_session_note}", fix,
        )
    all_repeat = all(h == current for h, _s, _sig in distinct)
    oldest_sig = distinct[-1][2]
    corpus_advanced = bool(oldest_sig) and oldest_sig != corpus
    if all_repeat and corpus_advanced:
        return CheckResult(
            name, False,
            "render identical across "
            f"{NOVELTY_DISTINCT_SESSION_REPEATS + 1} distinct sessions "
            f"while corpus advanced ({current[:12]})", fix,
        )
    detail = (f"corpus idle since prior sessions ({current[:12]})"
              if all_repeat
              else f"render tracking corpus ({current[:12]})")
    return CheckResult(name, True, detail + same_session_note, fix)


def check_claude_md_deprecated(paths: HealthPaths) -> CheckResult:
    """Check 10: CLAUDE.md must not still carry deprecated tool references.
    Implemented as a presence scan for the operator-registered deprecated
    symbols (repeatable --deprecated-token flags baked into the hook command
    when tooling is retired). With no tokens registered the scan finds
    nothing and the check is green.

    (The CLAUDE_MD_DEPRECATED_MAX_AGE_DAYS=30 constant documents the spec's
    'block older than 30 days' framing for when dated markers do land.)"""
    name = "claude_md_deprecated"
    fix = ("remove the deprecated tool references from CLAUDE.md (or adjust "
           "the --deprecated-token list this gate was installed with)")
    text = _read_text(paths.claude_md)
    if text is None:
        return CheckResult(name, False, f"{paths.claude_md} missing", fix)
    found = [tok for tok in paths.deprecated_tokens if tok in text]
    ok = not found
    detail = ("no deprecated tokens present" if ok
              else f"{len(found)} deprecated token(s): {', '.join(found)}")
    return CheckResult(name, ok, detail, fix)


# --------------------------------------------------------------------------
# check 11: end-to-end embedding-path probe (2026-08-19). Pure stdlib
# (urllib); yaml is imported DEFENSIVELY like sqlite_vec, with a flat-line
# fallback parser for the simple `key: value` shape ProjectConfig.write
# emits.
# --------------------------------------------------------------------------
def _read_claude_mem_config(config_path: Path) -> Dict[str, object]:
    """Best-effort read of the project's .claude-mem/config.yaml. Uses
    PyYAML when importable; otherwise falls back to parsing flat
    top-level `key: value` lines (splitting on the FIRST colon, so values
    like `qwen3-embedding:0.6b` survive). Returns {} on any failure --
    the caller then uses the claude-mem defaults. Never raises."""
    text = _read_text(config_path)
    if text is None:
        return {}
    try:
        import yaml  # defensive third-party import (see module docstring)
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
        return {}
    except Exception:
        pass
    out: Dict[str, object] = {}
    for line in text.splitlines():
        if not line or line[0] in "#\t " or ":" not in line:
            continue
        key, _, value = line.partition(":")
        v = value.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        if v == "" or v.lower() in ("null", "~"):
            continue
        out[key.strip()] = v
    return out


def _resolve_embed_settings(paths: HealthPaths) -> Tuple[str, str, str]:
    """(endpoint, model, keep_alive) for check 11: CLI flag overrides win,
    then the project's .claude-mem/config.yaml, then the claude-mem
    defaults (config.py DEFAULT_CONFIG values, mirrored as module
    constants here)."""
    cfg = _read_claude_mem_config(
        paths.project_root / CLAUDE_MEM_DIRNAME / CLAUDE_MEM_CONFIG_NAME
    )
    endpoint = (paths.ollama_endpoint
                or str(cfg.get("ollama_endpoint") or "")
                or DEFAULT_OLLAMA_ENDPOINT)
    model = (paths.embed_model
             or str(cfg.get("embedding_model") or "")
             or DEFAULT_EMBED_MODEL)
    keep_alive = (str(cfg.get("embedding_keep_alive") or "")
                  or DEFAULT_EMBED_KEEP_ALIVE)
    return endpoint, model, keep_alive


def _probe_url(
    url: str, timeout: float, payload: Optional[dict] = None,
) -> Tuple[str, object]:
    """Fetch `url` with stdlib urllib (GET, or POST when `payload` is
    given), CLASSIFYING the failure mode instead of raising:

      ("ok", parsed_json)            2xx with a JSON body
      ("http_error", (status, head)) non-2xx HTTP response
      ("timeout", detail)            no response within `timeout` seconds
      ("unreachable", detail)        connection refused / DNS / socket error
      ("malformed", detail)          2xx but the body is not JSON

    Never raises -- check 11's discrimination contract depends on these
    five outcomes and nothing else."""
    data: Optional[bytes] = None
    headers: Dict[str, str] = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read()
    except urllib.error.HTTPError as exc:
        try:
            head = exc.read()[:200].decode("utf-8", "replace")
        except Exception:
            head = ""
        return ("http_error", (exc.code, head))
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            return ("timeout", f"no response within {timeout:.1f}s")
        return ("unreachable", str(reason))
    except (TimeoutError, socket.timeout):
        return ("timeout", f"no response within {timeout:.1f}s")
    except OSError as exc:
        return ("unreachable", str(exc))
    try:
        return ("ok", json.loads(body.decode("utf-8")))
    except (ValueError, UnicodeDecodeError) as exc:
        return ("malformed", f"non-JSON response: {exc}")


def _cfg_explicit_embedding_dim(cfg: Dict[str, object]) -> Optional[int]:
    """The project's pinned vector width, or None when config.yaml does
    not set one. Dimension validation runs ONLY against an explicit pin:
    the index table's true width is a property of the database, and
    guessing it from library defaults would manufacture false REDs on
    projects that never wrote the key."""
    dim = cfg.get("embedding_dim")
    if dim is None:
        return None
    try:
        return int(dim)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _fallback_dim_verdict(
    ep: str, cfg: Dict[str, object], installed: set,
) -> Optional[Tuple[str, str]]:
    """Return (detail, fix_hint) when the configured embedding_fallback is
    structurally unable to serve, else None. The 2026-08-25 shakedown
    lesson: a fallback whose vector width differs from the index table can
    NEVER succeed -- the embed dimension guard refuses its vectors -- yet
    it silently taxes every failure path with an extra model load and a
    second timeout chain. Validate the configuration at gate time via
    /api/show (model metadata only; loads nothing), not at failure time.
    Metadata that cannot be fetched or parsed is 'cannot validate', never
    a RED."""
    fallback = cfg.get("embedding_fallback")
    if not fallback or not isinstance(fallback, str):
        return None
    # The no-PyYAML fallback parser yields literal strings; a config
    # that says `embedding_fallback: null` means NO fallback, never a
    # model named "null".
    if fallback.strip().lower() in ("null", "none", "~"):
        return None
    wanted = {fallback}
    if ":" not in fallback:
        wanted.add(fallback + ":latest")
    if installed and not (wanted & installed):
        return (
            f"embedding_fallback {fallback} is not installed in Ollama -- "
            "it can never serve",
            f"run `ollama pull {fallback}` or set embedding_fallback: null",
        )
    dim = _cfg_explicit_embedding_dim(cfg)
    if dim is None:
        return None
    status, result = _probe_url(
        f"{ep}/api/show", EMBED_PROBE_TAGS_TIMEOUT_S,
        payload={"model": fallback},
    )
    if status != "ok" or not isinstance(result, dict):
        return None
    info = result.get("model_info")
    if not isinstance(info, dict):
        return None
    fb_dim: Optional[int] = None
    for key, val in info.items():
        if isinstance(key, str) and key.endswith(".embedding_length"):
            try:
                fb_dim = int(val)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                fb_dim = None
            break
    if fb_dim is None or fb_dim == dim:
        return None
    return (
        f"embedding_fallback {fallback} emits {fb_dim}-dim vectors vs the "
        f"index's {dim}-dim table -- it can never serve (the dimension "
        "guard refuses mismatched vectors) and only adds a model load + "
        "timeout chain to every failure path",
        f"set embedding_fallback: null or use a {dim}-dim model",
    )


def check_embedding_path(paths: HealthPaths) -> CheckResult:
    """Check 11 (2026-08-19): probe the embedding path END-TO-END. The
    hybrid search's vector leg degrades to BM25-only on any embed failure
    (by design -- retrieval must never block a hook), which means an
    Ollama that is down, missing the embed model, or serving cold is
    INVISIBLE at search time. This check makes it visible at every
    SessionStart, discriminating the three live failure modes with
    matching fix hints:

      1. GET /api/tags fails            -> 'Ollama server unreachable'
                                            FIX: start Ollama
      2. model absent from /api/tags     -> 'embedding model missing'
                                            FIX: ollama pull <model>
      3. POST /api/embeddings times out  -> 'timeout (model cold?)'
                                            FIX: retry -- model loading

    A successful probe doubles as a keep-alive: it carries the same
    keep_alive + num_ctx the real embed path sends, re-arming Ollama's
    residency timer at every SessionStart (see the check-11 constants)."""
    name = "embedding_path"
    endpoint, model, keep_alive = _resolve_embed_settings(paths)
    ep = endpoint.rstrip("/")

    status, result = _probe_url(f"{ep}/api/tags", EMBED_PROBE_TAGS_TIMEOUT_S)
    if status in ("timeout", "unreachable"):
        return CheckResult(
            name, False,
            f"Ollama server unreachable at {ep} ({result})",
            "start Ollama (`ollama serve` or the desktop app), then re-run "
            "this gate",
        )
    if status == "http_error":
        code, head = result  # type: ignore[misc]
        return CheckResult(
            name, False, f"/api/tags returned {code}: {head}",
            "check the Ollama server logs / restart Ollama",
        )
    if status == "malformed":
        return CheckResult(
            name, False, f"/api/tags {result}",
            "endpoint does not answer like an Ollama server -- verify the "
            "ollama_endpoint config / --ollama-endpoint flag",
        )
    installed: set = set()
    models = result.get("models") if isinstance(result, dict) else None
    if isinstance(models, list):
        for m in models:
            if isinstance(m, dict):
                for key in ("name", "model"):
                    v = m.get(key)
                    if isinstance(v, str):
                        installed.add(v)
    wanted = {model}
    if ":" not in model:
        wanted.add(model + ":latest")  # Ollama's implicit default tag
    if not (wanted & installed):
        return CheckResult(
            name, False,
            f"embedding model {model} missing from Ollama "
            f"({len(installed)} model(s) installed)",
            f"run `ollama pull {model}`",
        )

    cfg_raw = _read_claude_mem_config(
        paths.project_root / CLAUDE_MEM_DIRNAME / CLAUDE_MEM_CONFIG_NAME
    )
    fb_bad = _fallback_dim_verdict(ep, cfg_raw, installed)
    if fb_bad is not None:
        fb_detail, fb_hint = fb_bad
        return CheckResult(name, False, fb_detail, fb_hint)

    probe_payload = {"model": model, "prompt": "memory-health embed probe",
                     "keep_alive": keep_alive,
                     "options": {"num_ctx": EMBED_PROBE_NUM_CTX}}
    t0 = time.monotonic()
    status, result = _probe_url(
        f"{ep}/api/embeddings", EMBED_PROBE_READ_TIMEOUT_S,
        payload=probe_payload,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    warmed_note = ""
    if status == "timeout":
        # Cold-start self-warming (day-9 shakedown review): give the
        # model load ONE generous window, then re-verdict. A server that
        # is genuinely dead or wedged times out again and stays RED; a
        # merely-cold model comes up GREEN with the warm-up named in the
        # detail instead of paging the operator to "retry".
        t1 = time.monotonic()
        status, result = _probe_url(
            f"{ep}/api/embeddings", EMBED_WARM_RETRY_TIMEOUT_S,
            payload=probe_payload,
        )
        warm_s = time.monotonic() - t1
        elapsed_ms = warm_s * 1000.0
        if status == "timeout":
            return CheckResult(
                name, False,
                f"embed probe timeout after {EMBED_PROBE_READ_TIMEOUT_S:.1f}s "
                f"AND a {EMBED_WARM_RETRY_TIMEOUT_S:.0f}s warm-up retry "
                "(server wedged, or the model cannot load?)",
                "check `ollama ps`, GPU memory, and the Ollama server logs",
            )
        warmed_note = f" (cold start: warmed in {warm_s:.1f}s)"
    if status == "unreachable":
        return CheckResult(
            name, False,
            f"Ollama server unreachable at {ep} ({result})",
            "start Ollama (`ollama serve` or the desktop app), then re-run "
            "this gate",
        )
    if status == "http_error":
        code, head = result  # type: ignore[misc]
        return CheckResult(
            name, False, f"embed probe returned {code}: {head}",
            "check the Ollama server logs (model load failure?)",
        )
    if status == "malformed":
        return CheckResult(
            name, False, f"embed probe {result}",
            "check the Ollama server logs",
        )
    vec = result.get("embedding") if isinstance(result, dict) else None
    if not isinstance(vec, list) or not vec:
        return CheckResult(
            name, False, "embed probe response missing 'embedding'",
            "check the Ollama server logs (model load failure?)",
        )
    pinned_dim = _cfg_explicit_embedding_dim(cfg_raw)
    if pinned_dim is not None and len(vec) != pinned_dim:
        return CheckResult(
            name, False,
            f"embedding model {model} emits {len(vec)}-dim vectors vs the "
            f"index's pinned {pinned_dim}-dim table -- every vector insert "
            "is refused by the dimension guard, so the vector leg is dead "
            "while looking configured",
            f"use a {pinned_dim}-dim embedding model, or rebuild the index "
            f"with embedding_dim: {len(vec)}",
        )
    return CheckResult(
        name, True,
        f"end-to-end embed OK: {len(vec)}-dim in {elapsed_ms:.0f}ms "
        f"via {model}{warmed_note}",
        "",
    )


# --------------------------------------------------------------------------
# Novelty bookkeeping in telemetry.db (the gate's OWN store; not observed by
# any check). Kept minimal + self-initializing; never raises. v2 adds
# session_id (the render's receiving session) and corpus_sig (the corpus
# signature at observation time); legacy 3-column tables are ALTERed in
# place -- rows are never dropped.
# --------------------------------------------------------------------------
_NOVELTY_TABLE = (
    "CREATE TABLE IF NOT EXISTS memhealth_novelty ("
    "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
    "sessionstart_hash TEXT NOT NULL, session_id TEXT, corpus_sig TEXT)"
)
# History window the repetition rule walks. Bounded so the gate's per-run
# read stays O(1) as the table grows; NOVELTY_DISTINCT_SESSION_REPEATS
# distinct sessions are found within a handful of rows in practice.
_NOVELTY_HISTORY_LIMIT = 200


def _ensure_novelty_schema(conn: sqlite3.Connection) -> None:
    conn.execute(_NOVELTY_TABLE)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(memhealth_novelty)")}
    if "session_id" not in cols:
        conn.execute("ALTER TABLE memhealth_novelty ADD COLUMN session_id TEXT")
    if "corpus_sig" not in cols:
        conn.execute("ALTER TABLE memhealth_novelty ADD COLUMN corpus_sig TEXT")


def _novelty_rows(telemetry_db: Path) -> List[Tuple[str, str, str]]:
    """Prior observations, newest first: (hash, session_id, corpus_sig).
    NULL session/sig (pre-v2 rows) render as '' -- they carry no
    cross-session evidence."""
    if not telemetry_db.is_file():
        return []
    try:
        conn = sqlite3.connect(telemetry_db)
    except sqlite3.Error:
        return []
    try:
        _ensure_novelty_schema(conn)
        rows = conn.execute(
            "SELECT sessionstart_hash, session_id, corpus_sig "
            "FROM memhealth_novelty ORDER BY id DESC LIMIT ?",
            (_NOVELTY_HISTORY_LIMIT,),
        ).fetchall()
        return [(r[0], r[1] or "", r[2] or "") for r in rows]
    except sqlite3.Error:
        return []
    finally:
        conn.close()


def _record_novelty_hash(
    telemetry_db: Path, digest: str, session_id: str = "", corpus_sig: str = ""
) -> None:
    try:
        telemetry_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(telemetry_db)
    except (sqlite3.Error, OSError):
        return
    try:
        _ensure_novelty_schema(conn)
        conn.execute(
            "INSERT INTO memhealth_novelty "
            "(timestamp, sessionstart_hash, session_id, corpus_sig) "
            "VALUES (?, ?, ?, ?)",
            (datetime.now(timezone.utc).isoformat(), digest,
             session_id, corpus_sig),
        )
        conn.commit()
    except sqlite3.Error:
        pass
    finally:
        conn.close()


def _corpus_sig(index_db: Path) -> str:
    """Corpus signature: SHA256 over the sorted incr:* watermark pairs plus
    the chunks and decisions row counts. The counts make the signature move
    under EVERY ingestion path (Ingester.add, capture-synthesize), not only
    the incremental pipeline that rewrites incr:* -- the render's inputs are
    chunks (corrections) + decisions, so this is the set that can change it.
    The claude_mem session_start hook computes the same concept independently
    for its in-band flag (the two signatures are never compared to each
    other; each store only ever compares its own). Missing db/tables degrade
    to sentinel components -- always returns a stable hex string, never
    raises."""
    parts: List[str] = []
    conn = _connect_ro(index_db)
    if conn is None:
        parts.append("no-index")
    else:
        try:
            try:
                rows = conn.execute(
                    "SELECT key, value FROM meta WHERE key LIKE 'incr:%' "
                    "ORDER BY key"
                ).fetchall()
                parts.extend(f"{r['key']}={r['value']}" for r in rows)
            except sqlite3.Error:
                parts.append("no-meta")
            for table in ("chunks", "decisions"):
                try:
                    n = conn.execute(
                        f"SELECT COUNT(*) AS n FROM {table}").fetchone()["n"]
                except sqlite3.Error:
                    n = -1
                parts.append(f"{table}:{n}")
        finally:
            conn.close()
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
_CHECKS = (
    check_ingest_watermark,
    check_vector_coverage,
    check_hook_heartbeats,
    check_pending_capture_queue,
    check_lineage_cache_age,
    check_module_state_age,
    check_graphify_corpus_age,
    check_memory_md_health,
    check_sessionstart_novelty,
    check_claude_md_deprecated,
    check_embedding_path,  # 11 (2026-08-19) -- appended so 1-10 keep their numbers
)


def run_checks(paths: HealthPaths) -> List[CheckResult]:
    """Run every registered check (len(_CHECKS) of them; the render line's
    N/N total is recalculated from this roster, never hardcoded). A check
    that raises unexpectedly is converted into a RED result (never
    propagated) so the gate can never crash on a single check's edge
    case."""
    results: List[CheckResult] = []
    for fn in _CHECKS:
        try:
            results.append(fn(paths))
        except Exception as exc:  # defensive: unknown data-shape edge case
            results.append(CheckResult(
                fn.__name__.replace("check_", ""), False,
                f"check raised: {exc!r}", "investigate the data source",
            ))
    return results


def render(results: Sequence[CheckResult]) -> str:
    """One green line when all pass; else a red block naming each failure +
    its fix hint."""
    total = len(results)
    green = sum(1 for r in results if r.ok)
    if green == total:
        return f"MEMORY-HEALTH: {green}/{total} GREEN"
    reds = [r for r in results if not r.ok]
    lines = [f"MEMORY-HEALTH: {green}/{total} GREEN ({len(reds)} RED)"]
    for r in reds:
        lines.append(f"  RED {r.name}: {r.detail}  FIX: {r.fix_hint}")
    return "\n".join(lines)


def _write_self_heartbeat(telemetry_db: Path, ok: bool, detail: str) -> None:
    """Record the gate's OWN heartbeat via Task 5's mechanism so check #3
    covers memory_health itself next session (spec risk section 9). Uses the
    installed claude_mem.telemetry.record_hook_heartbeat when importable;
    falls back to a direct insert otherwise. Never raises."""
    try:
        _ensure_sqlite_vec_on_path()  # user-site may also carry claude_mem
        from claude_mem.telemetry import record_hook_heartbeat
        record_hook_heartbeat(
            telemetry_db, hook=SELF_HOOK_NAME, ok=ok, detail=detail[:500],
        )
        return
    except Exception:
        pass
    # Fallback: self-contained insert (same schema as telemetry.py).
    try:
        telemetry_db.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(telemetry_db)
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS hook_heartbeat ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TEXT NOT NULL, "
                "hook TEXT NOT NULL, ok INTEGER NOT NULL, detail TEXT)"
            )
            conn.execute(
                "INSERT INTO hook_heartbeat (timestamp, hook, ok, detail) "
                "VALUES (?, ?, ?, ?)",
                (datetime.now(timezone.utc).isoformat(),
                 SELF_HOOK_NAME, int(bool(ok)), detail[:500]),
            )
            conn.commit()
        finally:
            conn.close()
    except Exception:
        pass


def main_with_paths(paths: HealthPaths) -> int:
    """Run the gate against explicit paths, print the render, write the gate's
    own heartbeat, and return 0 ALWAYS (report-only, never blocks)."""
    results = run_checks(paths)
    output = render(results)
    print(output)
    green = sum(1 for r in results if r.ok)
    _write_self_heartbeat(
        paths.telemetry_db, ok=True,
        detail=f"{green}/{len(results)} green",
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_health",
        description=f"Memory-health gate: {len(_CHECKS)} pre-registered "
                    "checks over the harness memory stack. Report-only "
                    "(exits 0 once the flags parse). Both path flags are "
                    "REQUIRED: the harness init tool bakes concrete paths "
                    "into the installed hook command.",
    )
    p.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the target project root. REQUIRED. index.db /"
             " telemetry.db resolve to <root>/.claude-mem/, CLAUDE.md to "
             "<root>/CLAUDE.md, and fix-hint commands name this root.",
    )
    p.add_argument(
        "--memory-dir",
        required=True,
        help="Absolute path to the memory directory holding MEMORY.md. "
             "REQUIRED.",
    )
    p.add_argument(
        "--module-states-dir",
        default=None,
        help="Directory of *.state.yaml auto-derived module states (check 6)."
             " Default: <project-root>/docs/marathon/module_states.",
    )
    p.add_argument(
        "--graphify-dir",
        action="append",
        default=None,
        metavar="DIR",
        help="Graphify corpus out-dir candidate (check 7); repeatable, first "
             "EXISTING dir wins. With no flag, check 7 reports 'no graphify "
             "out-dir found' RED.",
    )
    p.add_argument(
        "--deprecated-token",
        action="append",
        default=None,
        metavar="TOKEN",
        help="Deprecated tool symbol CLAUDE.md must no longer mention "
             "(check 10); repeatable. With no flag, check 10 scans for "
             "nothing and is green.",
    )
    p.add_argument(
        "--ollama-endpoint",
        default=None,
        metavar="URL",
        help="Ollama endpoint for check 11's embedding-path probe. "
             "Default: the project's .claude-mem/config.yaml "
             "ollama_endpoint, else " + DEFAULT_OLLAMA_ENDPOINT + ".",
    )
    p.add_argument(
        "--embed-model",
        default=None,
        metavar="MODEL",
        help="Embedding model check 11 probes for. Default: the project's "
             ".claude-mem/config.yaml embedding_model, else "
             + DEFAULT_EMBED_MODEL + ".",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    # cp1252 console guard: this exact class of bug killed gen_decisions_state.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = build_arg_parser().parse_args(argv)
    paths = HealthPaths.from_cli(
        project_root=Path(args.project_root),
        memory_dir=Path(args.memory_dir),
        module_states_dir=(
            Path(args.module_states_dir)
            if args.module_states_dir is not None else None
        ),
        graphify_dirs=tuple(Path(d) for d in (args.graphify_dir or [])),
        deprecated_tokens=tuple(args.deprecated_token or []),
        ollama_endpoint=args.ollama_endpoint,
        embed_model=args.embed_model,
    )
    return main_with_paths(paths)


if __name__ == "__main__":
    raise SystemExit(main())
