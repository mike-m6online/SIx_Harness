"""Incremental ingestion wired into SessionEnd (spec R3).

`claude-mem bulk` is a full canonical re-scan of the corpus -- correct for
a backfill, wrong to run every session (it re-reads every doc/memory/
session file every time). This module is the SessionEnd-safe complement:
a budgeted, watermarked, resumable pass that ingests only what changed
since the last run, so the index stays current without ever re-scanning
history it has already seen.

Two watermark shapes, matching the two change-detection strategies:

  docs / memory / ledgers -- mtime scan. A SINGLE per-source watermark
    (`incr:docs`, `incr:memory`, `incr:ledgers` in the `meta` table) holds
    the newest file mtime observed as of the last completed pass over
    that source. Each run lists every matching file and re-chunks any
    file whose mtime is STRICTLY newer than the stored watermark (a file
    with mtime <= watermark was already seen and is skipped without
    reading its content). Re-chunking is delete-and-replace BY
    file_path: this is the ONE permitted delete in this codebase (see
    CLAUDE.md Rule #2 / the project's "never DELETE chunks" convention)
    -- a stale chunk of a file that has since been rewritten is not
    provenance, it is wrong data, and it is scoped to exactly the file
    being re-ingested (a DELETE ... WHERE file_path = ?, never a
    broader wipe). New chunks are then inserted and embedded inline via
    the same Ingester used by `bulk`.

  sessions -- byte-offset scan, reusing the capture-pipeline pattern in
    hooks/session_end.py (`extract_offset:<file>` there; `incr:sessions:
    <file>` here, same first-observation-anchors-forward semantics: a
    session JSONL seen for the first time anchors its watermark at the
    current end-of-file and ingests nothing from it this run, because
    its historical content is already covered by the Rung-1 bulk
    backfill -- only bytes appended AFTER that anchor are ever mined).
    Every session message is passed through
    `claude_mem.filters.is_harness_content` before ingestion; harness-
    origin content (skill bodies, hook stdout, system reminders,
    compaction summaries) is dropped, matching the `bulk` command's
    filter-before-grade discipline.

Budget: a single hard wall-clock deadline for the ENTIRE run (not reset
per source). Checked between files/sessions (never mid-file), so a run
that exceeds its budget stops cleanly at a file boundary; the watermark
records exactly the progress actually made, so the NEXT run resumes
forward from there -- no file is doubly processed and no file is skipped.

Embedding: constructed by the caller (hooks/session_end.py, the CLI) with
an explicit BULK_READ_TIMEOUT_S read timeout -- NOT the interactive-hook
2.0s default (see embed.py's module docstring for why 2.0s silently
drops embeddings under a contended Ollama daemon). Every embed failure
is logged to `ingestion_log` (action='embed_fail') by Ingester.add()
itself and never crashes the run -- ingestion of the surrounding files
continues; embed-backfill sweeps up any stragglers later.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Protocol

import sqlite_vec

from claude_mem.acronyms import derive_aliases
from claude_mem.bulk import (
    collect_do_not_rebuild_modules, collect_module_names, detect_module,
    doc_signal_weight, is_curated_correction_file, is_decision,
    is_do_not_rebuild, parse_markdown_doc, parse_memory_md,
)
from claude_mem.filters import is_harness_content
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

# Ledger glob (local filesystem only; ledgers on remote machines arrive
# via synced docs -- this module never SSHes out to fetch them).
_LEDGER_GLOB = "**/.superpowers/sdd/progress.md"

# Directories that would otherwise blow up an rglob (version control,
# virtualenvs, node_modules) -- mirrors the exclusions bulk/docs scanning
# already relies on implicitly by scanning under docs/ only; ledgers and
# memory scans walk wider trees so they need an explicit skip-list.
_SKIP_DIR_NAMES = {".git", "node_modules", "__pycache__", ".venv", "venv"}


def _path_is_skippable(path: Path) -> bool:
    return any(part in _SKIP_DIR_NAMES for part in path.parts)


class _Embedder(Protocol):
    def embed(self, text: str) -> list: ...


@dataclass
class _Budget:
    """Single wall-clock deadline shared across every source in a run."""

    deadline: float
    now_fn: Callable[[], float]

    def expired(self) -> bool:
        return self.now_fn() >= self.deadline


def _get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
    return row[0] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute(
        "INSERT INTO meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, value),
    )
    conn.commit()


def _delete_chunks_for_file(conn: sqlite3.Connection, file_path: str) -> None:
    """The ONE permitted delete in this codebase: scoped exactly to the
    file_path being re-chunked (never a broader wipe). A stale chunk of a
    rewritten file is wrong data, not provenance -- see module docstring.
    Also removes the matching chunks_vec rows (foreign id, not a FK, so
    the delete must be explicit) and lets the chunks_ad FTS trigger keep
    chunks_fts in sync."""
    ids = [
        r[0] for r in conn.execute(
            "SELECT id FROM chunks WHERE file_path = ?", (file_path,)
        ).fetchall()
    ]
    if not ids:
        return
    conn.executemany("DELETE FROM chunks_vec WHERE chunk_id = ?", [(i,) for i in ids])
    conn.executemany("DELETE FROM chunks WHERE id = ?", [(i,) for i in ids])
    conn.commit()


@dataclass
class _SourceResult:
    files_ingested: int = 0
    files_skipped: int = 0
    chunks_added: int = 0


def _run_mtime_source(
    *,
    conn: sqlite3.Connection,
    ingester: Ingester,
    watermark_key: str,
    files: List[Path],
    parse_fn: Callable[[Path], list],
    source_name: str,
    budget: _Budget,
    chunk_kwargs_fn: Callable[[dict], dict],
) -> _SourceResult:
    """Shared mtime-scan implementation for docs/memory/ledgers.

    `parse_fn` returns the list-of-chunk-dicts shape shared by
    parse_markdown_doc / parse_memory_md (content/file_path/file_mtime/
    line_start/line_end[/metadata]). `chunk_kwargs_fn` maps one such dict
    to the extra Chunk(**kwargs) fields specific to this source (module
    tagging, signal weight, do_not_rebuild, aliases).

    Watermark shape: JSON {"mtime": <float>, "at_boundary": [file_path,
    ...]}, not a bare float. Root cause this guards against: several
    files written in the same batch (a git checkout, a sync, a bulk doc
    edit) commonly share the EXACT SAME mtime on filesystems with coarse
    mtime resolution (confirmed empirically on this project's Windows
    filesystem: files written microseconds apart round to an identical
    st_mtime). A bare `mtime > watermark` scheme ingests the first such
    file, advances the watermark to that shared mtime, and then silently
    skips every sibling at the same mtime FOREVER on every future run --
    silent, permanent data loss with no error signal. The fix: a file at
    exactly the watermark mtime is still eligible unless its file_path is
    already recorded in `at_boundary` (the set of file_paths already
    ingested AT that exact mtime). `at_boundary` resets whenever the max
    mtime strictly advances.
    """
    result = _SourceResult()
    watermark_raw = _get_meta(conn, watermark_key)
    if watermark_raw is None:
        watermark_mtime = 0.0
        at_boundary: set = set()
    else:
        try:
            parsed = json.loads(watermark_raw)
            watermark_mtime = float(parsed["mtime"])
            at_boundary = set(parsed.get("at_boundary", []))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            # Pre-existing bare-float watermark from before this fix (or
            # any other unreadable value) -- treat as "seen nothing at
            # the boundary yet" rather than crashing SessionEnd.
            try:
                watermark_mtime = float(watermark_raw)
            except (TypeError, ValueError):
                watermark_mtime = 0.0
            at_boundary = set()

    newest_mtime = watermark_mtime
    newest_at_boundary = set(at_boundary)

    # Deterministic order so a budget-truncated run is reproducible run to
    # run (same files ingested first, same files deferred).
    for path in sorted(files, key=lambda p: str(p)):
        if budget.expired():
            break
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        file_path_str = str(path)
        if mtime < watermark_mtime:
            result.files_skipped += 1
            continue
        if mtime == watermark_mtime and file_path_str in at_boundary:
            result.files_skipped += 1
            continue

        try:
            chunk_dicts = parse_fn(path)
        except OSError as exc:
            # A file that vanished / became unreadable between the
            # directory listing and this read (TOCTOU: deleted mid-scan,
            # a transient permission error, a network-drive hiccup) must
            # not crash the whole incremental-ingest run -- log it and
            # move on. Deliberately does NOT advance the watermark for
            # this file: an mtime-based watermark has nothing else to key
            # on, and leaving it below the file's mtime means the next
            # run retries the read rather than silently giving up on the
            # file forever (mirrors embed_backfill's "retry, don't give
            # up" discipline for embed failures).
            conn.execute(
                """
                INSERT INTO ingestion_log
                    (timestamp, source_path, chunks_added, file_mtime,
                     chunk_ids, action, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _now_iso(), file_path_str, 0, mtime, "[]",
                    "read_fail", str(exc)[:2000],
                ),
            )
            conn.commit()
            result.files_skipped += 1
            continue

        _delete_chunks_for_file(conn, file_path_str)
        for cd in chunk_dicts:
            extra = chunk_kwargs_fn(cd)
            if ingester.add(Chunk(
                content=cd["content"],
                source=source_name,
                file_path=cd.get("file_path", file_path_str),
                line_start=cd.get("line_start"),
                line_end=cd.get("line_end"),
                file_mtime=cd.get("file_mtime", mtime),
                **extra,
            )):
                result.chunks_added += 1
        result.files_ingested += 1

        if mtime > newest_mtime:
            newest_mtime = mtime
            newest_at_boundary = {file_path_str}
        elif mtime == newest_mtime:
            newest_at_boundary.add(file_path_str)
        # Persist progress after EVERY file so a budget expiry (or a
        # crash) between files never loses work already committed --
        # each file's chunk inserts are already durable via Ingester.add's
        # per-chunk commit; this just advances the resume point to match.
        _set_meta(conn, watermark_key, json.dumps({
            "mtime": newest_mtime,
            "at_boundary": sorted(newest_at_boundary),
        }))

    return result


def _scan_docs(root: Path, budget: _Budget, conn, ingester: Ingester, modules, dnr_modules) -> _SourceResult:
    docs_dir = root / "docs"
    if not docs_dir.is_dir():
        return _SourceResult()
    files = [p for p in docs_dir.rglob("*.md") if not _path_is_skippable(p)]

    def _kwargs(cd: dict) -> dict:
        mod = detect_module(cd["content"], modules)
        return dict(
            module=mod,
            signal_weight=doc_signal_weight(cd["file_path"]),
            is_decision=is_decision(cd["content"]),
            do_not_rebuild=(mod in dnr_modules) or is_do_not_rebuild(cd["content"]),
            aliases=derive_aliases(cd["content"]),
        )

    return _run_mtime_source(
        conn=conn, ingester=ingester, watermark_key="incr:docs",
        files=files, parse_fn=parse_markdown_doc, source_name="doc",
        budget=budget, chunk_kwargs_fn=_kwargs,
    )


def _scan_memory(root: Path, budget: _Budget, conn, ingester: Ingester, modules, dnr_modules) -> _SourceResult:
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    mem_dir = Path.home() / ".claude" / "projects" / proj_slug / "memory"
    if not mem_dir.is_dir():
        return _SourceResult()
    files = [p for p in mem_dir.rglob("*.md") if not _path_is_skippable(p)]

    def _kwargs(cd: dict) -> dict:
        mod = detect_module(cd["content"], modules)
        return dict(
            module=mod,
            signal_weight=40,
            is_decision=is_decision(cd["content"]),
            # Operator-curated correction files (feedback_*.md /
            # invariant_*.md) carry is_correction=1 by NAME: the phrase
            # scanner has ~zero recall on curated corrections, and the
            # basename is the operator's own curation signal (see
            # bulk.is_curated_correction_file).
            is_correction=is_curated_correction_file(cd["file_path"]),
            do_not_rebuild=(mod in dnr_modules) or is_do_not_rebuild(cd["content"]),
            aliases=derive_aliases(cd["content"]),
        )

    return _run_mtime_source(
        conn=conn, ingester=ingester, watermark_key="incr:memory",
        files=files, parse_fn=parse_memory_md, source_name="memory",
        budget=budget, chunk_kwargs_fn=_kwargs,
    )


def _scan_ledgers(root: Path, budget: _Budget, conn, ingester: Ingester, modules, dnr_modules) -> _SourceResult:
    files = [p for p in root.glob(_LEDGER_GLOB) if not _path_is_skippable(p)]

    def _kwargs(cd: dict) -> dict:
        mod = detect_module(cd["content"], modules)
        return dict(
            module=mod,
            signal_weight=30,
            is_decision=is_decision(cd["content"]),
            do_not_rebuild=(mod in dnr_modules) or is_do_not_rebuild(cd["content"]),
            aliases=derive_aliases(cd["content"]),
        )

    return _run_mtime_source(
        conn=conn, ingester=ingester, watermark_key="incr:ledgers",
        files=files, parse_fn=parse_markdown_doc, source_name="ledger",
        budget=budget, chunk_kwargs_fn=_kwargs,
    )


def _scan_sessions(root: Path, budget: _Budget, conn, ingester: Ingester, modules) -> _SourceResult:
    result = _SourceResult()
    home_proj_dir = Path.home() / ".claude" / "projects"
    if not home_proj_dir.is_dir():
        return result
    proj_slug = str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    jsonls = sorted(
        home_proj_dir.rglob(f"*{proj_slug}*/*.jsonl"), key=lambda p: str(p),
    )
    for jsonl in jsonls:
        if budget.expired():
            break
        watermark_key = f"incr:sessions:{jsonl.name}"
        try:
            size = jsonl.stat().st_size
        except OSError:
            continue
        prev = _get_meta(conn, watermark_key)
        if prev is None:
            # First observation: anchor forward, ingest nothing now. The
            # Rung-1 bulk backfill already covers historical session
            # content; re-mining a multi-GB log here on first sight would
            # duplicate that work every time a new session file appears.
            _set_meta(conn, watermark_key, str(size))
            continue
        start = int(prev)
        if start > size:
            start = size  # file shrank/rotated; resume from current end.
        if start >= size:
            continue  # nothing new appended since last watermark.

        added_any = False
        try:
            with jsonl.open("rb") as fh:
                fh.seek(start)
                raw = fh.read()
        except OSError as exc:
            # Session file vanished/rotated/unreadable between the listing
            # and this read -- log and move to the next session file
            # rather than crashing SessionEnd. Watermark is left
            # unadvanced so a transient failure is retried next run.
            conn.execute(
                """
                INSERT INTO ingestion_log
                    (timestamp, source_path, chunks_added, file_mtime,
                     chunk_ids, action, detail)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (_now_iso(), str(jsonl), 0, None, "[]", "read_fail", str(exc)[:2000]),
            )
            conn.commit()
            continue
        offset_bytes = raw.decode("utf-8", errors="replace")
        for line in offset_bytes.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            msg = rec.get("message") or {}
            role = msg.get("role") or rec.get("type")
            content = msg.get("content")
            if isinstance(content, list):
                content = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not isinstance(content, str) or not content.strip():
                continue
            if is_harness_content(content):
                continue
            if ingester.add(Chunk(
                content=content,
                source="claude_code",
                role=role,
                session_id=jsonl.stem,
                date=rec.get("timestamp"),
                module=detect_module(content, modules),
                signal_weight=10,
                is_decision=is_decision(content),
                aliases=derive_aliases(content),
            )):
                result.chunks_added += 1
            added_any = True
        _set_meta(conn, watermark_key, str(size))
        if added_any:
            result.files_ingested += 1
    return result


def run_incremental(
    project_root: Path,
    *,
    embedder: _Embedder,
    budget_s: float = 55.0,
    now_fn: Callable[[], float] = time.monotonic,
) -> Dict[str, Dict[str, int]]:
    """Run one budgeted incremental-ingestion pass. Returns a summary dict
    keyed by source ("docs", "memory", "ledgers", "sessions"), each with
    files_ingested / files_skipped / chunks_added.

    Raises RuntimeError if no index.db exists at project_root/.claude-mem
    (mirrors migrate_regrade / embed_backfill's refuse-to-run-against-
    nothing behavior) -- callers on the SessionEnd path (hooks/
    session_end.py) check cfg.db_path.is_file() first and no-op instead
    of calling this, exactly as the existing capture-extract step does.
    """
    root = Path(project_root)
    db_path = root / ".claude-mem" / "index.db"
    if not db_path.is_file():
        raise RuntimeError(f"incremental: no database at {db_path}")

    init_db(db_path)
    budget = _Budget(deadline=now_fn() + budget_s, now_fn=now_fn)

    modules = collect_module_names(root)
    dnr_modules = collect_do_not_rebuild_modules(root)

    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    # Belt-and-braces: raise the wait ceiling above the live DB's
    # busy_timeout=5000 default for this connection's own short writes
    # (meta watermarks, the scoped re-chunk delete) -- see Ingester.add's
    # docstring for the long-lock (embed-across-transaction) fix, which
    # this pragma complements rather than replaces.
    conn.execute("PRAGMA busy_timeout = 30000")
    ingester = Ingester(db_path=db_path, embedder=embedder)
    try:
        # Source order is memory -> ledgers -> sessions -> docs, NOT the
        # docs-first order this shipped with originally. Root cause: the
        # budget is one shared wall-clock deadline for the whole run (see
        # module docstring), and docs/ routinely holds hundreds of files
        # (375 on this project) vs. a handful of curated memory notes --
        # scanning docs first starved memory/ledgers/sessions of any
        # budget at all on a live project (confirmed: `incr:docs` was
        # advancing across runs while `incr:memory` had never been
        # written). Memory is the highest-signal, lowest-volume source
        # (curated by a human, signal_weight=40) and ledgers/sessions are
        # comparatively small too, so putting all three ahead of the
        # large, lower-priority docs corpus means a budget-truncated run
        # always finishes the small sources before spending its remaining
        # time on the long tail of docs.
        memory_result = _scan_memory(root, budget, conn, ingester, modules, dnr_modules)
        ledgers_result = _scan_ledgers(root, budget, conn, ingester, modules, dnr_modules)
        sessions_result = _scan_sessions(root, budget, conn, ingester, modules)
        docs_result = _scan_docs(root, budget, conn, ingester, modules, dnr_modules)
    finally:
        ingester.close()
        conn.close()

    def _as_dict(r: _SourceResult) -> Dict[str, int]:
        return {
            "files_ingested": r.files_ingested,
            "files_skipped": r.files_skipped,
            "chunks_added": r.chunks_added,
        }

    return {
        "docs": _as_dict(docs_result),
        "memory": _as_dict(memory_result),
        "ledgers": _as_dict(ledgers_result),
        "sessions": _as_dict(sessions_result),
    }
