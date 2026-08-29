"""SessionEnd hook step: correction-detection on the just-ended session
(Phase 4.5): extract user-correction events, index them at sw=100,
boost matching chunks by +20.

Watermarked (spec R3 discipline): the live session JSONL is multi-GB,
so each run scans only bytes appended since the last run, via the same
per-file byte-offset meta watermark capture_from_jsonl uses. First
observation of a file anchors the watermark at the current end --
historical content is covered by the bulk/backfill passes, and a
full-file parse inside a hook is exactly the silent-death class this
arc closes."""
from __future__ import annotations

from pathlib import Path

from typing import List, Tuple

from claude_mem.config import ProjectConfig
from claude_mem.corrections import apply_corrections, scan_corrections
from claude_mem.embed import EmbeddingClient
from ..capture import CaptureStore
from ..extract_decisions import ScanSkips, scan_candidates


def _safe_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def _matching_session_jsonls(
    session_id: str, project_root: Path,
) -> List[Path]:
    """Session JSONLs for this project, preferring an exact session_id
    match, falling back to the most-recently-modified project JSONL
    (SessionEnd fires for the session that just ended, whose file is the
    freshest one when the id is missing or unmatched)."""
    home_proj_dir = Path.home() / ".claude" / "projects"
    if not home_proj_dir.is_dir():
        return []
    proj_slug = (
        str(project_root).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    proj_jsonls = list(home_proj_dir.rglob(f"*{proj_slug}*/*.jsonl"))
    matching: List[Path] = []
    if session_id:
        matching = [j for j in proj_jsonls if j.stem == session_id]
        if not matching:
            matching = [
                j for j in home_proj_dir.rglob("*.jsonl")
                if j.stem == session_id
            ]
    if not matching and proj_jsonls:
        matching = [max(proj_jsonls, key=_safe_mtime)]
    return matching


def run(session_id: str, project_root: Path) -> str:
    from claude_mem.embed import BULK_READ_TIMEOUT_S

    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""
    matching = _matching_session_jsonls(session_id, project_root)
    if not matching:
        return ""
    total_events = 0
    total_applied = 0
    embedder = None
    store = CaptureStore(cfg.db_path)
    try:
        for jsonl in matching:
            wm_key = f"corrections_offset:{jsonl.name}"
            size = jsonl.stat().st_size
            prev = store.get_meta(wm_key)
            if prev is None:
                # First observation: anchor at the current end (same
                # rationale as capture_from_jsonl -- never re-read a
                # multi-GB historical log inside a hook).
                store.set_meta(wm_key, str(size))
                continue
            start = min(int(prev), size)  # shrank/rotated -> current end
            events, new_offset = scan_corrections(
                jsonl, start_offset=start,
            )
            if events:
                if embedder is None:
                    # SessionEnd is post-turn: the generous bulk timeout
                    # is correct here, like run_incremental_ingest.
                    embedder = EmbeddingClient(
                        model=cfg.values["embedding_model"],
                        fallback_model=cfg.values["embedding_fallback"],
                        endpoint=cfg.values["ollama_endpoint"],
                        keep_alive=cfg.values["embedding_keep_alive"],
                        read_timeout_s=BULK_READ_TIMEOUT_S,
                    )
                total_events += len(events)
                total_applied += apply_corrections(
                    events, cfg.db_path, embedder=embedder,
                )
            store.set_meta(wm_key, str(new_offset))
    finally:
        store.close()
        if embedder is not None:
            embedder.close()
    if total_events == 0:
        return ""
    return (
        f"[claude-mem] session-end: detected {total_events} correction "
        f"events; applied {total_applied}."
    )


def capture_from_jsonl(
    store: "CaptureStore", jsonl: Path
) -> Tuple[int, int, ScanSkips]:
    """Resume-aware capture for one session JSONL. A per-file byte-offset
    watermark in the meta table makes each run process only newly-appended
    records, so the continuous multi-GB project log is never re-read from the
    start. Returns (n_decisions_added, n_dead_ends_added, skips) where skips
    counts candidates/records dropped by the precision deny-list."""
    wm_key = f"extract_offset:{jsonl.name}"
    size = jsonl.stat().st_size
    prev = store.get_meta(wm_key)
    if prev is None:
        # First observation of this file: anchor the watermark at the current
        # end and capture forward from the next session. Historical content is
        # already covered by the Rung-1 doc/memory backfill; re-mining a
        # multi-GB log here would fill the per-type caps with old content.
        store.set_meta(wm_key, str(size))
        return (0, 0, ScanSkips())
    start = int(prev)
    if start > size:
        # File shrank or rotated; resume from the current end.
        start = size
    decisions, dead_ends, new_offset, skips = scan_candidates(
        jsonl, start_offset=start,
    )
    n_dec = sum(1 for c in decisions if store.add_decision(c))
    n_de = sum(1 for c in dead_ends if store.add_dead_end(c))
    store.set_meta(wm_key, str(new_offset))
    return (n_dec, n_de, skips)


# Default incremental-ingest budget for the SessionEnd path. Deliberately
# short of the 60s CLAUDE.md SessionEnd ceiling: run_candidates() above
# (candidate mining, no embeddings) and capture-synthesize (a SEPARATE
# hook step in settings.local.json, not part of this budget) already
# consume part of the overall hook time, and the incremental-ingest CLI
# entry point (`ingest-incremental`) defaults to the same 55.0s for
# standalone invocation.
DEFAULT_INCREMENTAL_BUDGET_S = 55.0


def run_incremental_ingest(project_root: Path, budget_s: float = DEFAULT_INCREMENTAL_BUDGET_S) -> str:
    """SessionEnd step: budgeted/watermarked/resumable incremental
    ingestion of docs/memory/ledgers/sessions (spec R3). Called from the
    `capture-extract` CLI command AFTER run_candidates(), so ordering
    (candidate mining first, then incremental ingest) is code-controlled
    within one hook invocation rather than relying on settings.json
    hook-array ordering.

    Never raises: a missing index.db, an unreachable Ollama, or any
    embedding failure must not crash SessionEnd -- embedding failures are
    logged to ingestion_log by Ingester.add() itself (never silently
    dropped, never fatal here); a missing-index RuntimeError from
    run_incremental() is caught and reported as a no-op string instead of
    propagating."""
    from claude_mem.embed import BULK_READ_TIMEOUT_S
    from claude_mem.incremental import run_incremental

    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        read_timeout_s=BULK_READ_TIMEOUT_S,
    )
    try:
        summary = run_incremental(project_root, embedder=embedder, budget_s=budget_s)
    except RuntimeError:
        return ""
    finally:
        embedder.close()
    total_files = sum(s["files_ingested"] for s in summary.values())
    total_chunks = sum(s["chunks_added"] for s in summary.values())
    if total_files == 0 and total_chunks == 0:
        return ""
    return (
        f"[claude-mem] session-end incremental-ingest: {total_files} file(s), "
        f"{total_chunks} chunk(s) added "
        f"(docs={summary['docs']['files_ingested']}, "
        f"memory={summary['memory']['files_ingested']}, "
        f"ledgers={summary['ledgers']['files_ingested']}, "
        f"sessions={summary['sessions']['files_ingested']})."
    )


def run_candidates(session_id: str, project_root: Path) -> str:
    """SessionEnd capture: mine the just-ended session for candidate decisions
    and dead-ends, write them as pending rows. No embeddings -- cheap enough
    for the non-blocking SessionEnd event. Robust to a missing/unmatched
    session_id: SessionEnd fires for the session that just ended, whose JSONL
    is the most-recently-modified one under this project's home-projects dir."""
    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""
    matching = _matching_session_jsonls(session_id, project_root)
    if not matching:
        return ""
    store = CaptureStore(cfg.db_path)
    n_dec = n_de = 0
    skips = ScanSkips()
    try:
        for jsonl in matching:
            d, e, s = capture_from_jsonl(store, jsonl)
            n_dec += d
            n_de += e
            skips = skips + s
    finally:
        store.close()
    if n_dec == 0 and n_de == 0 and skips.total == 0:
        return ""
    msg = (
        f"[claude-mem] session-end capture: {n_dec} pending decision(s), "
        f"{n_de} pending dead-end(s) written for review."
    )
    if skips.total > 0:
        # Precision-gate visibility: the operator sees how much the
        # deny-list dropped, per category, instead of the drops being
        # silent (the 2026-08-19 triage found 66/119 junk rows landing
        # unobserved).
        msg += f" {skips.summary()}."
    return msg
