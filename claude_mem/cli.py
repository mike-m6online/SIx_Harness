import contextlib
import json as _json
import shutil
import sys
from pathlib import Path

# Hook stdio is UTF-8: Claude Code pipes UTF-8 JSON to stdin and reads
# the rendered block from stdout. Windows Python defaults PIPED streams
# to the ANSI code page (cp1252), so stdout raises on the first
# non-cp1252 char in a render and stdin silently mis-decodes UTF-8 into
# mojibake / lone surrogates that explode at later encode boundaries.
# Reconfigure before any I/O; errors="replace" because a hook must
# degrade, never crash Claude Code's turn (nudge-not-stop).
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass  # None/closed streams or non-reconfigurable test wrappers
del _stream

import click


def _read_stdin_json() -> dict:
    """Read Claude Code's hook JSON from stdin. Returns {} on empty / parse error."""
    try:
        raw = sys.stdin.read()
    except (OSError, ValueError):
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        return _json.loads(raw)
    except _json.JSONDecodeError:
        return {}

from claude_mem.acronyms import derive_aliases
from claude_mem.bulk import (
    collect_do_not_rebuild_modules, collect_module_names, detect_module,
    doc_signal_weight, is_correction, is_curated_correction_file,
    is_decision, is_do_not_rebuild,
    parse_claude_code_jsonl, parse_git_log, parse_markdown_doc,
    parse_memory_md, parse_progress_jsonl, signal_weight,
)
from claude_mem.config import ProjectConfig
from claude_mem.corrections import apply_corrections, extract_corrections
from claude_mem.cross_project import search_cross_project
from claude_mem.embed import EmbeddingClient
from claude_mem.filters import is_harness_content
from claude_mem.hooks import (
    prompt_submit as _ps_hook, session_end as _se_hook,
    session_start as _ss_hook, tool_use as _tu_hook,
)
from claude_mem.ingest import Chunk, Ingester
from claude_mem.migrate_regrade import migrate_regrade
from claude_mem.schema import init_db
from claude_mem.report import write_weekly_summary
from claude_mem.search import Searcher
from claude_mem.telemetry import probe_components, record_hook_heartbeat
from claude_mem.textutil import clip, first_sentence
from claude_mem.version import __version__


@click.group()
@click.version_option(__version__)
def cli() -> None:
    """claude-mem -- portable per-project memory for Claude Code."""


@contextlib.contextmanager
def _heartbeat(hook: str, project_root: str):
    """Every hook entry point wraps its body in this: a try/finally that
    writes ONE hook_heartbeat row to telemetry.db, success or failure,
    cheap (<10ms, single-row INSERT) and itself failure-swallowing
    (record_hook_heartbeat never raises). Task 7's memory-health gate
    (check #3) reads these rows to detect a hook that silently died --
    the cp1252 gen_decisions_state crash class of bug that went
    unnoticed for a week.

    Mike-locked nudge-not-stop discipline applies here too: a claude-mem
    hook must never block or crash Claude Code's turn (see tool_use.py's
    docstring). So the ORIGINAL exception is recorded to the heartbeat
    then swallowed, not re-raised -- the failure is visible to the NEXT
    session's health gate instead of surfacing as a crash in this one.

    ProjectConfig construction is INSIDE the try-guard: a corrupted
    config.yaml (yaml.safe_load raise) or a Path.resolve() failure must
    degrade to no-heartbeat, never crash the hook -- the never-crash
    contract outranks heartbeat bookkeeping (Task-5 review carry-item).
    """
    telemetry_path = None
    try:
        telemetry_path = ProjectConfig(
            project_root=Path(project_root).resolve()
        ).telemetry_path
    except Exception:
        # Cannot even locate the telemetry DB (corrupt config / bad path).
        # Run the hook body without heartbeat rather than crash it.
        telemetry_path = None
    try:
        yield
    except Exception as exc:
        if telemetry_path is not None:
            record_hook_heartbeat(
                telemetry_path, hook=hook, ok=False, detail=repr(exc),
            )
    else:
        if telemetry_path is not None:
            record_hook_heartbeat(telemetry_path, hook=hook, ok=True, detail="")


def _hook_echo(message: str) -> None:
    """stdout write for HOOK entry points only (session-start, prompt-submit,
    session-end, capture-extract, capture-synthesize, tool-use,
    tool-use-post). The Claude Code process that spawned a hook can exit
    before a long hook body finishes; a write/flush to that dead pipe then
    raises OSError(22, 'Invalid argument') on Windows (POSIX raises
    BrokenPipeError, an OSError subclass), and a within-process closed
    stdout raises ValueError. Hook stdout is purely informational -- when
    the reader is gone the message has no destination -- so absorbing the
    write failure is the correct semantics, while letting it propagate is
    not: it aborts the REMAINING hook work (capture-extract's incremental
    ingest never ran on any SessionEnd from 2026-07-17 to 2026-07-21
    because the echo between mining and ingest raised first) and records a
    false hook-failure heartbeat. Interactive CLI commands keep plain
    click.echo -- for them a stdout failure is a real error that must
    surface."""
    try:
        click.echo(message)
    except (OSError, ValueError):
        pass


@cli.command()
@click.option(
    "--project-root", type=click.Path(file_okay=False), default=".",
    help="Project root directory (default: current dir).",
)
@click.option(
    "--isolate", is_flag=True,
    help="Set isolate_from_cross_project=true in this project's config.",
)
def init(project_root: str, isolate: bool) -> None:
    """Initialize .claude-mem/ in PROJECT_ROOT."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if isolate:
        cfg.values["isolate_from_cross_project"] = True
    cfg.write()
    init_db(cfg.db_path, embedding_dim=cfg.values["embedding_dim"])
    click.echo(f"Initialized {cfg.state_dir}")


@cli.command()
@click.option(
    "--project-root", type=click.Path(file_okay=False), default=".",
)
@click.argument("query")
@click.option("--top-k", type=int, default=None)
@click.option("--filter-do-not-rebuild", is_flag=True)
@click.option(
    "--cross-project", is_flag=True,
    help="Search the global cross-project index instead of the per-project one.",
)
def search(
    project_root: str,
    query: str,
    top_k: int | None,
    filter_do_not_rebuild: bool,
    cross_project: bool,
) -> None:
    """Run a hybrid BM25 + vector search."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cross_project and not cfg.db_path.is_file():
        click.echo(
            f"ERROR: no index at {cfg.db_path}. Run `claude-mem init` first.",
            err=True,
        )
        sys.exit(1)
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        embedding_dim=cfg.values["embedding_dim"],
    )
    if cross_project:
        results = search_cross_project(
            query, embedder,
            top_k=top_k or cfg.values["top_k_default"],
            filter_do_not_rebuild=filter_do_not_rebuild,
        )
    else:
        searcher = Searcher(db_path=cfg.db_path, embedder=embedder)
        try:
            results = searcher.search(
                query,
                top_k=top_k or cfg.values["top_k_default"],
                filter_do_not_rebuild=filter_do_not_rebuild,
            )
        finally:
            searcher.close()
    if not results:
        click.echo("no results")
        return
    for r in results:
        score = r.get("final_score", 0.0)
        src = r.get("source", "?")
        mod = r.get("module") or "-"
        origin = r.get("origin_project")
        prefix = f"[{origin}] " if origin else ""
        head = clip(r["content"], 140)
        click.echo(f"{prefix}[{score:.4f}] ({src} / {mod}) {head}")


@cli.command()
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--sample", type=int, default=0,
    help="If >0, dump the first N proposed chunks to bulk_sample.json "
         "instead of indexing.",
)
@click.option(
    "--include-git/--no-include-git", default=True,
    help="Include git log commits in the backfill.",
)
@click.option(
    "--include-claude-web", type=click.Path(exists=True), default=None,
    help="Path to a claude.ai export ZIP for additional ingestion.",
)
def bulk(
    project_root: str,
    sample: int,
    include_git: bool,
    include_claude_web: str | None,
) -> None:
    """Canonical backfill: convos + docs + tests + experiments + memory + git."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo("ERROR: run `claude-mem init` first.", err=True)
        sys.exit(1)
    modules = collect_module_names(root)
    dnr_modules = collect_do_not_rebuild_modules(root)
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        embedding_dim=cfg.values["embedding_dim"],
    )
    ingester = Ingester(db_path=cfg.db_path, embedder=embedder)
    proposed: list[dict] = []
    harness_filtered = 0

    def push(chunk: Chunk) -> bool:
        """Append to proposed sample; ingest immediately when not in sample
        mode. Returns False once the sample is full."""
        proposed.append({
            "id": chunk.id,
            "source": chunk.source,
            "module": chunk.module,
            "signal_weight": chunk.signal_weight,
            "is_correction": chunk.is_correction,
            "head": chunk.content[:120],
        })
        if not sample:
            ingester.add(chunk)
        if sample and len(proposed) >= sample:
            return False
        return True

    # 1. Claude Code session JSONLs for this project
    home_proj_dir = Path.home() / ".claude" / "projects"
    proj_slug = (
        str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    if home_proj_dir.is_dir():
        for jsonl in home_proj_dir.rglob(f"*{proj_slug}*/*.jsonl"):
            for msg in parse_claude_code_jsonl(jsonl):
                # Harness-content filter runs BEFORE phrase grading: a
                # skill-file body / hook-stdout echo / system-reminder is
                # injected into the transcript with role="user" but did
                # not originate from the human, so it must never reach
                # signal_weight()'s phrase grader (which would rank it a
                # "correction" merely for containing the bait phrase).
                if is_harness_content(msg["content"]):
                    harness_filtered += 1
                    continue
                sw = signal_weight(msg["content"], role=msg["role"])
                if sw == 0:
                    continue
                if not push(Chunk(
                    content=msg["content"],
                    source="claude_code",
                    role=msg["role"],
                    session_id=msg["session_id"],
                    date=msg.get("timestamp"),
                    module=detect_module(msg["content"], modules),
                    signal_weight=sw,
                    is_correction=is_correction(
                        msg["content"], role=msg["role"]),
                    is_decision=is_decision(msg["content"]),
                    aliases=derive_aliases(msg["content"]),
                )):
                    break
            if sample and len(proposed) >= sample:
                break

    # 2. Docs (*.md under docs/) — parse_markdown_doc returns a list of
    #    section-level chunks (heading-granularity); iterate like parse_memory_md.
    docs_dir = root / "docs"
    if docs_dir.is_dir() and (not sample or len(proposed) < sample):
        for md in docs_dir.rglob("*.md"):
            for chunk_data in parse_markdown_doc(md):
                _mod = detect_module(chunk_data["content"], modules)
                if not push(Chunk(
                    content=chunk_data["content"],
                    source="doc",
                    file_path=chunk_data["file_path"],
                    line_start=chunk_data.get("line_start"),
                    line_end=chunk_data.get("line_end"),
                    module=_mod,
                    signal_weight=doc_signal_weight(chunk_data["file_path"]),
                    is_decision=is_decision(chunk_data["content"]),
                    do_not_rebuild=(_mod in dnr_modules) or is_do_not_rebuild(chunk_data["content"]),
                    file_mtime=chunk_data["file_mtime"],
                    aliases=derive_aliases(chunk_data["content"]),
                )):
                    break
            if sample and len(proposed) >= sample:
                break

    # 3. Memory entries (per-project Claude Code memory)
    mem_dir = Path.home() / ".claude" / "projects" / proj_slug / "memory"
    if mem_dir.is_dir() and (not sample or len(proposed) < sample):
        for mf in mem_dir.rglob("*.md"):
            for chunk_data in parse_memory_md(mf):
                _mod = detect_module(chunk_data["content"], modules)
                if not push(Chunk(
                    content=chunk_data["content"],
                    source="memory",
                    file_path=chunk_data["file_path"],
                    module=_mod,
                    signal_weight=40,
                    is_decision=is_decision(chunk_data["content"]),
                    # Operator-curated correction files (feedback_*.md /
                    # invariant_*.md) are is_correction=1 by NAME -- the
                    # phrase scanner has ~zero recall on curated
                    # corrections (see bulk.is_curated_correction_file).
                    is_correction=is_curated_correction_file(
                        chunk_data["file_path"]),
                    do_not_rebuild=(_mod in dnr_modules) or is_do_not_rebuild(chunk_data["content"]),
                    file_mtime=chunk_data["file_mtime"],
                    aliases=derive_aliases(chunk_data["content"]),
                )):
                    break
            if sample and len(proposed) >= sample:
                break

    # 4. Experiment progress summaries
    exp = root / "experiments"
    if exp.is_dir() and (not sample or len(proposed) < sample):
        for prog in exp.rglob("progress.jsonl"):
            for chunk_data in parse_progress_jsonl(prog):
                if not push(Chunk(
                    content=chunk_data["content"],
                    source="experiment_summary",
                    file_path=chunk_data["file_path"],
                    signal_weight=30,
                    file_mtime=chunk_data["file_mtime"],
                    aliases=derive_aliases(chunk_data["content"]),
                )):
                    break
            if sample and len(proposed) >= sample:
                break

    # 5. Git log (gated by --include-git)
    if include_git and (not sample or len(proposed) < sample):
        for commit in parse_git_log(root, since="6 months ago"):
            if not push(Chunk(
                content=commit["content"],
                source="git",
                file_path=commit["file_path"],
                date=commit["date"],
                signal_weight=15,
                aliases=derive_aliases(commit["content"]),
            )):
                break

    ingester.close()  # release SQLite connection in both branches
    if sample:
        out = cfg.state_dir / "bulk_sample.json"
        out.write_text(_json.dumps(proposed[:sample], indent=2))
        click.echo(
            f"Wrote {len(proposed[:sample])}-chunk sample to {out}; "
            f"NO chunks indexed (dry-run). Review then re-run without --sample. "
            f"({harness_filtered} harness-content chunks filtered before grading.)"
        )
    else:
        click.echo(
            f"Bulk ingestion complete: {len(proposed)} chunks indexed. "
            f"({harness_filtered} harness-content chunks filtered before grading.)"
        )


@cli.command(name="session-start")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code hook JSON from stdin (cwd overrides --project-root).",
)
def session_start(project_root: str, stdin: bool) -> None:
    """Hook invoked by Claude Code SessionStart -- emits high-signal
    context to stdout for system-prompt injection."""
    session_id = ""
    if stdin:
        data = _read_stdin_json()
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
        session_id = data.get("session_id") or ""
        if not session_id:
            # Compact / resume-source SessionStart payloads can omit
            # session_id but still carry transcript_path
            # (<...>/<session_id>.jsonl). Derive the id from its stem so the
            # frozen-render guard (memory_health check 9) ARMS instead of
            # sitting in the disarmed "session unknown" fallback -- the
            # 2026-07-04 dormant-guard gap, where the guard silently never
            # fired because no session identity ever reached the meta triple.
            transcript_path = data.get("transcript_path") or ""
            if transcript_path:
                session_id = Path(transcript_path).stem
    with _heartbeat("session_start", project_root):
        out = _ss_hook.run(Path(project_root).resolve(), session_id=session_id)
        if out:
            _hook_echo(out)


@cli.command(name="prompt-submit")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--session-id", default=None)
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code hook JSON from stdin (prompt/cwd/session_id from JSON).",
)
@click.argument("prompt", required=False)
def prompt_submit(
    project_root: str, session_id: str | None, stdin: bool,
    prompt: str | None,
) -> None:
    """Hook invoked by Claude Code UserPromptSubmit -- emits
    DO NOT REBUILD warning + topic-targeted chunks when build-intent
    or investigation-intent fires."""
    if stdin:
        data = _read_stdin_json()
        prompt = data.get("prompt") or prompt
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
        session_id = data.get("session_id") or session_id
    if not prompt:
        return
    with _heartbeat("prompt_submit", project_root):
        # session_id enables the per-session injection damping (Task B2)
        # and stamps the wrapper_invocations telemetry row (Task B5).
        out = _ps_hook.run(
            prompt, Path(project_root).resolve(), session_id=session_id,
        )
        if out:
            _hook_echo(out)


@cli.command(name="session-end")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--session-id", default=None)
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code hook JSON from stdin (cwd/session_id from JSON).",
)
def session_end(
    project_root: str, session_id: str | None, stdin: bool,
) -> None:
    """Hook invoked by Claude Code Stop -- runs correction detection
    on the just-ended session."""
    if stdin:
        data = _read_stdin_json()
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
        session_id = data.get("session_id") or session_id
    with _heartbeat("session_end", project_root):
        out = _se_hook.run(session_id or "", Path(project_root).resolve())
        if out:
            _hook_echo(out)


@cli.command(name="capture-extract")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--session-id", default="", help="session id (when not using --stdin)")
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code hook JSON from stdin (cwd/session_id from JSON).",
)
@click.option(
    "--skip-incremental", is_flag=True,
    help="Skip the incremental-ingest step (candidate mining only). Used "
         "by tests / manual re-runs that want capture-extract's original "
         "narrow scope without also paying the incremental-ingest budget.",
)
@click.option(
    "--incremental-budget", "incremental_budget_s", type=float,
    default=None,
    help="Override the incremental-ingest wall-clock budget in seconds "
         "(default: hooks.session_end.DEFAULT_INCREMENTAL_BUDGET_S, 55.0).",
)
def capture_extract(
    project_root: str, session_id: str, stdin: bool,
    skip_incremental: bool, incremental_budget_s: float | None,
) -> None:
    """SessionEnd hook: mine the ended session for candidate decisions/
    dead-ends, THEN run budgeted incremental ingestion (spec R3). Ordering
    is code-controlled here (candidate mining first, then incremental
    ingest) rather than relying on settings.json hook-array ordering --
    both run inside this one SessionEnd invocation."""
    from .hooks import session_end as _se_mod

    if stdin:
        data = _read_stdin_json()
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
        session_id = data.get("session_id") or session_id
    root = Path(project_root).resolve()
    with _heartbeat("capture_extract", project_root):
        msg = _se_mod.run_candidates(session_id, root)
        if msg:
            _hook_echo(msg)
        if not skip_incremental:
            kwargs = {}
            if incremental_budget_s is not None:
                kwargs["budget_s"] = incremental_budget_s
            incr_msg = _se_mod.run_incremental_ingest(root, **kwargs)
            if incr_msg:
                _hook_echo(incr_msg)


def _get_capture_store(project_root: str):
    from .capture import CaptureStore
    cfg = ProjectConfig(project_root=Path(project_root).resolve())
    return CaptureStore(cfg.db_path)


@cli.command(name="thread-add")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--name", required=True)
@click.option("--summary", default="")
@click.option("--state", default="open")
def thread_add(project_root: str, name: str, summary: str, state: str) -> None:
    """Author a problem-thread row."""
    from .capture import Thread
    store = _get_capture_store(project_root)
    try:
        added = store.add_thread(Thread(name=name, summary=summary, state=state))
    finally:
        store.close()
    click.echo("added" if added else "exists")


@cli.command(name="decision-add")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--title", required=True)
@click.option("--date", default="")
@click.option("--rationale", default="")
@click.option("--thread", "thread_id", default="")
@click.option("--rejected", "rejected", multiple=True)
@click.option("--commit", "commits", multiple=True)
@click.option("--confirmed", is_flag=True, help="mark state=confirmed (not pending)")
@click.option("--mike-approved", is_flag=True)
def decision_add(project_root, title, date, rationale, thread_id, rejected,
                 commits, confirmed, mike_approved) -> None:
    """Author a decision row."""
    from .capture import Decision
    store = _get_capture_store(project_root)
    try:
        added = store.add_decision(Decision(
            title=title, date=date or None, rationale=rationale,
            options_rejected=list(rejected), thread_id=thread_id or None,
            linked_commits=list(commits),
            state="confirmed" if confirmed else "pending",
            mike_approved=mike_approved,
        ))
    finally:
        store.close()
    click.echo("added" if added else "exists")


@cli.command(name="dead-end-add")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--approach", required=True)
@click.option("--date", default="")
@click.option("--why", "why_shelved", default="")
@click.option("--superseded-by", "superseded_by", default="")
@click.option("--thread", "thread_id", default="")
@click.option("--confirmed", is_flag=True)
def dead_end_add(project_root, approach, date, why_shelved, superseded_by,
                 thread_id, confirmed) -> None:
    """Author a dead-end row."""
    from .capture import DeadEnd
    store = _get_capture_store(project_root)
    try:
        added = store.add_dead_end(DeadEnd(
            approach=approach, date=date or None, why_shelved=why_shelved,
            superseded_by=superseded_by, thread_id=thread_id or None,
            state="confirmed" if confirmed else "pending",
        ))
    finally:
        store.close()
    click.echo("added" if added else "exists")


@cli.command(name="capture-backfill-chunks")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--limit", type=int, default=50,
              help="max is_decision chunks to backfill (highest signal first)")
def capture_backfill_chunks(project_root: str, limit: int) -> None:
    """Backfill pending decision rows from existing is_decision-tagged chunks."""
    import sqlite3 as _sql
    from .capture import CaptureStore, Decision

    cfg = ProjectConfig(project_root=Path(project_root).resolve())
    conn = _sql.connect(str(cfg.db_path))
    conn.row_factory = _sql.Row
    rows = conn.execute(
        "SELECT content, date FROM chunks "
        "WHERE is_decision=1 AND source IN ('doc', 'memory') "
        "ORDER BY signal_weight DESC, ingested_at DESC"
    ).fetchall()
    conn.close()

    total = len(rows)
    capped = rows[:limit]
    store = CaptureStore(cfg.db_path)
    written = 0
    try:
        for r in capped:
            content = r["content"] or ""
            # Whole-word clip (was a bare [:120] mid-word slice stored
            # permanently); the rationale carries the chunk's FULL first
            # sentence (whitespace-collapsed, word-boundary capped at
            # 600) instead of a static marker, so triage sees the
            # untruncated claim. Rows written before 2026-08-19 are NOT
            # migrated -- their titles were clipped at write time.
            title = clip(content, 200)
            if not title:
                continue
            if store.add_decision(Decision(
                title=title, date=r["date"] or None, state="pending",
                rationale=clip(first_sentence(content), 600),
            )):
                written += 1
    finally:
        store.close()
    dropped = max(0, total - len(capped))
    click.echo(
        f"backfill: {total} is_decision doc/memory chunks matched; {written} "
        f"pending decisions written; {dropped} beyond --limit={limit} not "
        f"backfilled. (claude_code conversation chunks excluded as low-precision.)"
    )


@cli.command(name="capture-synthesize")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--thread", "thread_id", default="", help="one thread id (default: all)")
@click.option("--endpoint", default="", help="override the Ollama endpoint")
def capture_synthesize(project_root: str, thread_id: str, endpoint: str) -> None:
    """Regenerate stale per-thread lineages via local Ollama (off the hot path)."""
    from .capture import CaptureStore
    from .generate import GenerationClient, GenerationError
    from .synthesize import get_or_regenerate_lineage

    cfg = ProjectConfig(project_root=Path(project_root).resolve())
    client = GenerationClient(
        model=cfg.values["generation_model"],
        endpoint=endpoint or cfg.values["ollama_endpoint"],
    )
    store = CaptureStore(cfg.db_path)
    regen = failed = fresh = 0
    with _heartbeat("capture_synthesize", project_root):
        try:
            threads = (
                [store.get_thread(thread_id)] if thread_id else store.list_threads()
            )
            for t in threads:
                if not t:
                    continue
                cached, key = store.get_cached_lineage(t["id"])
                if cached and key == t["last_updated"]:
                    fresh += 1
                    continue
                try:
                    get_or_regenerate_lineage(store, t["id"], client.generate)
                    regen += 1
                except GenerationError:
                    failed += 1
        finally:
            store.close()
            client.close()
    _hook_echo(
        f"capture-synthesize: {regen} thread lineage(s) regenerated; "
        f"{fresh} already fresh; {failed} failed (Ollama unreachable -- "
        f"prompt-time fallback covers these)."
    )


@cli.command(name="capture-list")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--pending", is_flag=True, help="only pending candidates")
def capture_list(project_root: str, pending: bool) -> None:
    """List threads, decisions, and dead-ends."""
    store = _get_capture_store(project_root)
    try:
        threads = store.list_threads()
        state = "pending" if pending else None
        decisions = store.list_decisions(state=state)
        dead_ends = store.list_dead_ends(state=state)
    finally:
        store.close()
    for t in threads:
        click.echo(f"THREAD [{t['state']}] {t['name']} (updated {t['last_updated']})")
    for d in decisions:
        tag = " (Mike-approved)" if d["mike_approved"] else ""
        click.echo(f"DECISION [{d['state']}]{tag} {d['id']} {d['date']} {d['title']}")
    for e in dead_ends:
        click.echo(f"DEAD-END [{e['state']}] {e['id']} {e['date']} {e['approach']}")


@cli.command(name="decision-confirm")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--id", "decision_id", required=True)
@click.option("--mike-approved/--no-mike-approved", default=None)
def decision_confirm(project_root: str, decision_id: str, mike_approved) -> None:
    """Mark a pending decision confirmed (optionally Mike-approved)."""
    store = _get_capture_store(project_root)
    try:
        store.confirm_decision(decision_id, mike_approved=mike_approved)
    finally:
        store.close()
    click.echo("confirmed")


@cli.command(name="capture-triage")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--limit", type=int, default=20,
    help="Max pending decisions + max pending dead-ends to render (each "
         "capped independently), most recent first (default: 20).",
)
@click.option(
    "--apply", "apply_json", default="",
    help="JSON array of verdicts: "
         '[{"id": <decision_or_dead_end_id>, "verdict": "confirm"|"reject", '
         '"thread": <thread_id>|"new:<Thread Name>" (optional), '
         '"title_edit": <str> (optional)}]. When set, renders nothing and '
         "applies the batch instead.",
)
def capture_triage(project_root: str, limit: int, apply_json: str) -> None:
    """Render pending decisions/dead-ends as a numbered review sheet, or
    apply a batch of confirm/reject/attach verdicts (spec R6).

    Review-sheet mode (default): the operator triage smoke pass this
    supports -- read the sheet, decide what's a genuine decision worth
    keeping vs. harness/fragment junk, then re-invoke with --apply.

    --apply mode: looks each id up in BOTH decisions and dead_ends (ids are
    content-SHA and do not collide across the two tables in practice), sets
    state=confirmed/rejected, and -- when "thread" is given -- attaches the
    row to that thread (creating it first if the id is "new:<Name>") via
    set_decision_thread/set_dead_end_thread, which bumps the thread's
    last_updated and so unfreezes synthesize's lineage cache gate. Unknown
    ids are reported in the output, never raised (an --apply batch must not
    abort partway through on one bad id)."""
    from .capture import CaptureStore, Thread

    store = _get_capture_store(project_root)
    try:
        if apply_json:
            try:
                verdicts = _json.loads(apply_json)
            except _json.JSONDecodeError as exc:
                click.echo(f"ERROR: --apply is not valid JSON: {exc}", err=True)
                sys.exit(1)
            confirmed = rejected = attached = unknown = 0
            new_thread_cache: dict[str, str] = {}
            for v in verdicts:
                vid = v.get("id", "")
                verdict = v.get("verdict", "")
                thread_spec = v.get("thread") or ""
                title_edit = v.get("title_edit")

                is_decision_row = store._conn.execute(
                    "SELECT 1 FROM decisions WHERE id=?", (vid,)
                ).fetchone() is not None
                is_dead_end_row = (not is_decision_row) and store._conn.execute(
                    "SELECT 1 FROM dead_ends WHERE id=?", (vid,)
                ).fetchone() is not None

                if not is_decision_row and not is_dead_end_row:
                    click.echo(f"UNKNOWN id (no matching decision/dead-end): {vid}")
                    unknown += 1
                    continue

                thread_id = None
                if thread_spec:
                    if thread_spec.startswith("new:"):
                        thread_name = thread_spec[len("new:"):]
                        thread_id = new_thread_cache.get(thread_name)
                        if thread_id is None:
                            t = Thread(name=thread_name)
                            store.add_thread(t)  # no-op if it already exists
                            thread_id = t.id
                            new_thread_cache[thread_name] = thread_id
                    else:
                        thread_id = thread_spec

                if is_decision_row:
                    if verdict == "confirm":
                        store.confirm_decision(vid)
                        confirmed += 1
                    elif verdict == "reject":
                        store.reject_decision(vid)
                        rejected += 1
                    if thread_id:
                        store.set_decision_thread(vid, thread_id, title=title_edit)
                        attached += 1
                    elif title_edit is not None:
                        store._conn.execute(
                            "UPDATE decisions SET title=? WHERE id=?",
                            (title_edit, vid),
                        )
                        store._conn.commit()
                else:
                    if verdict == "confirm":
                        store.confirm_dead_end(vid)
                        confirmed += 1
                    elif verdict == "reject":
                        store.reject_dead_end(vid)
                        rejected += 1
                    if thread_id:
                        store.set_dead_end_thread(vid, thread_id, approach=title_edit)
                        attached += 1
                    elif title_edit is not None:
                        store._conn.execute(
                            "UPDATE dead_ends SET approach=? WHERE id=?",
                            (title_edit, vid),
                        )
                        store._conn.commit()
            click.echo(
                f"capture-triage --apply: {confirmed} confirmed, {rejected} "
                f"rejected, {attached} attached to a thread, {unknown} unknown "
                f"id(s) skipped."
            )
            return

        decisions = store.list_decisions(state="pending")[:limit]
        dead_ends = store.list_dead_ends(state="pending")[:limit]
    finally:
        store.close()

    if not decisions and not dead_ends:
        click.echo("capture-triage: no pending decisions or dead-ends.")
        return

    click.echo(
        f"=== CAPTURE TRIAGE -- {len(decisions)} pending decision(s), "
        f"{len(dead_ends)} pending dead-end(s) (most recent first, "
        f"--limit={limit}) ==="
    )
    click.echo(
        "Review each row: genuine decision/dead-end worth keeping -> "
        '{"id": ..., "verdict": "confirm"} (add "thread": "<id>" or '
        '"new:<Name>" to attach; "title_edit": "..." to clean up the '
        'title); harness/fragment junk -> {"id": ..., "verdict": "reject"}. '
        "Feed the array to --apply."
    )
    click.echo("")
    if decisions:
        click.echo("-- DECISIONS --")
        for i, d in enumerate(decisions, 1):
            click.echo(f"{i}. [{d['date']}] id={d['id']} {d['title']}")
    if dead_ends:
        click.echo("-- DEAD-ENDS --")
        for i, e in enumerate(dead_ends, 1):
            click.echo(f"{i}. [{e['date']}] id={e['id']} {e['approach']}")


@cli.command(name="tool-use")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--tool-name", default=None,
              help="Tool name (Edit / Write / Bash / Agent / NotebookEdit). "
                   "Read from stdin JSON when --stdin is set.")
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code hook JSON from stdin (tool_name / tool_input / "
         "cwd from JSON).",
)
def tool_use(
    project_root: str, tool_name: str | None, stdin: bool,
) -> None:
    """Hook invoked by Claude Code PreToolUse -- watches the assistant's
    tool inputs for build/investigation intent against existing-subsystem
    matches. Emits a NUDGE (not a block) when meaningful overlap exists.

    Mike-locked design (2026-05-25): nudge-not-stop. The hook always
    exits 0; stdout becomes a system reminder the assistant sees before
    the next message. The tool call itself is never blocked."""
    tool_input: dict = {}
    if stdin:
        data = _read_stdin_json()
        tool_name = data.get("tool_name") or tool_name
        tool_input = data.get("tool_input") or {}
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
    if not tool_name:
        return
    with _heartbeat("tool_use", project_root):
        out = _tu_hook.run(tool_name, tool_input, Path(project_root).resolve())
        if out:
            _hook_echo(out)


@cli.command(name="tool-use-post")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--stdin", is_flag=True,
    help="Read Claude Code PostToolUse hook JSON from stdin (tool_name / "
         "tool_input / tool_response / session_id / cwd).",
)
def tool_use_post_cmd(project_root: str, stdin: bool) -> None:
    """Hook invoked by Claude Code PostToolUse -- work-aware injection.
    Searches the index against the finished tool's input + result and emits
    at most one novel, relevant prior-work nudge. Nudge-not-stop: always
    exits 0; stdout becomes a system reminder the assistant sees before the
    next message."""
    from .hooks import tool_use_post as _tup_hook
    tool_name = None
    tool_input: dict = {}
    tool_response: object = ""
    session_id = ""
    if stdin:
        data = _read_stdin_json()
        tool_name = data.get("tool_name")
        tool_input = data.get("tool_input") or {}
        tool_response = data.get("tool_response", "")
        session_id = data.get("session_id") or ""
        cwd = data.get("cwd")
        if cwd:
            project_root = cwd
    if not tool_name:
        return
    with _heartbeat("tool_use_post", project_root):
        out = _tup_hook.run(
            tool_name, tool_input, tool_response, session_id,
            Path(project_root).resolve(),
        )
        if out:
            _hook_echo(out)


# Fail-loud shell shims. Each hook:
#   - captures claude-mem stderr to /tmp/claude-mem-err.$$
#   - emits a one-line [claude-mem WARN] message to stderr on failure
#   - always exits 0 so Claude Code is never blocked
# Mike-locked 2026-05-24: fail-loud (silent failure burned us with
# Virgil's watcher).
_HOOK_TEMPLATES = {
    "SessionStart.sh": """#!/bin/bash
ERR="/tmp/claude-mem-err.$$"
if ! claude-mem session-start --stdin 2>"$ERR"; then
    echo "[claude-mem WARN] session-start failed; see $ERR" >&2
fi
exit 0
""",
    "UserPromptSubmit.sh": """#!/bin/bash
ERR="/tmp/claude-mem-err.$$"
if ! claude-mem prompt-submit --stdin 2>"$ERR"; then
    echo "[claude-mem WARN] prompt-submit failed; see $ERR" >&2
fi
exit 0
""",
    "Stop.sh": """#!/bin/bash
ERR="/tmp/claude-mem-err.$$"
( if ! claude-mem session-end --stdin 2>"$ERR"; then
      echo "[claude-mem WARN] session-end failed; see $ERR" >&2
  fi
) &
disown
exit 0
""",
    "PreToolUse.sh": """#!/bin/bash
ERR="/tmp/claude-mem-err.$$"
if ! claude-mem tool-use --stdin 2>"$ERR"; then
    echo "[claude-mem WARN] tool-use failed; see $ERR" >&2
fi
exit 0
""",
}


# Hook events registered in settings.json by default. SessionStart fires
# once per session (one-time cost is acceptable); UserPromptSubmit fires
# per turn and is timeout-capped. Stop is intentionally NOT registered
# in settings: correction-extraction over the project's JSONL history
# is slow + blocks Claude Code's next-prompt acceptance. The .sh shim
# at ~/.claude/hooks/Stop.sh backgrounds the call with `& disown` and
# remains available for users who opt in via shim-based wiring.
_SETTINGS_EVENT_TO_ENTRY = {
    "SessionStart":     {"subcmds": [
        # 10000 (was 5000): production-calibrated — the curated render over
        # a large index can exceed 5s on session start (origin-project
        # wiring 2026-08).
        {"subcmd": "session-start", "timeout": 10000, "stdin": True,
         "project_root": False},
    ]},
    "UserPromptSubmit": {"subcmds": [
        {"subcmd": "prompt-submit", "timeout": 3000, "stdin": True,
         "project_root": False},
    ]},
    # PreToolUse fires per assistant tool call. Match on the
    # write-class tools only (extraction is no-op on the rest, but the
    # matcher avoids paying invocation cost for Read / Grep / Glob /
    # TodoWrite). Timeout is short -- nudges are advisory; if claude-mem
    # cannot respond fast, the assistant proceeds without one.
    "PreToolUse":       {
        "matcher": "Edit|Write|Bash|Agent|NotebookEdit",
        "subcmds": [
            {"subcmd": "tool-use", "timeout": 3000, "stdin": True,
             "project_root": False},
        ],
    },
    # PostToolUse fires after each assistant tool call. Work-aware injection:
    # search the index against what Claude just did (input + result) and
    # surface one novel, relevant prior-work match. Advisory; short timeout.
    "PostToolUse":      {
        "matcher": "Edit|Write|Bash|Agent",
        "subcmds": [
            {"subcmd": "tool-use-post", "timeout": 3000, "stdin": True,
             "project_root": False},
        ],
    },
    # SessionEnd runs the capture pipeline. These three are load-bearing
    # for the memory-health gate's hook-heartbeat checks (a project wired
    # without them reports RED heartbeats). Timeouts are production-
    # calibrated (origin-project wiring 2026-08): synthesis is LLM-backed and can
    # take two minutes after a long session. capture-synthesize takes no
    # --stdin (it scans the project's transcript history itself) and both
    # capture steps bake an explicit --project-root so they are
    # cwd-independent.
    "SessionEnd":       {"subcmds": [
        {"subcmd": "session-end", "timeout": 60000, "stdin": True,
         "project_root": True},
        {"subcmd": "capture-extract", "timeout": 60000, "stdin": True,
         "project_root": True},
        {"subcmd": "capture-synthesize", "timeout": 120000, "stdin": False,
         "project_root": True},
    ]},
}

# Public alias: THE single source of truth for claude-mem's settings-file
# hook wiring. harness/init.py builds its merge plan from this table so the
# canonical hook set is never duplicated across the two registrars (they
# keep different merge SEMANTICS by design: _patch_settings prunes and
# re-adds claude-mem entries; harness init preserves pre-existing wiring).
SETTINGS_HOOK_TABLE = _SETTINGS_EVENT_TO_ENTRY


def _hook_command(exe_path: str, spec: dict, project_root: str | None) -> str:
    """Render one settings-file hook command from a SETTINGS_HOOK_TABLE spec."""
    quoted = f'"{exe_path}"' if " " in exe_path else exe_path
    command = f"{quoted} {spec['subcmd']}"
    if spec.get("stdin", True):
        command += " --stdin"
    if project_root and spec.get("project_root", False):
        command += f" --project-root {project_root}"
    return command


def _patch_settings(
    settings_path: Path, exe_path: str, project_root: str | None = None,
) -> None:
    """Register claude-mem hooks in a Claude Code settings.json file.

    Idempotent: removes any prior `claude-mem` entries (across ALL hook
    events including ones no longer registered) before adding the fresh
    ones, so re-running the command does not duplicate AND prunes legacy
    entries like a previously-registered Stop hook. Preserves unrelated
    hooks (e.g. project state generators) untouched. When ``project_root``
    is given, the subcommands flagged ``project_root`` in the table bake
    an explicit ``--project-root`` (cwd-independent; recommended).
    """
    if settings_path.is_file():
        data = _json.loads(settings_path.read_text(encoding="utf-8"))
    else:
        data = {}
    hooks_root = data.setdefault("hooks", {})
    for event, entries in list(hooks_root.items()):
        if not isinstance(entries, list):
            continue
        for entry in entries:
            entry["hooks"] = [
                h for h in entry.get("hooks", [])
                if "claude-mem" not in (h.get("command") or "")
            ]
        hooks_root[event] = [e for e in entries if e.get("hooks")]
    for event, cfg in _SETTINGS_EVENT_TO_ENTRY.items():
        entries = hooks_root.setdefault(event, [])
        wanted_matcher = cfg.get("matcher", "")
        target = next(
            (e for e in entries if e.get("matcher", "") == wanted_matcher),
            None,
        )
        if target is None:
            target = {"matcher": wanted_matcher, "hooks": []}
            entries.append(target)
        for spec in cfg["subcmds"]:
            target["hooks"].append({
                "type": "command",
                "command": _hook_command(exe_path, spec, project_root),
                "timeout": spec["timeout"],
            })
        hooks_root[event] = [e for e in entries if e.get("hooks")]
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        _json.dumps(data, indent=2) + "\n", encoding="utf-8",
    )


@cli.command(name="extract-corrections")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--session-id", default=None,
    help="If set, only process the session whose JSONL stem matches.",
)
def extract_corrections_cmd(project_root: str, session_id: str | None) -> None:
    """Scan ~/.claude/projects/<project-slug>/*.jsonl; emit corrections
    and apply auto-boost + correction-event indexing."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo("ERROR: run `claude-mem init` first.", err=True)
        sys.exit(1)
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        embedding_dim=cfg.values["embedding_dim"],
    )
    proj_slug = (
        str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    home_proj_dir = Path.home() / ".claude" / "projects"
    total_events = 0
    total_applied = 0
    if home_proj_dir.is_dir():
        for jsonl in home_proj_dir.rglob(f"*{proj_slug}*/*.jsonl"):
            if session_id and jsonl.stem != session_id:
                continue
            events = extract_corrections(jsonl)
            if not events:
                continue
            total_events += len(events)
            total_applied += apply_corrections(
                events, cfg.db_path, embedder=embedder,
            )
    click.echo(
        f"Detected {total_events} correction events; applied {total_applied}."
    )


@cli.command(name="heartbeat")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
def heartbeat(project_root: str) -> None:
    """Probe each component (ollama, index); record a heartbeat row.

    Cron-friendly: schedule every 5 min. Output one line per component."""
    cfg = ProjectConfig(project_root=Path(project_root).resolve())
    statuses = probe_components(cfg)
    for component, (status, detail) in statuses.items():
        click.echo(f"{component}: {status}  detail={detail}")


@cli.command(name="prune-candidates")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--min-age-days", type=int, default=30,
    help="Only flag chunks ingested more than N days ago (default: 30).",
)
@click.option(
    "--limit", type=int, default=50,
    help="Maximum number of candidates to list (default: 50).",
)
def prune_candidates(project_root: str, min_age_days: int, limit: int) -> None:
    """List never-surfaced stale chunks (access_count=0 AND old).

    The actionable signal from the usage-feedback wire: memories that have
    never been retrieved/injected since ingestion AND are older than
    --min-age-days are prune candidates. REPORT ONLY -- never auto-deletes
    (Rule #2: archive, never delete experiment/memory data). Use this to
    decide what to manually review for removal."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo("ERROR: run `claude-mem init` first.", err=True)
        sys.exit(1)
    from claude_mem.maintenance import find_prune_candidates
    rows = find_prune_candidates(
        cfg.db_path, min_age_days=min_age_days, limit=limit,
    )
    if not rows:
        click.echo(
            f"No prune candidates (no access_count=0 chunks older than "
            f"{min_age_days} days)."
        )
        return
    click.echo(
        f"Prune candidates: {len(rows)} never-surfaced chunk(s) older than "
        f"{min_age_days} days (REPORT ONLY -- not deleted):"
    )
    for r in rows:
        topic = r.get("module") or "-"
        src = r.get("source") or "?"
        age = r.get("age_days")
        age_str = f"{age}d" if age is not None else "?"
        preview = (r.get("content") or "")[:100].replace("\n", " ")
        click.echo(f"  [{src} / {topic}] age={age_str} id={r['id'][:12]} {preview}")


@cli.command(name="maintenance")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--retag-corrections", "retag_corrections_flag", is_flag=True,
    help="Apply the curated-correction rule (memory chunks from "
         "feedback_*.md / invariant_*.md files get is_correction=1) to "
         "already-ingested chunks. Idempotent, UPDATE-only; run once on "
         "a live index after upgrading past 2026-08-19.",
)
def maintenance_cmd(project_root: str, retag_corrections_flag: bool) -> None:
    """Maintenance passes over the index (flag retags; report surfaces).

    Currently: --retag-corrections. Never deletes anything (Rule #2)."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo("ERROR: run `claude-mem init` first.", err=True)
        sys.exit(1)
    if not retag_corrections_flag:
        click.echo(
            "maintenance: no pass selected. Available: --retag-corrections."
        )
        return
    from claude_mem.maintenance import retag_corrections
    counts = retag_corrections(cfg.db_path)
    click.echo(
        f"retag-corrections: {counts['scanned']} memory chunk(s) scanned; "
        f"{counts['matched']} matched the feedback_/invariant_ rule; "
        f"{counts['retagged']} newly tagged is_correction=1; "
        f"{counts['already_tagged']} already tagged."
    )


@cli.command(name="migrate-regrade")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--backup-suffix", default=None,
    help="Backup file suffix: writes <db>.bak-<suffix> before touching the "
         "DB. Defaults to a per-run timestamp -- the old fixed default "
         "aimed every run at the same file and destroyed a prior backup; "
         "_backup now also refuses to overwrite an existing one.",
)
def migrate_regrade_cmd(project_root: str, backup_suffix: str | None) -> None:
    """Corpus re-grade + module-tag migration (spec R2a).

    Demotes chunks that phrase-graded as high-signal before the Task-1
    harness-content filter existed (sw=0, is_correction=0, is_decision=0;
    module untouched), and retags NULL-module chunks via detect_module()
    plus a doc-path fallback. UPDATE-only, one transaction, idempotent.
    Refuses to run unless the pre-migration backup copy succeeds."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo(f"ERROR: no index at {cfg.db_path}. Run `claude-mem init` first.", err=True)
        sys.exit(1)
    if backup_suffix is None:
        from datetime import datetime, timezone
        backup_suffix = datetime.now(timezone.utc).strftime(
            "regrade-%Y%m%d-%H%M%S")
    try:
        summary = migrate_regrade(
            cfg.db_path, backup_suffix=backup_suffix, project_root=root,
        )
    except RuntimeError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"migrate-regrade: backed up to {summary['backed_up_to']}; "
        f"{summary['demoted']} demoted, {summary['retagged_module']} "
        f"module-retagged, {summary['demoted_assistant_corrections']} "
        f"assistant-corrections cleared, {summary['unchanged']} unchanged."
    )
    if summary["demoted_by_source"]:
        click.echo("demoted by source:")
        for src, count in sorted(
            summary["demoted_by_source"].items(), key=lambda kv: -kv[1]
        ):
            click.echo(f"  {src}: {count}")


@cli.command(name="embed-backfill")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--batch", "batch_size", type=int, default=200,
    help="Chunks embedded per commit (default: 200). Smaller batches "
         "commit more often -- less work lost if the process is killed, "
         "at the cost of more frequent fsync overhead.",
)
@click.option(
    "--resume", is_flag=True,
    help="Explicit opt-in marker for resuming an interrupted backfill. "
         "The underlying pass is ALWAYS resumable by construction (only "
         "chunks currently missing a chunks_vec row are ever selected), "
         "so this flag does not change behavior -- it documents operator "
         "intent and is accepted so a resumed invocation is self-"
         "describing in shell history / logs.",
)
def embed_backfill_cmd(project_root: str, batch_size: int, resume: bool) -> None:
    """Embed every chunk missing a chunks_vec row (spec R2b).

    Resumable: selects chunks via `chunks LEFT JOIN chunks_vec WHERE
    chunk_id IS NULL`, so chunks already embedded (by a prior run, or by
    normal ingestion) are never re-embedded. Commits after EVERY chunk
    (never on a batch cadence -- see embed_backfill's module docstring:
    holding the write lock across a slow embed() network call starves
    concurrent readers). --batch instead gates progress-line cadence.
    Every embedding failure is logged to ingestion_log
    (action='embed_fail', detail=<exception text>) and counted in the
    summary -- never silently dropped. Uses embed.BULK_READ_TIMEOUT_S
    (60s) instead of the hook-tuned 2.0s default so a slow-but-alive
    Ollama daemon (e.g. contended by a concurrent generation-model
    call) is awaited rather than abandoned.
    """
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo(
            f"ERROR: no index at {cfg.db_path}. Run `claude-mem init` first.",
            err=True,
        )
        sys.exit(1)
    from claude_mem.embed import BULK_READ_TIMEOUT_S
    from claude_mem.embed_backfill import backfill
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        read_timeout_s=BULK_READ_TIMEOUT_S,
        embedding_dim=cfg.values["embedding_dim"],
    )

    def _progress(done: int, total: int) -> None:
        click.echo(f"  ...{done}/{total} pending chunks processed", err=True)

    try:
        summary = backfill(
            cfg.db_path, embedder=embedder, batch_size=batch_size,
            progress_cb=_progress,
        )
    except RuntimeError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    click.echo(
        f"embed-backfill: {summary['total']} total chunks; "
        f"{summary['embedded']} embedded, {summary['failed']} failed, "
        f"{summary['skipped']} already had vectors."
    )
    if summary["failed"]:
        click.echo(
            f"  {summary['failed']} failure(s) logged to ingestion_log "
            f"(action='embed_fail') -- re-run embed-backfill --resume to retry."
        )


@cli.command(name="ingest-incremental")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option(
    "--budget", "budget_s", type=float, default=55.0,
    help="Hard wall-clock budget in seconds for the whole run (default: "
         "55.0, leaving headroom inside a SessionEnd hook's ~60s ceiling). "
         "The deadline is checked between files/sessions, never mid-file, "
         "so a run that exceeds it stops cleanly and resumes forward on "
         "the next invocation via the incr:* watermarks.",
)
def ingest_incremental_cmd(project_root: str, budget_s: float) -> None:
    """Budgeted, watermarked, resumable incremental ingestion (spec R3).

    Complements `claude-mem bulk` (a full canonical re-scan): scans docs/
    memory/ledgers by mtime and sessions by byte offset, ingesting only
    what changed since the last run. Safe to call every SessionEnd --
    embeds inline using embed.BULK_READ_TIMEOUT_S (60s) so a slow-but-
    alive Ollama daemon is awaited rather than abandoned, and every embed
    failure is logged to ingestion_log (action='embed_fail') instead of
    silently dropped.
    """
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    if not cfg.db_path.is_file():
        click.echo(
            f"ERROR: no index at {cfg.db_path}. Run `claude-mem init` first.",
            err=True,
        )
        sys.exit(1)
    from claude_mem.embed import BULK_READ_TIMEOUT_S
    from claude_mem.incremental import run_incremental
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
        read_timeout_s=BULK_READ_TIMEOUT_S,
        embedding_dim=cfg.values["embedding_dim"],
    )
    try:
        summary = run_incremental(root, embedder=embedder, budget_s=budget_s)
    except RuntimeError as exc:
        click.echo(f"ERROR: {exc}", err=True)
        sys.exit(1)
    total_files = sum(s["files_ingested"] for s in summary.values())
    total_chunks = sum(s["chunks_added"] for s in summary.values())
    click.echo(
        f"ingest-incremental: {total_files} file(s) ingested, "
        f"{total_chunks} chunk(s) added "
        f"(docs={summary['docs']['files_ingested']}, "
        f"memory={summary['memory']['files_ingested']}, "
        f"ledgers={summary['ledgers']['files_ingested']}, "
        f"sessions={summary['sessions']['files_ingested']})."
    )


@cli.command(name="report")
@click.option("--project-root", type=click.Path(file_okay=False), default=".")
@click.option("--days", type=int, default=7)
def report(project_root: str, days: int) -> None:
    """Generate the rolling weekly summary; write to
    docs/marathon/memory_system_weekly/ (or .claude-mem/reports/)."""
    root = Path(project_root).resolve()
    cfg = ProjectConfig(project_root=root)
    out = write_weekly_summary(cfg.telemetry_path, root, days=days)
    click.echo(f"Wrote {out}")


@cli.command(name="install-hooks")
@click.option(
    "--hook-dir",
    type=click.Path(file_okay=False),
    default=str(Path.home() / ".claude" / "hooks"),
    help="Where to write the shell shims (default: ~/.claude/hooks/).",
)
@click.option(
    "--settings-path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also register hooks in this Claude Code settings JSON file "
         "(e.g. .claude/settings.local.json or ~/.claude/settings.json). "
         "Uses an absolute path to the claude-mem executable so PATH does "
         "not need to include the install location.",
)
@click.option(
    "--no-shims", is_flag=True,
    help="Skip writing .sh shims to --hook-dir.",
)
@click.option(
    "--exe-path",
    type=click.Path(dir_okay=False),
    default=None,
    help="Override the resolved claude-mem executable path written into "
         "settings (default: shutil.which('claude-mem')).",
)
@click.option(
    "--project-root",
    type=click.Path(file_okay=False),
    default=None,
    help="Bake an explicit --project-root into the subcommands that take "
         "one (the SessionEnd capture pipeline), making the wiring "
         "cwd-independent. Recommended when registering per-project "
         "settings files.",
)
def install_hooks(
    hook_dir: str, settings_path: str | None, no_shims: bool,
    exe_path: str | None, project_root: str | None,
) -> None:
    """Install claude-mem hook integration for Claude Code.

    Default behavior writes three fail-loud shell shims (SessionStart,
    UserPromptSubmit, Stop) under --hook-dir. With --settings-path, also
    registers the hooks directly in a Claude Code settings JSON file
    using an absolute path to the claude-mem executable -- preferred on
    Windows where the install location is not on PATH.
    """
    if not no_shims:
        target = Path(hook_dir)
        target.mkdir(parents=True, exist_ok=True)
        for name, content in _HOOK_TEMPLATES.items():
            p = target / name
            p.write_text(content, encoding="utf-8")
            try:
                p.chmod(0o755)
            except OSError:
                pass  # Windows: chmod is a no-op
            click.echo(f"wrote {p}")
    if settings_path:
        exe = exe_path or shutil.which("claude-mem") or "claude-mem"
        _patch_settings(Path(settings_path), exe, project_root=project_root)
        click.echo(f"registered hooks in {settings_path} (exe: {exe})")


if __name__ == "__main__":
    cli()
