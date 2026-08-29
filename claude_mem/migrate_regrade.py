"""Corpus re-grade + module-tag migration (spec R2a).

Two independent repairs applied to an EXISTING index.db in one pass:

1. Demotion: chunks ingested before the Task-1 harness-content filter
   (`claude_mem.filters.is_harness_content`) existed were phrase-graded
   as if they were genuine user/assistant prose -- a skill-file body or
   a `<system-reminder>` echo containing bait phrases like "no
   shortcuts" or "the approach" was ranked sw=100/is_correction=1 by
   bulk.py's signal_weight(). This migration re-runs the harness filter
   over every chunk's content and demotes any positive match: sw=0,
   is_correction=0, is_decision=0. `module` is left untouched -- the
   provenance of *what* the chunk is about is orthogonal to whether it
   should rank as a correction/decision, and Rule #2 (never delete
   experiment/memory data) extends here as "never delete rows, only
   re-grade in place."

2. Module retag: chunks ingested with module=NULL (predates
   collect_module_names() coverage, or the content simply didn't
   contain a recognized flag at ingest time) are re-run through
   bulk.detect_module() using the CURRENT flag roster. Doc chunks that
   still come up empty fall back to their file_path: a chunk sourced
   from docs/marathon/module_states/<flag>.state.yaml or any doc path
   segment that names a known module family is retagged from the path
   even when the prose doesn't repeat the flag token verbatim.

Mandatory DB backup precedes ANY write: the migration refuses to touch
the database if the backup copy fails (see `migrate_regrade`). All
changes are plain UPDATEs inside one transaction; no row is ever
deleted.
"""
from __future__ import annotations

import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

import sqlite_vec

from claude_mem.bulk import collect_module_names, detect_module
from claude_mem.filters import is_harness_content


def _backup(db_path: Path, backup_suffix: str) -> Path:
    """Copy db_path alongside itself as `<name>.bak-<suffix>`. Raises
    RuntimeError (not the underlying OSError) so callers get one
    consistent failure type to catch/report, and so the migration can
    unambiguously refuse to proceed.

    REFUSES to overwrite an existing backup: a re-used suffix silently
    destroyed the Task-2 pre-migration snapshot on 2026-07-02 (the CLI's
    old fixed default aimed every later run at the same file). A backup
    that clobbers a backup is the never-delete violation this guard
    makes structurally impossible -- pass a fresh suffix instead."""
    backup_path = db_path.parent / f"{db_path.name}.bak-{backup_suffix}"
    if backup_path.exists():
        raise RuntimeError(
            f"migrate_regrade: backup target {backup_path} already exists; "
            f"refusing to overwrite a prior backup -- pass a different "
            f"--backup-suffix."
        )
    try:
        shutil.copy2(db_path, backup_path)
    except OSError as exc:
        raise RuntimeError(
            f"migrate_regrade: backup to {backup_path} failed ({exc}); "
            f"refusing to run the migration against an unbacked-up DB."
        ) from exc
    if not backup_path.is_file():
        raise RuntimeError(
            f"migrate_regrade: backup copy reported success but "
            f"{backup_path} does not exist; refusing to run."
        )
    return backup_path


# Doc-path fallback: path segments (lowercased) under which a doc chunk's
# file_path names its module family even when detect_module() finds no
# flag token in the prose itself. module_states/<flag>.state.yaml is the
# canonical per-flag state file (see bulk.collect_module_names); the
# filename stem (minus .state.yaml) IS the flag name.
_MODULE_STATE_DIR_MARKER = "module_states"
_MODULE_STATE_SUFFIX = ".state.yaml"

# Other doc-path families worth a fallback pass: a path segment that is
# itself a substrate/system name recognized as a module, e.g. a chunk
# from docs/marathon/apollo_hypothesis_loop.md or
# docs/superpowers/specs/... whose stem/segment matches a known module.
_FALLBACK_PATH_ROOTS = ("docs/marathon", "docs/superpowers")


def _doc_path_module(file_path: Optional[str], known_modules: List[str]) -> Optional[str]:
    """Return a module name derived from `file_path` alone, or None.

    Two rules, in order:
      1. module_states/<flag>.state.yaml -> <flag> (exact, no roster
         lookup needed -- the filename stem IS the canonical flag name
         per bulk.collect_module_names()).
      2. Any path segment (split on / and \\, case-insensitive) that
         exactly matches a name in `known_modules` -- covers docs that
         live under docs/marathon or docs/superpowers and are named
         after the module/system they document.
    """
    if not file_path:
        return None
    norm = file_path.replace("\\", "/")
    if _MODULE_STATE_DIR_MARKER in norm and norm.endswith(_MODULE_STATE_SUFFIX):
        stem = Path(norm).name[: -len(_MODULE_STATE_SUFFIX)]
        if stem:
            return stem
    if not any(root in norm.lower() for root in _FALLBACK_PATH_ROOTS):
        return None
    segments = {seg.lower() for seg in norm.split("/")}
    # also match the filename stem without extension (e.g. "apollo_hypothesis_loop")
    name = Path(norm).stem.lower()
    segments.add(name)
    for m in known_modules:
        if m.lower() in segments:
            return m
    return None


def migrate_regrade(
    db_path: Path,
    *,
    backup_suffix: str,
    project_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run the corpus re-grade + module-tag migration against `db_path`.

    Backs up `db_path` first (refuses to proceed on backup failure).
    Then, in ONE transaction: demotes harness-content chunks, retags
    NULL-module chunks, and clears is_correction on non-user chunks
    (a correction is the human correcting the assistant; the old
    content-only grader flagged assistant prose that merely echoed a
    correction phrase -- see bulk.is_correction). signal_weight is NOT
    touched by the corrections pass: it is a separate axis, and an
    assistant chunk can be legitimately high-signal without being a
    correction. UPDATE-only -- no DELETE, ever.

    Returns a summary dict:
      {demoted, retagged_module, demoted_assistant_corrections,
       unchanged, backed_up_to, demoted_by_source: {source: count}}

    Idempotent: a second run against the migrated DB reports
    demoted=0, retagged_module=0, demoted_assistant_corrections=0
    (every row that could be improved already has been).
    """
    if not db_path.is_file():
        raise RuntimeError(f"migrate_regrade: no database at {db_path}")

    backup_path = _backup(db_path, backup_suffix)

    known_modules: List[str] = []
    if project_root is not None:
        known_modules = collect_module_names(project_root)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)

    demoted = 0
    retagged_module = 0
    demoted_assistant_corrections = 0
    demoted_by_source: Dict[str, int] = {}
    touched_ids: set = set()

    try:
        # --- 3. Assistant-corrections demotion (set-based: the rule is
        # purely relational, no content inspection needed). ids are
        # collected first so the `unchanged` accounting below stays
        # exact. ----------------------------------------------------------
        ac_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM chunks WHERE is_correction = 1 "
                "AND (role IS NULL OR role != 'user')"
            )
        ]
        if ac_ids:
            conn.executemany(
                "UPDATE chunks SET is_correction = 0 WHERE id = ?",
                [(i,) for i in ac_ids],
            )
        demoted_assistant_corrections = len(ac_ids)
        touched_ids.update(ac_ids)

        rows = conn.execute(
            "SELECT id, content, source, module, file_path, "
            "signal_weight, is_correction, is_decision FROM chunks"
        ).fetchall()

        for row in rows:
            # --- 1. Harness-content demotion --------------------------
            already_demoted = (
                row["signal_weight"] == 0
                and row["is_correction"] == 0
                and row["is_decision"] == 0
            )
            if not already_demoted and is_harness_content(row["content"]):
                conn.execute(
                    "UPDATE chunks SET signal_weight = 0, is_correction = 0, "
                    "is_decision = 0 WHERE id = ?",
                    (row["id"],),
                )
                demoted += 1
                demoted_by_source[row["source"]] = (
                    demoted_by_source.get(row["source"], 0) + 1
                )
                touched_ids.add(row["id"])

            # --- 2. Module retag (NULL-module chunks only) -------------
            if row["module"] is None:
                new_module = None
                if known_modules:
                    new_module = detect_module(row["content"], known_modules)
                    if new_module is None:
                        new_module = _doc_path_module(
                            row["file_path"], known_modules,
                        )
                if new_module is not None:
                    conn.execute(
                        "UPDATE chunks SET module = ? WHERE id = ?",
                        (new_module, row["id"]),
                    )
                    retagged_module += 1
                    touched_ids.add(row["id"])

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    unchanged = len(rows) - len(touched_ids)

    return {
        "demoted": demoted,
        "retagged_module": retagged_module,
        "demoted_assistant_corrections": demoted_assistant_corrections,
        "unchanged": unchanged,
        "backed_up_to": str(backup_path),
        "demoted_by_source": demoted_by_source,
    }
