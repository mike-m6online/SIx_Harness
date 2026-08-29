#!/usr/bin/env python3
"""MEMORY.md auto decay / dedup maintenance reporter.

Borrowed (correctly) from the memory-os reference: the live MEMORY.md
index is injected into every session's system prompt, so it has a hard
load budget. Left unattended it accumulates `## SUPERSEDED ...` hook
sections until it silently truncates on injection (this happened on the
source project 2026-06-02: 1290 lines / 222KB, hand-pruned to 69 lines by
archiving the superseded sections to MEMORY_ARCHIVE.md). NOTHING
auto-prevented the recurrence -- that is this tool's job.

Pure-stdlib (no third-party deps) so it can run as a SessionStart hook
with a tight timeout. UTF-8 without BOM throughout, matching the existing
memory files.

Actions:
  - archive-superseded (the only auto-mutation): every MEMORY.md section
    whose header matches `^##+ SUPERSEDED` (case-insensitive) is MOVED out
    of MEMORY.md and PREPENDED into MEMORY_ARCHIVE.md (newest-first, under
    the archive's existing header). Content preserved verbatim. Idempotent:
    a second run finds no SUPERSEDED sections and is a no-op.
  - budget check (report-only): after archiving, if MEMORY.md exceeds
    --max-lines OR --max-bytes, emit a WARNING naming the largest
    non-index sections as consolidation candidates. Never auto-mutates
    non-SUPERSEDED content -- the live "AFTER COMPACTION" section is
    hand-owned.
  - dup-link detection (report-only): any memory-file link `](foo.md)`
    appearing more than once in MEMORY.md.
  - orphan detection (report-only): `*.md` files in the memory dir
    (excluding MEMORY.md, MEMORY_ARCHIVE.md, and `_*.md`) with no link in
    MEMORY.md.

Exit code is always 0 once the flags parse -- this is a maintenance
reporter, not a gate. The WARNING is textual so a SessionStart hook
surfaces it to the operator.

Harness-kit parameterization (the ONLY deltas from the origin-project original):
  - `--project-root` and `--memory-dir` are REQUIRED flags; omitting
    either exits 2 with argparse's standard clear message. The harness
    init tool bakes concrete absolute paths into the installed hook
    command, so runtime discovery is unnecessary and no default exists.
  - `--project-root` is accepted for the uniform harness hook interface
    (every installed hook receives the same flag pair); the decay pass
    itself operates entirely within `--memory-dir`.
"""
from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path
from typing import List, Tuple

MEMORY_NAME = "MEMORY.md"
ARCHIVE_NAME = "MEMORY_ARCHIVE.md"

DEFAULT_MAX_LINES = 280
DEFAULT_MAX_BYTES = 90000

# A section header is a markdown header line (## or deeper). A SUPERSEDED
# section is one whose header text begins with "SUPERSEDED" (case-insensitive),
# which covers both "## SUPERSEDED HOOK ..." and "## SUPERSEDED: prior hook".
_HEADER_RE = re.compile(r"^#{2,}\s+")
_SUPERSEDED_RE = re.compile(r"^#{2,}\s+SUPERSEDED\b", re.IGNORECASE)
# Memory-file links of the form ](something.md) -- captures the target.
_LINK_RE = re.compile(r"\]\(([^)]+\.md)\)")

_ARCHIVE_DEFAULT_HEADER = (
    "# MEMORY ARCHIVE -- superseded checkpoint hooks\n"
    "\n"
    "Moved out of MEMORY.md to keep the live index under the session load budget.\n"
    "These are SUPERSEDED hooks; their detail lives in the referenced "
    "checkpoint_*.md files.\n"
    "Recall-only -- do NOT treat as current state. Newest first.\n"
    "\n"
    "---\n"
)


def _read(path: Path) -> str:
    """Read a file as UTF-8, stripping a leading BOM if present."""
    with io.open(path, "r", encoding="utf-8") as f:
        text = f.read()
    if text.startswith("﻿"):
        text = text[1:]
    return text


def _write(path: Path, text: str) -> None:
    """Write a file as UTF-8 with NO BOM (newline='' so we control line
    endings explicitly and do not let the platform rewrite them)."""
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text)


class Section:
    """A header line plus everything up to (not including) the next header."""

    __slots__ = ("header", "body_lines")

    def __init__(self, header: str, body_lines: List[str]) -> None:
        self.header = header
        self.body_lines = body_lines

    @property
    def is_superseded(self) -> bool:
        return bool(_SUPERSEDED_RE.match(self.header))

    @property
    def is_header(self) -> bool:
        return bool(_HEADER_RE.match(self.header))

    def text(self) -> str:
        return "\n".join([self.header] + self.body_lines)

    def line_count(self) -> int:
        return 1 + len(self.body_lines)


def _split_sections(text: str) -> Tuple[List[str], List[Section]]:
    """Split a markdown document into a preamble (lines before the first
    `##`+ header) and an ordered list of header-delimited sections.

    A section runs from a `^##+ ` header line to the line before the next
    `^##+ ` header (or EOF). The single-`#` title line, if any, stays in
    the preamble.
    """
    lines = text.split("\n")
    preamble: List[str] = []
    sections: List[Section] = []
    current: Section | None = None
    for line in lines:
        if _HEADER_RE.match(line):
            current = Section(line, [])
            sections.append(current)
        elif current is None:
            preamble.append(line)
        else:
            current.body_lines.append(line)
    return preamble, sections


def _reassemble(preamble: List[str], sections: List[Section]) -> str:
    parts: List[str] = []
    if preamble:
        parts.append("\n".join(preamble))
    for sec in sections:
        parts.append(sec.text())
    body = "\n".join(p for p in parts)
    # Normalize trailing whitespace to exactly one trailing newline.
    return body.rstrip("\n") + "\n"


def _prepend_to_archive(archive_path: Path, moved_sections: List[Section]) -> None:
    """Prepend moved sections into MEMORY_ARCHIVE.md, newest-first, after
    the archive's top-of-file header block (the lines through the first
    `---` separator) and before existing archived sections. Creates the
    archive with a default header if it does not exist."""
    moved_block = "\n\n".join(sec.text().rstrip("\n") for sec in moved_sections)
    if not archive_path.is_file():
        new_text = (
            _ARCHIVE_DEFAULT_HEADER
            + "\n"
            + moved_block
            + "\n"
        )
        _write(archive_path, new_text)
        return

    existing = _read(archive_path)
    header_block, remainder = _split_archive_header(existing)
    # header_block already ends with the "---\n" separator line.
    new_text = (
        header_block.rstrip("\n")
        + "\n\n"
        + moved_block
        + ("\n\n" + remainder.lstrip("\n") if remainder.strip() else "\n")
    )
    _write(archive_path, new_text.rstrip("\n") + "\n")


def _split_archive_header(text: str) -> Tuple[str, str]:
    """Split the archive into (header_block_through_first_---, remainder).

    The header block is the `# MEMORY ARCHIVE ...` title plus its intro
    paragraph plus the first standalone `---` separator line. New sections
    are inserted just after that separator. If no `---` separator exists,
    treat the whole prefix up to the first `^##+ ` header as the header
    block.
    """
    lines = text.split("\n")
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            header = "\n".join(lines[: idx + 1])
            remainder = "\n".join(lines[idx + 1 :])
            return header, remainder
    # No separator: header is everything before the first ##+ section.
    for idx, line in enumerate(lines):
        if _HEADER_RE.match(line):
            header = "\n".join(lines[:idx])
            remainder = "\n".join(lines[idx:])
            return header, remainder
    return text, ""


def _find_dup_links(text: str) -> List[Tuple[str, int]]:
    counts: dict[str, int] = {}
    for m in _LINK_RE.finditer(text):
        target = m.group(1)
        counts[target] = counts.get(target, 0) + 1
    return sorted(
        ((t, c) for t, c in counts.items() if c > 1),
        key=lambda tc: (-tc[1], tc[0]),
    )


def _find_orphans(memory_dir: Path, memory_text: str) -> List[str]:
    """List *.md files in the memory dir that have no link in MEMORY.md.
    Excludes MEMORY.md, MEMORY_ARCHIVE.md, and any `_*.md` files."""
    linked = set(_LINK_RE.findall(memory_text))
    # Links may carry relative path prefixes; normalize to basenames for
    # comparison since topic-memory files live flat in the memory dir.
    linked_basenames = {Path(t).name for t in linked}
    orphans: List[str] = []
    for p in sorted(memory_dir.glob("*.md")):
        name = p.name
        if name in (MEMORY_NAME, ARCHIVE_NAME):
            continue
        if name.startswith("_"):
            continue
        if name not in linked_basenames:
            orphans.append(name)
    return orphans


def _largest_non_index_sections(
    sections: List[Section], top_n: int = 5
) -> List[Tuple[str, int]]:
    """Return the (header, line_count) of the largest sections, biggest
    first. Used only for the budget-WARNING consolidation hint -- nothing
    here is mutated."""
    sized = [(sec.header.strip(), sec.line_count()) for sec in sections]
    sized.sort(key=lambda hc: -hc[1])
    return sized[:top_n]


def run(
    memory_dir: Path,
    *,
    apply: bool,
    max_lines: int,
    max_bytes: int,
    orphan_list_limit: int = 20,
) -> str:
    """Execute the decay pass. Returns a human-readable summary.

    When `apply` is True and SUPERSEDED sections exist, MEMORY.md is
    rewritten (sections removed) and MEMORY_ARCHIVE.md is updated. When
    `apply` is False (dry-run), no files are written; the summary reports
    what WOULD move.
    """
    lines_out: List[str] = []
    memory_path = memory_dir / MEMORY_NAME
    archive_path = memory_dir / ARCHIVE_NAME

    lines_out.append("memory_decay -- MEMORY.md maintenance report")
    lines_out.append(f"  memory dir: {memory_dir}")

    if not memory_path.is_file():
        lines_out.append(f"  ERROR: {MEMORY_NAME} not found; nothing to do.")
        return "\n".join(lines_out)

    memory_text = _read(memory_path)
    preamble, sections = _split_sections(memory_text)

    superseded = [s for s in sections if s.is_superseded]
    kept = [s for s in sections if not s.is_superseded]

    # --- archive-superseded (the only auto-mutation) ---
    if superseded:
        lines_out.append(
            f"  SUPERSEDED sections found: {len(superseded)} "
            f"({'archiving' if apply else 'dry-run, NOT archiving'})"
        )
        for s in superseded:
            lines_out.append(
                f"    - {s.header.strip()[:90]}  ({s.line_count()} lines)"
            )
        if apply:
            _prepend_to_archive(archive_path, superseded)
            new_memory_text = _reassemble(preamble, kept)
            _write(memory_path, new_memory_text)
            memory_text = new_memory_text
            sections = kept
            lines_out.append(
                f"    -> moved into {ARCHIVE_NAME} (newest-first) and removed "
                f"from {MEMORY_NAME}"
            )
    else:
        lines_out.append("  SUPERSEDED sections found: 0 (no-op)")

    # --- budget check (report-only) ---
    n_lines = memory_text.count("\n") + (
        0 if memory_text.endswith("\n") or memory_text == "" else 1
    )
    n_bytes = len(memory_text.encode("utf-8"))
    lines_out.append(
        f"  budget: {n_lines} lines (max {max_lines}), "
        f"{n_bytes} bytes (max {max_bytes})"
    )
    over_lines = n_lines > max_lines
    over_bytes = n_bytes > max_bytes
    if over_lines or over_bytes:
        which = []
        if over_lines:
            which.append(f"lines {n_lines}>{max_lines}")
        if over_bytes:
            which.append(f"bytes {n_bytes}>{max_bytes}")
        lines_out.append(
            "  WARNING: MEMORY.md is OVER BUDGET (" + ", ".join(which) + "). "
            "Consolidate the largest non-index sections below "
            "(manual -- non-SUPERSEDED content is hand-owned, never auto-pruned):"
        )
        for header, lc in _largest_non_index_sections(sections):
            lines_out.append(f"    - {lc} lines  {header[:90]}")
    else:
        lines_out.append("  budget: OK (under both limits)")

    # --- dup-link detection (report-only) ---
    dups = _find_dup_links(memory_text)
    if dups:
        lines_out.append("  duplicate index links (appear >1x):")
        for target, count in dups:
            lines_out.append(f"    - {target}  (x{count})")
    else:
        lines_out.append("  duplicate index links: none")

    # --- orphan detection (report-only) ---
    # The memory dir intentionally holds hundreds of topic/checkpoint files
    # while MEMORY.md links only a curated subset, so a large orphan count is
    # normal and not actionable. Cap the per-name listing to keep the hook
    # surface readable; always report the total.
    orphans = _find_orphans(memory_dir, memory_text)
    if orphans:
        lines_out.append(
            f"  indexed-orphan candidates ({len(orphans)} *.md files not "
            f"linked in MEMORY.md):"
        )
        limit = len(orphans) if orphan_list_limit <= 0 else orphan_list_limit
        for name in orphans[:limit]:
            lines_out.append(f"    - {name}")
        if len(orphans) > limit:
            lines_out.append(
                f"    ... and {len(orphans) - limit} more "
                f"(pass --orphan-list-limit 0 to list all)"
            )
    else:
        lines_out.append("  indexed-orphan candidates: none")

    return "\n".join(lines_out)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="memory_decay",
        description="MEMORY.md auto decay / dedup maintenance reporter. "
                    "Both path flags are REQUIRED: the harness init tool "
                    "bakes concrete paths into the installed hook command.",
    )
    p.add_argument(
        "--project-root",
        required=True,
        help="Absolute path to the target project root. REQUIRED. Accepted "
             "for the uniform harness hook interface (every hook receives "
             "the same flag pair); the decay pass itself operates entirely "
             "within --memory-dir.",
    )
    p.add_argument(
        "--memory-dir",
        required=True,
        help="Absolute path to the memory directory holding MEMORY.md and "
             "MEMORY_ARCHIVE.md. REQUIRED.",
    )
    p.add_argument(
        "--max-lines", type=int, default=DEFAULT_MAX_LINES,
        help=f"MEMORY.md line budget (default: {DEFAULT_MAX_LINES}).",
    )
    p.add_argument(
        "--max-bytes", type=int, default=DEFAULT_MAX_BYTES,
        help=f"MEMORY.md byte budget (default: {DEFAULT_MAX_BYTES}).",
    )
    p.add_argument(
        "--orphan-list-limit", type=int, default=20,
        help="Max orphan filenames to list (0 = list all; the memory dir "
             "normally holds hundreds of un-indexed topic files).",
    )
    p.add_argument(
        "--print", dest="print_only", action="store_true",
        help="Human summary to stdout, then perform the safe archive "
             "(SessionStart hook mode).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Report only; write nothing.",
    )
    p.add_argument(
        "--apply", action="store_true",
        help="Perform the safe archive + writes (the default action).",
    )
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    # Default action is to apply the safe archive. --dry-run suppresses
    # writes. --print and --apply both apply (--print just emphasizes the
    # stdout summary for the hook). --dry-run wins if both are passed.
    apply = not args.dry_run
    summary = run(
        Path(args.memory_dir),
        apply=apply,
        max_lines=args.max_lines,
        max_bytes=args.max_bytes,
        orphan_list_limit=args.orphan_list_limit,
    )
    sys.stdout.write(summary + "\n")
    return 0  # always succeed -- maintenance reporter, never a gate


if __name__ == "__main__":
    raise SystemExit(main())
