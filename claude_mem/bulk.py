"""Source parsers + signal weighting for canonical backfill.

Canonical scope (Mike-locked 2026-05-24): every parser here is called
by the `claude-mem bulk` command. The aggregate index is the project's
single source of truth.

Signal weights (cannibalized from agentmemory + Mike's evidence
inversion -- user corrections are highest signal, assistant filler
is lowest):
  user correction          -> 100
  user decision            ->  50
  user investigation       ->  30
  user other               ->  10
  assistant decision       ->  20
  assistant other          ->   0 (filtered out)
  doc / commit / memory    ->  signal_weight=20 unless overridden
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional


CORRECTION_PHRASES = [
    "we already built", "already exists", "doesn't that exist",
    "you already wrote", "we have that", "that's already in",
    "no, look at", "check the existing", "see the existing",
    "we ruled out", "we abandoned", "didn't work last time",
    "we tried that", "you missed", "you didn't read",
    "no shortcuts", "stop doing", "don't do that",
    "wrong, the actual", "actually it's", "you're wrong about",
]

DECISION_PHRASES = [
    "the design is", "we decided", "this is why", "the fix is",
    "root cause", "the issue was", "going forward", "the approach",
    "already exists", "we already built", "was built", "is production",
    "is dormant", "was abandoned", "do not rebuild", "the finding",
    "the result", "the conclusion", "the architecture", "the spec",
]

INVESTIGATION_PHRASES = [
    "let's investigate", "trace why", "diagnose", "let's look at",
    "what's going on with", "deeper trace", "investigate this",
]


# Machine-origin compaction summaries are injected as role="user" but are a
# pointer to prior work, not new signal. They quote past corrections/decisions
# verbatim, so the phrase graders below would mis-rank them above the real thing.
_COMPACTION_MARKERS = (
    "this session is being continued from a previous conversation",
    "the summary below covers the earlier portion of the conversation",
)


def signal_weight(content: str, *, role: str) -> int:
    """Return signal weight 0-100 for a message based on role + content."""
    c = content.lower()
    if any(m in c for m in _COMPACTION_MARKERS):
        return 0
    if role == "user":
        if any(p in c for p in CORRECTION_PHRASES):
            return 100
        if any(p in c for p in DECISION_PHRASES):
            return 50
        if any(p in c for p in INVESTIGATION_PHRASES):
            return 30
        return 10
    if role == "assistant":
        if any(p in c for p in DECISION_PHRASES):
            return 20
        return 0
    return 20  # doc / commit / memory / other -- non-zero by default


# Canonical Tier-2 reference docs whose authority exceeds Tier-3 narrative
# memory (signal_weight 40) per the CLAUDE.md ground-truth hierarchy: the
# auto-derived substrate-capability inventory is the single source of truth
# for which subsystems exist and are dormant, so its chunks must outrank
# design-era narrative when a query's vocabulary aligns with both.
_CANONICAL_DOC_NAMES = ("substrate_capability_inventory.md",)
_CANONICAL_DOC_WEIGHT = 50
_DEFAULT_DOC_WEIGHT = 20


def doc_signal_weight(file_path: str) -> int:
    """Signal weight for a doc chunk. Canonical Tier-2 reference docs rank
    above Tier-3 memory (40); every other doc defaults to 20."""
    name = Path(file_path).name
    return _CANONICAL_DOC_WEIGHT if name in _CANONICAL_DOC_NAMES else _DEFAULT_DOC_WEIGHT


def is_correction(content: str, *, role: str) -> bool:
    """A correction is the HUMAN correcting the assistant: only
    role="user" content can carry the flag. Assistant prose echoing a
    correction phrase (acknowledgements like "you're right, we already
    built X", essays ABOUT corrections) is not itself a correction --
    the pre-fix content-only grader marked those, and the session-start
    RECENT CORRECTIONS render surfaced assistant essays instead of the
    operator's words. role is keyword-required so no call site can
    forget which side of the conversation it is grading."""
    if role != "user":
        return False
    c = content.lower()
    return any(p in c for p in CORRECTION_PHRASES)


def is_decision(content: str) -> bool:
    c = content.lower()
    return any(p in c for p in DECISION_PHRASES)


# The operator's CURATED corrections live as memory files named
# feedback_*.md (hard rules distilled from being corrected) and
# invariant_*.md (first-class invariants). These are the highest-signal
# correction record the project has, but the phrase-matching scanner
# above has ~zero recall on them (replay over live transcripts:
# 0 genuine hits), so until 2026-08-19 the RECENT CORRECTIONS pool held
# only 9 frozen conversation chunks. The basename rule below tags every
# chunk ingested from such a file is_correction=1 at BOTH memory
# ingestion paths (cli.bulk step 3 + incremental._scan_memory);
# maintenance.retag_corrections applies the same rule to already-
# ingested chunks. Case-insensitive: Windows filesystems are.
_CURATED_CORRECTION_PREFIXES = ("feedback_", "invariant_")


def is_curated_correction_file(file_path: str) -> bool:
    """True when ``file_path``'s basename marks an operator-curated
    correction memory file (feedback_*.md / invariant_*.md). Chunks
    parsed from such files carry is_correction=1 regardless of content
    phrasing -- the file NAME is the operator's own curation signal."""
    return Path(file_path).name.lower().startswith(
        _CURATED_CORRECTION_PREFIXES
    )


# Doc-content phrases that mark a chunk as do_not_rebuild without
# requiring a formal MODULE.state PRODUCTION/DORMANT status. Covers
# projects that document architecture in prose only.
DO_NOT_REBUILD_PHRASES = [
    "do not rebuild", "do not re-build", "already built",
    "is production", "is dormant", "was built",
    "dormant", "built but never enabled", "never enabled in production",
]


def is_do_not_rebuild(content: str) -> bool:
    c = content.lower()
    return any(p in c for p in DO_NOT_REBUILD_PHRASES)


def parse_claude_code_jsonl(path: Path) -> Iterator[dict]:
    """Yield messages from a Claude Code session JSONL file."""
    with path.open(encoding="utf-8") as f:
        for line in f:
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
                # Anthropic content blocks; flatten text-only
                content = "".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            if not isinstance(content, str) or not content.strip():
                continue
            yield {
                "role": role,
                "content": content,
                "timestamp": rec.get("timestamp"),
                "session_id": path.stem,
            }


_MAX_SECTION_CHARS = 6000


def _split_sections(content: str) -> List[tuple]:
    """Split markdown on ## / ### headings. Return list of (start_line,
    end_line, text) tuples, 1-indexed inclusive.  Long sections are
    windowed to _MAX_SECTION_CHARS so no single chunk exceeds retrieval
    budget.  start/end refer to the original line positions in content,
    tracked precisely per window (not collapsed to section bounds)."""
    lines = content.splitlines()
    bounds = [
        i for i, ln in enumerate(lines)
        if ln.lstrip().startswith(("## ", "### "))
    ]
    starts = [0] + bounds
    sections = []
    for idx, s in enumerate(starts):
        e = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        if e <= s:
            continue
        joined = "\n".join(lines[s:e])
        text = joined.strip()
        if not text:
            continue
        # leading whitespace stripped from `joined` shifts the first line of
        # `text`; count the newlines in the stripped prefix to find base_line.
        base_line = s + 1 + joined[: len(joined) - len(joined.lstrip())].count("\n")
        for off in range(0, len(text), _MAX_SECTION_CHARS):
            window = text[off : off + _MAX_SECTION_CHARS]
            w_start = base_line + text[:off].count("\n")
            w_end = w_start + window.count("\n")
            sections.append((w_start, w_end, window))
    return sections


def parse_markdown_doc(path: Path) -> List[dict]:
    """Read a markdown doc and split it into heading-level passage chunks.

    Returns a list of dicts — one per section (or windowed sub-section for
    very long sections).  Each dict has: content, file_path, file_mtime,
    line_start, line_end.
    """
    content = path.read_text(encoding="utf-8", errors="replace")
    mtime = path.stat().st_mtime
    out = []
    for start, end, text in _split_sections(content):
        out.append({
            "content": text,
            "file_path": str(path),
            "file_mtime": mtime,
            "line_start": start,
            "line_end": end,
        })
    return out


def parse_memory_md(path: Path) -> List[dict]:
    """Parse an individual memory file. Frontmatter holds type / name /
    description; body is split into heading-level passage chunks (small
    files yield one chunk).  errors='ignore' removed: encoding errors
    surface as UnicodeDecodeError so bad files are caught explicitly."""
    content = path.read_text(encoding="utf-8", errors="replace")
    fm_match = re.match(r"^---\n(.*?)\n---\n(.*)$", content, re.DOTALL)
    if fm_match:
        front_raw, body = fm_match.group(1), fm_match.group(2)
        meta = {}
        for line in front_raw.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                meta[k.strip()] = v.strip()
    else:
        meta, body = {}, content
    mtime = path.stat().st_mtime
    sections = _split_sections(body) or [(1, max(len(body.splitlines()), 1), body.strip())]
    return [
        {
            "content": text,
            "file_path": str(path),
            "file_mtime": mtime,
            "line_start": start,
            "line_end": end,
            "metadata": meta,
        }
        for start, end, text in sections
        if text
    ]


def parse_progress_jsonl(path: Path) -> List[dict]:
    """Parse an experiment progress.jsonl (one JSON object per checkpoint).
    Returns a single chunk containing a compact summary of all checkpoints."""
    rows: List[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not rows:
        return []
    last = rows[-1]
    cov = last.get("coverage_classes", "?")
    if isinstance(cov, list):
        cov = len(cov)
    summary = (
        f"Experiment {path.parent.name}: "
        f"final_tick={last.get('tick', '?')} "
        f"n_completed_tasks={last.get('n_completed_tasks', '?')} "
        f"coverage_total={last.get('coverage_total', '?')} "
        f"coverage_classes={cov} "
        f"ddx_intents_total={last.get('ddx_intents_total', '?')}"
    )
    return [{
        "content": summary,
        "file_path": str(path),
        "file_mtime": path.stat().st_mtime,
    }]


def parse_git_log(repo_root: Path, since: Optional[str] = None) -> List[dict]:
    """Run git log; emit one chunk per commit."""
    cmd = ["git", "-C", str(repo_root), "log",
           "--pretty=format:%H%x09%ai%x09%s%x09%b%x1e"]
    if since:
        cmd.extend(["--since", since])
    try:
        out = subprocess.check_output(cmd, text=True, encoding="utf-8")
    except subprocess.CalledProcessError:
        return []
    chunks = []
    for rec in out.split("\x1e"):
        rec = rec.strip()
        if not rec:
            continue
        parts = rec.split("\t", 3)
        if len(parts) < 3:
            continue
        sha, ts, subject = parts[0], parts[1], parts[2]
        body = parts[3] if len(parts) > 3 else ""
        chunks.append({
            "content": f"{subject}\n\n{body}".strip(),
            "file_path": f"git:{sha[:12]}",
            "date": ts,
        })
    return chunks


def detect_module(content: str, modules: List[str]) -> Optional[str]:
    """Return the module name from `modules` whose identifier-bounded token
    appears EARLIEST in `content` (by character position), or None. Earliest-
    position (not list/alphabetical order) so a chunk spanning a cluster
    heading plus a later unrelated flag is tagged by the flag it leads with,
    not by whichever sorts first alphabetically. Boundary: start/end of string
    OR a non-[a-z0-9_] char on the prefix side / non-[a-z0-9] on the suffix
    side, so `use_apollo` matches but `apollomania` does not."""
    c_lower = content.lower()
    best_module = None
    best_pos = None
    for m in modules:
        pat = rf"(?:^|[^a-z0-9])({re.escape(m.lower())})(?:$|[^a-z0-9])"
        match = re.search(pat, c_lower)
        if match is None:
            continue
        pos = match.start(1)
        if best_pos is None or pos < best_pos:
            best_pos = pos
            best_module = m
    return best_module


def collect_module_names(project_root: Path) -> List[str]:
    """Return substrate/flag names from the per-flag state files at
    docs/marathon/module_states/*.state.yaml. The name is the flag
    (e.g. use_apollo), derived from auto_derived.config_flag when present
    else the filename stem (minus the trailing .state)."""
    names = set()
    state_dir = project_root / "docs" / "marathon" / "module_states"
    for state_file in state_dir.glob("*.state.yaml"):
        name = None
        try:
            for line in state_file.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s.startswith("config_flag:"):
                    name = s.split(":", 1)[1].strip()
                    break
        except OSError:
            name = None
        if not name:
            name = state_file.name[: -len(".state.yaml")]
        if name:
            names.add(name)
    return sorted(names)


def collect_do_not_rebuild_modules(project_root: Path) -> set:
    """Flag names whose state file marks do_not_rebuild: true.

    Reads docs/marathon/module_states/*.state.yaml; for each file,
    extracts auto_derived.config_flag (or falls back to the stem) and
    the top-level do_not_rebuild field.  Returns the set of names where
    do_not_rebuild is true."""
    out = set()
    state_dir = project_root / "docs" / "marathon" / "module_states"
    for state_file in state_dir.glob("*.state.yaml"):
        try:
            txt = state_file.read_text(encoding="utf-8")
        except OSError:
            continue
        name = None
        dnr = False
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("config_flag:"):
                name = s.split(":", 1)[1].strip()
            if s.startswith("do_not_rebuild:"):
                dnr = s.split(":", 1)[1].strip().lower() == "true"
        if not name:
            name = state_file.name[: -len(".state.yaml")]
        if name and dnr:
            out.add(name)
    return out
