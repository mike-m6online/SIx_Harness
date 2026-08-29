"""Session-end candidate extractor (anti-recurrence Rung 2, HYBRID capture).

Mines a Claude Code session JSONL for *candidate* decisions and dead-ends,
returned as pending rows. The agent formalizes them later; this layer only
proposes, so recall is weighted over precision. Cue-phrase matching over
sentence-split message text; deduped by title and capped per type so a long
session cannot flood the tables.

Precision deny-list (2026-08-19, from a 119-row manual triage that
rejected 66 candidates): junk was dominated by (a) bare status-echo
fragments ("Locked in.", "All four locked in.") that carry a cue phrase
but zero recoverable content, and (b) MEMORY.md-maintenance narration
("Compacting the ...", "Trimming ...") whose "superseded" wording
matches the dead-end cues. scan_candidates now drops candidates whose
cleaned title is shorter than _MIN_TITLE_CHARS or full-matches
_STATUS_ECHO_RE, and skips records that OPEN with a maintenance marker.
Deliberately a conservative deny-list, not a classifier -- recall stays
weighted over precision for everything else. Skips are counted per
category and surfaced in the capture-extract summary output.

Stored-title quality (same date): titles are clipped on a whole-word
boundary at _TITLE_CLIP_CHARS via textutil.clip (the old bare [:120]
slice stored mid-word fragments PERMANENTLY), and the rationale /
why_shelved field now stores the full whitespace-collapsed sentence
(word-boundary capped at _RATIONALE_CAP_CHARS) instead of a copy of the
clipped title. Rows stored before this change are NOT migrated -- their
titles were clipped at write time and the source text offset has moved
on; they age out through triage.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from .capture import Decision, DeadEnd
from .filters import is_harness_content
from .textutil import SENTENCE_SPLIT, clip, collapse_ws

DECISION_CUES = [
    "we decided", "decision:", "locked in", "we lock in", "mike-approved",
    "mike approved", "let's go with", "we'll go with", "going with the",
    "we're going with", "route =", "we choose", "we chose", "approved:",
    "we will go with", "we are going with",
]
DEAD_END_CUES = [
    "rejected", "shelved", "deferred", "superseded", "not pursue",
    "won't pursue", "abandoned", "wrong paradigm", "dead end", "dead-end",
    "ruled out", "scrapped", "backed out", "we are explicitly not",
]

_MAX_PER_TYPE = 25
_SENT_SPLIT = SENTENCE_SPLIT

# Whole-word clip budget for stored titles/approaches (was a bare [:120]
# mid-word slice) and the cap for the full-sentence rationale field.
_TITLE_CLIP_CHARS = 200
_RATIONALE_CAP_CHARS = 600

# A candidate whose cleaned title is shorter than this is a bare
# fragment: too little content to triage or to steer a future session
# ("Locked in.", "Approved: go."). Calibrated against the 2026-08-19
# manual triage where every genuine candidate comfortably exceeded it.
_MIN_TITLE_CHARS = 40

# Bare-acknowledgment status echoes: the operator (or the assistant
# echoing them) confirming state, not recording a decision. Matched
# case-insensitively against the WHOLE cleaned title (anchored), with
# optional leading/trailing filler bounded so genuine sentences that
# merely CONTAIN "locked in" are never caught.
_STATUS_ECHO_RE = re.compile(
    r"^\s*(?:"
    r"locked in"                              # "Locked in."
    r"|all .{0,20} locked in"                 # "All four locked in."
    r"|everything.{0,20}locked in"            # "Everything is locked in."
    r"|(?:it's|its|that's|thats|this is|we're|were) .{0,20}locked in"
    r"|all .{0,20}(?:approved|confirmed|good|set|done|green)"
    r"|(?:approved|confirmed|agreed|done|understood|perfect|"
    r"sounds good|all set|ship it|got it)"    # bare acknowledgments
    r")\s*[.!]*\s*$",
    re.IGNORECASE,
)

# Records that OPEN with MEMORY.md-maintenance narration are ledger
# housekeeping, not decisions -- their "superseded"/"trimming" wording
# phrase-matches the dead-end cues. Prefix match on the lstripped record
# text (mirrors filters.py's prefix-marker discipline: only content the
# marker literally OPENS is caught, never prose that mentions it later).
_MAINTENANCE_PREFIXES = ("MEMORY.md", "Compacting the", "Trimming")


@dataclass
class ScanSkips:
    """Per-category counts of candidates/records dropped by the
    precision deny-list, aggregated across one scan_candidates call.
    Addable so multi-file callers (run_candidates) can merge them."""

    short_title: int = 0
    status_echo: int = 0
    maintenance_records: int = 0

    @property
    def total(self) -> int:
        return self.short_title + self.status_echo + self.maintenance_records

    def __add__(self, other: "ScanSkips") -> "ScanSkips":
        return ScanSkips(
            short_title=self.short_title + other.short_title,
            status_echo=self.status_echo + other.status_echo,
            maintenance_records=(
                self.maintenance_records + other.maintenance_records
            ),
        )

    def summary(self) -> str:
        """One-line human summary for the capture-extract hook echo."""
        return (
            f"{self.total} low-signal candidate(s) skipped "
            f"(short-title={self.short_title}, "
            f"status-echo={self.status_echo}, "
            f"memory-maintenance-records={self.maintenance_records})"
        )


def _is_status_echo(title: str) -> bool:
    """True when the whole cleaned title is a bare status acknowledgment
    (see _STATUS_ECHO_RE)."""
    return _STATUS_ECHO_RE.match(title) is not None


def _is_maintenance_record(text: str) -> bool:
    """True when the record text OPENS with a MEMORY.md-maintenance
    marker (prefix match after stripping leading whitespace)."""
    return text.lstrip().startswith(_MAINTENANCE_PREFIXES)


def _parse_record(rec: dict) -> Tuple[str, Optional[str], str]:
    msg = rec.get("message") if isinstance(rec.get("message"), dict) else rec
    role = msg.get("role") or rec.get("role") or ""
    ts = rec.get("timestamp") or ""
    content = msg.get("content")
    if isinstance(content, str):
        return role, content, ts
    if isinstance(content, list):
        parts = [
            blk["text"]
            for blk in content
            if isinstance(blk, dict) and isinstance(blk.get("text"), str)
        ]
        if parts:
            return role, "\n".join(parts), ts
    return role, None, ts


def _iter_messages(
    jsonl_path: Path, *, start_offset: int = 0
) -> Iterator[Tuple[str, Optional[str], str, int]]:
    """Yield (role, text, timestamp, end_offset) for each record from
    start_offset onward. Binary read so byte offsets are exact across
    platforms; end_offset is the byte position immediately after the record's
    line, i.e. the resumable watermark. text is None for records without
    usable text content."""
    with open(jsonl_path, "rb") as fh:
        fh.seek(start_offset)
        while True:
            raw = fh.readline()
            if not raw:
                break
            end_offset = fh.tell()
            line = raw.decode("utf-8", errors="replace").strip()
            role: str = ""
            text: Optional[str] = None
            ts: str = ""
            if line:
                try:
                    rec = json.loads(line)
                    role, text, ts = _parse_record(rec)
                except json.JSONDecodeError:
                    pass
            yield role, text, ts, end_offset


def _sentences_with_cue(text: str, cues: List[str]) -> List[str]:
    """Whitespace-collapsed FULL sentences containing any cue phrase.
    Clipping happens at the candidate-build site (title vs rationale get
    different budgets), never here -- the old [:120] slice at this layer
    is what stored mid-word fragments permanently."""
    hits = []
    for sent in _SENT_SPLIT.split(text):
        low = sent.lower()
        if any(cue in low for cue in cues):
            hits.append(collapse_ws(sent))
    return hits


def _passes_precision_gate(title: str, skips: ScanSkips) -> bool:
    """Apply the conservative deny-list to one cleaned candidate title,
    counting the drop reason on ``skips``. Status-echo is checked first:
    it is the more specific signal, so a short bare echo counts as an
    echo, not merely as short."""
    if _is_status_echo(title):
        skips.status_echo += 1
        return False
    if len(title) < _MIN_TITLE_CHARS:
        skips.short_title += 1
        return False
    return True


def scan_candidates(
    jsonl_path: Path, *, start_offset: int = 0
) -> Tuple[List[Decision], List[DeadEnd], int, ScanSkips]:
    """Single forward pass from start_offset. Returns
    (decision_candidates, dead_end_candidates, new_offset, skips), where
    new_offset is the byte position after the last record read -- the
    resumable watermark -- and skips counts the candidates/records the
    precision deny-list dropped (surfaced in the capture-extract summary).
    Each candidate's date derives from the record's own timestamp (stable
    across re-runs). Per-type additions are capped at _MAX_PER_TYPE per call;
    because the watermark makes each run read only newly-appended content, the
    cap bounds a single run's new content, not the whole file. Deny-listed
    candidates never consume cap slots."""
    decisions: List[Decision] = []
    dead_ends: List[DeadEnd] = []
    skips = ScanSkips()
    dseen, eseen = set(), set()
    offset = start_offset
    for _role, text, ts, end_offset in _iter_messages(
        jsonl_path, start_offset=start_offset
    ):
        offset = end_offset
        if text is None:
            continue
        # Harness-injected pseudo-user content (skill-file bodies, hook
        # stdout, system reminders, compaction summaries) is appended to
        # the session JSONL with role="user" but did not come from the
        # human; its cue-phrase hits (a skill file mentioning "we
        # decided" or "rejected") are not real decisions/dead-ends.
        if is_harness_content(text):
            continue
        # MEMORY.md-maintenance narration ("Compacting the ...",
        # "Trimming ...") phrase-matches the 'superseded' dead-end cue
        # but is ledger housekeeping, not a decision or dead-end.
        if _is_maintenance_record(text):
            skips.maintenance_records += 1
            continue
        cand_date = ts[:10] if ts else None
        if len(decisions) < _MAX_PER_TYPE:
            for sent in _sentences_with_cue(text, DECISION_CUES):
                title = clip(sent, _TITLE_CLIP_CHARS)
                if not _passes_precision_gate(title, skips):
                    continue
                if title in dseen:
                    continue
                dseen.add(title)
                decisions.append(Decision(
                    title=title,
                    rationale=clip(sent, _RATIONALE_CAP_CHARS),
                    state="pending", date=cand_date,
                ))
                if len(decisions) >= _MAX_PER_TYPE:
                    break
        if len(dead_ends) < _MAX_PER_TYPE:
            for sent in _sentences_with_cue(text, DEAD_END_CUES):
                approach = clip(sent, _TITLE_CLIP_CHARS)
                if not _passes_precision_gate(approach, skips):
                    continue
                if approach in eseen:
                    continue
                eseen.add(approach)
                dead_ends.append(DeadEnd(
                    approach=approach,
                    why_shelved=clip(sent, _RATIONALE_CAP_CHARS),
                    state="pending",
                    date=cand_date,
                ))
                if len(dead_ends) >= _MAX_PER_TYPE:
                    break
    return decisions, dead_ends, offset, skips


def extract_decision_candidates(
    jsonl_path: Path, *, date: str = ""
) -> List[Decision]:
    decisions, _de, _off, _skips = scan_candidates(jsonl_path)
    if date:
        decisions = [
            Decision(title=d.title, rationale=d.rationale, state="pending", date=date)
            for d in decisions
        ]
    return decisions


def extract_dead_end_candidates(
    jsonl_path: Path, *, date: str = ""
) -> List[DeadEnd]:
    _dec, dead_ends, _off, _skips = scan_candidates(jsonl_path)
    if date:
        dead_ends = [
            DeadEnd(approach=e.approach, why_shelved=e.why_shelved,
                    state="pending", date=date)
            for e in dead_ends
        ]
    return dead_ends
