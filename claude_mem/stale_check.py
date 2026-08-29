"""Stale-claim verification gate.

This session's biggest failure was acting on a Claude-authored
checkpoint claim that turned out wrong. The retrieval system should
surface a "verify against current code before acting" reminder
whenever the retrieved chunks include narrative sources (checkpoints,
docs/marathon, MEMORY.md) that name specific symbols.

Per memory rule `feedback_verify_audit_findings_against_code.md`:
audit docs decay; BROKEN / never fires / zero-fill findings need
verification (grep symbol + run tests + check git log) BEFORE being
actionable.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional


# Sources we consider Claude-authored narrative (susceptible to staleness).
_RISKY_SOURCES = ("memory", "checkpoint", "claude_code")

# Path-like markers in chunk metadata that indicate a narrative file
# (vs. e.g. a regular project doc which has source="doc").
_RISKY_PATH_FRAGMENTS = (
    "docs/marathon",
    "memory/checkpoint",
    "MEMORY.md",
)

# Symbol extraction patterns. Each pattern's group(1) is the symbol.
_SYMBOL_PATTERNS = [
    re.compile(r"\b((?:src|scripts|tests|tools)/[a-zA-Z0-9_/\.\-]+)\b"),
    re.compile(r"\b(use_[a-z_][a-z0-9_]*)\b"),
    re.compile(r"\b([a-z_][a-z0-9_]*(?:_[a-z0-9_]+){2,})\b"),  # 3+ snake parts
]


WARNING_TEMPLATE = """STALE-CLAIM VERIFICATION REMINDER

The following retrieved context is from a Claude-authored narrative
source (checkpoint / marathon doc / MEMORY.md). These sources can
decay; symbols named below may have been renamed, removed, or never
merged. Before acting on any claim from these sources:

  - If the source names a file path: verify the file exists
  - If the source names a function or flag: grep for it
  - If the user is about to act on a recommendation derived from these
    sources: verify first

{citations}

Per memory rule `feedback_verify_audit_findings_against_code.md`:
audit docs decay; BROKEN / never fires / zero-fill findings need
verification (grep symbol + run tests + check git log) BEFORE being
actionable."""


def extract_symbol_references(content: str) -> List[str]:
    """Return distinct symbol-like strings from `content`. Patterns:
    file paths under src/scripts/tests/tools; use_* flags; 3+-part
    snake_case identifiers. Preserves first-seen order."""
    seen = set()
    out: List[str] = []
    for pat in _SYMBOL_PATTERNS:
        for m in pat.finditer(content):
            sym = m.group(1)
            if sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


def _is_narrative_chunk(chunk: dict) -> bool:
    src = chunk.get("source", "")
    file_path = chunk.get("file_path") or ""
    if src in _RISKY_SOURCES:
        # claude_code sessions are narrative; doc/git/experiment_summary
        # are not.
        return True
    for frag in _RISKY_PATH_FRAGMENTS:
        if frag in file_path:
            return True
    return False


def _days_since(iso_ts: Optional[str]) -> Optional[int]:
    if not iso_ts:
        return None
    try:
        when = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    now = datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return (now - when).days


def check_for_stale_claims(retrieved: List[dict]) -> Optional[str]:
    """Return a warning string when any narrative chunk references
    a specific symbol. Returns None otherwise."""
    citations: List[str] = []
    for chunk in retrieved:
        if not _is_narrative_chunk(chunk):
            continue
        symbols = extract_symbol_references(chunk.get("content", ""))
        if not symbols:
            continue
        age = _days_since(chunk.get("date") or chunk.get("ingested_at"))
        age_str = f"{age}d" if age is not None else "?d"
        src = chunk.get("file_path") or chunk.get("source")
        head_syms = ", ".join(symbols[:5])
        more = f" (+{len(symbols) - 5} more)" if len(symbols) > 5 else ""
        citations.append(
            f"  - [{age_str} old] {src}: symbols {head_syms}{more}"
        )
    if not citations:
        return None
    return WARNING_TEMPLATE.format(citations="\n".join(citations))
