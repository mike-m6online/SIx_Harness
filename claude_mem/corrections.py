"""Correction-event detection + auto-boost.

When a user message in a Claude Code session contains a CORRECTION_PHRASE
(see claude_mem.bulk), the immediately-preceding assistant message is
the target of the correction. extract_corrections walks the session in
order and emits one CorrectionEvent per match.

apply_corrections does two things per event:
  1. Indexes the user_correction text itself at signal_weight=100 with
     is_correction=True (source="correction_event"). This makes the
     correction itself retrievable.
  2. Boosts signal_weight of every chunk that matches the event's topic
     by +20 (capped at 100). This raises the existing-subsystem chunks
     so the DO NOT REBUILD wrapper surfaces them sooner next time.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import sqlite_vec

from claude_mem.acronyms import derive_aliases
from claude_mem.bulk import (
    CORRECTION_PHRASES, parse_claude_code_jsonl,
)
from claude_mem.ingest import Chunk, Ingester


@dataclass
class CorrectionEvent:
    session_id: str
    tick: int
    topic: str
    user_correction: str
    assistant_message: str
    timestamp: Optional[str]


# Symbol patterns used to derive the correction topic from the
# combined user + assistant text. Order matters: more-specific first.
_SYMBOL_PATTERNS = [
    re.compile(r"\b([a-z_][a-z0-9_]*_(?:kernel|loop|gate|engine|module|"
               r"flag|head|hook|client|writer|reader|store))\b"),
    re.compile(r"\b(use_[a-z_][a-z0-9_]*)\b"),
    re.compile(r"\b([a-z_]+_[a-z_]+_[a-z_]+)\b"),  # any 3-part snake_case
]

# Stopwords filtered from the fallback word extraction. Includes correction
# phrase fragments + generic English filler that should never be a topic.
_TOPIC_STOPWORDS = {
    "already", "exists", "built", "build", "design", "implement", "create",
    "loop", "gate", "module", "kernel", "engine", "let's", "we've",
    "lets", "have", "that", "this", "with", "from", "into", "what",
    "when", "where", "which", "while", "would", "could", "should",
    "about", "after", "again", "their", "there", "these", "those",
    "thing", "things", "doing", "going", "looking", "saying",
    "check", "look", "see", "stop", "wrong", "right", "okay",
    "actually", "really", "just",
}


def extract_topic(text: str) -> str:
    """Return the most-distinctive symbol-like token in `text`. Tries
    symbol patterns first (snake_case, use_* flag, *_kernel/_loop/etc.);
    falls back to the first 4+ letter non-stopword."""
    lower = text.lower()
    for pat in _SYMBOL_PATTERNS:
        m = pat.search(lower)
        if m:
            return m.group(1)
    # Fallback: first 4+ letter word that is not a stopword.
    for w in re.findall(r"\b[a-z]{4,}\b", lower):
        if w not in _TOPIC_STOPWORDS:
            return w
    return ""


def scan_corrections(
    session_jsonl: Path, *, start_offset: int = 0,
) -> tuple[List[CorrectionEvent], int]:
    """Streaming forward pass from start_offset -- the byte-offset
    watermark pattern of extract_decisions.scan_candidates. The live
    session JSONL is multi-GB, so extract_corrections' full-file
    list() parse is not viable inside a SessionEnd hook.

    Emits one CorrectionEvent per (assistant msg, user msg containing a
    CORRECTION_PHRASE) adjacent-message pair, mirroring
    extract_corrections. Harness-injected pseudo-user content (skill
    bodies, hook stdout, system reminders carry role="user" in the
    JSONL) is excluded -- its correction-phrase hits are not the human
    correcting anything.

    Returns (events, new_offset). new_offset is normally the byte
    position after the last record read; when the window ENDS on an
    assistant message, that record's START offset is returned instead,
    so a correction appended in the next window still finds its pairing
    context. Re-reading an assistant record can never duplicate an
    event: events fire only on user records.

    CorrectionEvent.tick carries the user record's start byte offset
    (monotonic within the file); the full-file walk used the message
    index, which a resumable scan cannot know.
    """
    from claude_mem.extract_decisions import _iter_messages
    from claude_mem.filters import is_harness_content

    events: List[CorrectionEvent] = []
    prev_role: Optional[str] = None
    prev_text: str = ""
    record_start = start_offset
    new_offset = start_offset
    last_assistant_start: Optional[int] = None
    for role, text, ts, end_offset in _iter_messages(
        session_jsonl, start_offset=start_offset,
    ):
        record_start = new_offset
        new_offset = end_offset
        if text is None:
            continue
        if (
            role == "user"
            and prev_role == "assistant"
            and not is_harness_content(text)
            and any(p in text.lower() for p in CORRECTION_PHRASES)
        ):
            events.append(CorrectionEvent(
                session_id=session_jsonl.stem,
                tick=record_start,
                topic=extract_topic(text + " " + prev_text),
                user_correction=text,
                assistant_message=prev_text,
                timestamp=ts or None,
            ))
        prev_role, prev_text = role, text
        last_assistant_start = record_start if role == "assistant" else None
    if last_assistant_start is not None:
        return events, last_assistant_start
    return events, new_offset


def extract_corrections(session_jsonl: Path) -> List[CorrectionEvent]:
    """Walk a session JSONL; emit one CorrectionEvent per (assistant
    msg, user msg containing CORRECTION_PHRASE) pair."""
    messages = list(parse_claude_code_jsonl(session_jsonl))
    events: List[CorrectionEvent] = []
    for i, msg in enumerate(messages):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not any(p in content.lower() for p in CORRECTION_PHRASES):
            continue
        if i == 0:
            continue
        prev = messages[i - 1]
        if prev.get("role") != "assistant":
            continue
        combined = content + " " + prev.get("content", "")
        topic = extract_topic(combined)
        events.append(CorrectionEvent(
            session_id=msg.get("session_id", session_jsonl.stem),
            tick=i,
            topic=topic,
            user_correction=content,
            assistant_message=prev.get("content", ""),
            timestamp=msg.get("timestamp"),
        ))
    return events


def apply_corrections(
    events: List[CorrectionEvent],
    db_path: Path,
    *,
    embedder,
    boost_amount: int = 20,
) -> int:
    """Apply each event: index the correction at sw=100; boost
    matching chunks by `boost_amount`. Returns the number of events
    applied."""
    if not events:
        return 0
    ing = Ingester(db_path=db_path, embedder=embedder)
    n_applied = 0
    try:
        for ev in events:
            ing.add(Chunk(
                content=ev.user_correction,
                source="correction_event",
                role="user",
                session_id=ev.session_id,
                date=ev.timestamp,
                signal_weight=100,
                is_correction=True,
                module=ev.topic if ev.topic else None,
                aliases=derive_aliases(ev.user_correction),
            ))
            _boost_matching_chunks(db_path, ev.topic, boost_amount)
            n_applied += 1
    finally:
        ing.close()
    return n_applied


def _boost_matching_chunks(
    db_path: Path, topic: str, boost_amount: int,
) -> None:
    """Raise signal_weight by `boost_amount` (capped at 100) for every
    chunk whose content contains `topic`. No-op when topic is empty."""
    if not topic:
        return
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        conn.execute(
            """
            UPDATE chunks
            SET signal_weight = MIN(100, signal_weight + ?)
            WHERE source != 'correction_event'
              AND LOWER(content) LIKE '%' || LOWER(?) || '%'
            """,
            (boost_amount, topic),
        )
        conn.commit()
    finally:
        conn.close()
