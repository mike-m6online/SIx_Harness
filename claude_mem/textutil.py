"""Shared text-shaping utilities for render/capture surfaces.

Root cause this module closes (2026-08-19 render-quality findings):
five call sites independently truncated content with bare slices
(``text[:180]``, ``[:160]``, ``[:140]``, ``[:120]``) that cut MID-WORD
with no ellipsis -- the session-start corrections render, both hook
match formatters, the CLI search echo, and (permanently, into stored
rows) the decision/dead-end candidate miner. One whole-word clip lives
here so every surface truncates the same way, and the sentence-split
regex the candidate miner uses is exported so other extractors (the
capture-backfill rationale) split sentences identically instead of
growing a parallel definition.
"""
from __future__ import annotations

import re

#: Unicode horizontal ellipsis appended to any clipped string.
ELLIPSIS = "…"

#: Sentence boundary: split after ., ! or ? followed by whitespace.
#: Shared with extract_decisions (candidate mining) and the
#: capture-backfill rationale extractor so "first sentence" means the
#: same thing everywhere.
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def collapse_ws(text: str) -> str:
    """Collapse all runs of whitespace (including newlines) to single
    spaces and strip the ends. The canonical pre-step for any one-line
    render of multi-line content."""
    return " ".join(text.split())


def clip(text: str, limit: int) -> str:
    """Whitespace-collapse ``text`` and clip it to at most ``limit``
    characters on a whole-word boundary, appending an ellipsis when
    truncation happened.

    Guarantees:
      - the result is never longer than ``limit`` characters;
      - a string that fits after whitespace collapse is returned
        unmodified (no ellipsis);
      - truncation never cuts mid-word: the clip backs up to the last
        whitespace boundary inside the budget (a single token longer
        than the budget is hard-cut -- there is no word boundary to
        respect);
      - the ellipsis is a single character (U+2026) and counts toward
        ``limit``.
    """
    if limit <= 0:
        return ""
    collapsed = collapse_ws(text)
    if len(collapsed) <= limit:
        return collapsed
    # Reserve one character for the ellipsis.
    head = collapsed[: limit - 1]
    # If the cut landed exactly at a word end (the next source character
    # is a space), the last token in head is already complete -- only
    # back up to the previous space when the cut split a word.
    if collapsed[len(head)] != " ":
        boundary = head.rfind(" ")
        if boundary > 0:
            head = head[:boundary]
    return head.rstrip() + ELLIPSIS


def first_sentence(text: str) -> str:
    """The whitespace-collapsed first sentence of ``text`` (the full
    text when no sentence boundary exists). Callers wanting a bounded
    field apply :func:`clip` on top (e.g. ``clip(first_sentence(t),
    600)`` for capture rationale)."""
    collapsed = collapse_ws(text)
    if not collapsed:
        return ""
    return SENTENCE_SPLIT.split(collapsed, maxsplit=1)[0]
