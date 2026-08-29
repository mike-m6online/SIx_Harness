"""Harness-content filter (spec R1).

claude-mem's corpus was contaminated: harness-injected pseudo-user content
-- skill-file bodies, task-notifications, system reminders, hook stdout,
compaction summaries -- gets appended to a Claude Code session JSONL with
role="user" (it is literally injected into the user turn of the
transcript), so the bulk grader in cli.py and the candidate miner in
extract_decisions.py were phrase-grading it as if it were genuine user
prose. A skill file that happens to contain the literal string "no
shortcuts" or "the approach" then phrase-matches CORRECTION_PHRASES /
DECISION_PHRASES in bulk.py and outranks every real user correction.

This module is a content-ORIGIN filter, not a phrase filter: it identifies
text that originated from the harness (hooks, skill loaders, slash
commands, IDE context, compaction) rather than from the human typing at
the keyboard. It must NOT flag genuine user prose merely because that
prose happens to contain a trigger phrase or the word "hook" -- see the
negative-case fixtures in tests/test_filters.py.

Two marker classes:
  prefix -- the harness marker is the literal opening of injected content
            (skill-file headers, hook-stdout echoes, compaction
            preambles): the harness writes it as the first characters of
            the message, before any prose. Matched with `startswith`
            against the text after stripping incidental leading
            whitespace, so a genuine message that later happens to
            mention e.g. "the SessionStart hook design" mid-sentence is
            not caught -- only content the marker literally OPENS is.
  tag    -- the harness marker is a structural tag (<task-notification>,
            <system-reminder>, ...) that the harness never lets a human
            type organically. Checked against the full text.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

# Ordered (marker, reason, kind) triples. Order matters only for
# `harness_reason`, which reports the first marker that matched --
# earlier entries win ties.
#
# "prefix" markers must be the literal opening of the (whitespace-
# stripped) text -- see the docstring above for why `startswith`, not a
# bounded substring window.
# "tag" markers are matched anywhere in the text (they are structural
# harness tags a human would never organically type).
HARNESS_MARKERS: List[Tuple[str, str, str]] = [
    # (marker, reason, kind)
    ("Base directory for this skill:", "skill_file_body", "prefix"),
    ("<task-notification>", "task_notification", "tag"),
    ("<system-reminder>", "system_reminder", "tag"),
    ("<command-name>", "command_name", "tag"),
    ("<local-command-", "local_command", "tag"),
    ("<ide_selection>", "ide_selection", "tag"),
    ("UserPromptSubmit hook success", "hook_stdout", "prefix"),
    ("UserPromptSubmit hook", "hook_stdout", "prefix"),
    ("SessionStart hook", "hook_stdout", "prefix"),
    ("hook success", "hook_stdout", "prefix"),
    (
        "This session is being continued from a previous conversation",
        "compaction_summary",
        "prefix",
    ),
]


def harness_reason(text: str) -> Optional[str]:
    """Return the reason string for the first harness marker that matches
    `text`, or None if no marker matches. Case-sensitive: harness markers
    are fixed-case structural strings the harness itself emits verbatim,
    and case-folding would widen prefix markers (e.g. "hook success")
    into ordinary lowercase prose that merely mentions hooks."""
    if not text:
        return None
    stripped = text.lstrip()
    for marker, reason, kind in HARNESS_MARKERS:
        if kind == "prefix":
            if stripped.startswith(marker):
                return reason
        else:
            if marker in text:
                return reason
    return None


def is_harness_content(text: str) -> bool:
    """True if `text` originated from the harness (skill loader, hook
    stdout, slash command, IDE context, compaction summary) rather than
    from genuine user or assistant prose."""
    return harness_reason(text) is not None
