"""PreToolUse hook for write-class tools.

Watches Claude's OWN outputs (tool inputs) for build/investigation
intent against existing-subsystem matches in the index. Emits a NUDGE
when a meaningful match is found, so Claude sees relevant prior systems
before acting.

Mike-locked design (2026-05-25):
  - NUDGE, NOT STOP. Always exit 0. Hook stdout becomes a system
    reminder Claude sees on the next turn; the tool call is never blocked.
  - Watches the ASSISTANT's tool inputs, not user prompts. Most of the
    meaningful build/investigation text comes from the assistant
    (subagent dispatches, Edit/Write content, Bash commands). User
    prompts in execution mode are short ("approved", "status"); they
    miss the actual building-intent signal.
  - Soft framing ("hey look at this") rather than alarming
    ("DO NOT REBUILD"). Claude evaluates, decides whether to continue,
    ignore, or surface the conflict to the user.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from claude_mem.config import ProjectConfig
from claude_mem.embed import EmbeddingClient
from claude_mem.intent import has_existing_subsystem_intent
from claude_mem.search import Searcher
from claude_mem.textutil import clip


# Tools that warrant nudge-on-action. Read / Grep / Glob excluded
# (lookup-only, no rebuild risk). TodoWrite excluded (tracking, not
# action). Tool names match Claude Code's canonical tool identifiers.
NUDGE_TOOLS = {"Edit", "Write", "Bash", "Agent", "NotebookEdit"}

# Upper bound on the query text a nudge hook hands to search. The nudge
# needs the identifiers (paths, symbols, subsystem names) in a payload's
# HEAD; whole payloads (a multi-MB Write content or Bash output) add no
# ranking signal, blow past the embedder's context window (forcing its
# head-truncation retry loop against a 2s read timeout), and — before
# search.py's _BM25_MAX_TERMS bound — exploded the FTS5 MATCH tree
# (2026-07-06 incident: 12 GB hung hook holding index.db's lock). The
# hooks are Stage-A librarian by design: synchronous, cheap, advisory —
# bounded input is part of that contract. 16 KiB comfortably covers the
# identifier-bearing head of any tool payload.
MAX_QUERY_TEXT_CHARS = 16384

# Minimum relevance score for a match to be worth nudging. Empirically
# calibrated against the origin project's 2793-chunk index: noise queries
# (unrelated topics) score 0.003-0.008; keyword-overlap matches start around
# 0.015-0.030; topical matches with do_not_rebuild=True land 0.030+.
# We set the threshold below the keyword-overlap band so the nudge fires
# on plausible (not just perfect) matches; false positives are tolerable
# because the nudge is advisory and Claude evaluates before acting.
# Future work: pull from cfg.values so per-project tuning is possible.
MIN_NUDGE_SCORE = 0.01

NUDGE_TEMPLATE = """NUDGE (claude-mem) -- existing systems may overlap with what you are about to {action}:

{subsystems}

Soft signal, not a block: if you knew these exist and your action is intentional, continue. If you did not know, consider whether the existing implementation already covers your case before building parallel infrastructure. Surface to the user if there is a meaningful conflict you cannot resolve yourself."""


def run(
    tool_name: str,
    tool_input: Dict[str, Any],
    project_root: Path,
) -> str:
    """Return a NUDGE string when the assistant's tool input matches an
    existing-subsystem entry in the index; empty string otherwise."""
    if tool_name not in NUDGE_TOOLS:
        return ""
    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""
    text = _extract_text(tool_name, tool_input)
    if not text:
        return ""
    if _is_self_referential(text):
        return ""
    if not has_existing_subsystem_intent(text):
        return ""
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
    )
    searcher = Searcher(db_path=cfg.db_path, embedder=embedder)
    try:
        # filter_operator_vetted (decisions / corrections / DNR /
        # signal_weight>=50) is the broad gate; the origin project's corpus has
        # very few explicit do_not_rebuild=1 chunks but 956 decisions,
        # 146 corrections, 1215 signal_weight>=50. The strict DNR-only
        # filter would silence the hook against real corpora.
        subs = searcher.search(
            text, top_k=5, filter_operator_vetted=True,
        )
    finally:
        searcher.close()
    subs = [
        r for r in subs
        if float(r.get("final_score", r.get("fusion_score", 0.0)))
        >= MIN_NUDGE_SCORE
    ]
    if not subs:
        return ""
    return NUDGE_TEMPLATE.format(
        action=_action_verb(tool_name),
        subsystems=_format_matches(subs[:3]),
    )


def _extract_text(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Pull the actionable text from each tool's input payload."""
    if not isinstance(tool_input, dict):
        return ""
    if tool_name == "Edit":
        parts = [
            tool_input.get("file_path", ""),
            tool_input.get("old_string", ""),
            tool_input.get("new_string", ""),
        ]
    elif tool_name == "Write":
        parts = [
            tool_input.get("file_path", ""),
            tool_input.get("content", ""),
        ]
    elif tool_name == "Bash":
        parts = [
            tool_input.get("description", ""),
            tool_input.get("command", ""),
        ]
    elif tool_name == "Agent":
        parts = [
            tool_input.get("subagent_type", ""),
            tool_input.get("description", ""),
            tool_input.get("prompt", ""),
        ]
    elif tool_name == "NotebookEdit":
        parts = [
            tool_input.get("notebook_path", ""),
            tool_input.get("new_source", ""),
        ]
    else:
        return ""
    joined = "\n".join(p for p in parts if isinstance(p, str) and p)
    return joined[:MAX_QUERY_TEXT_CHARS]


def _is_self_referential(text: str) -> bool:
    """Loop-back guard: skip nudges when the tool input is operating on
    claude-mem itself (otherwise we recurse and fire on every claude-mem
    config / docs edit)."""
    lower = text.lower()
    return (
        "claude-mem" in lower
        or "claude_mem" in lower
        or "tools/claude_mem" in lower
    )


def _action_verb(tool_name: str) -> str:
    return {
        "Edit": "edit",
        "Write": "write",
        "Bash": "run via shell",
        "Agent": "dispatch a subagent for",
        "NotebookEdit": "edit a notebook cell for",
    }.get(tool_name, tool_name.lower())


def _format_matches(rows: List[dict]) -> str:
    out = []
    for r in rows:
        head = clip(r["content"], 160)
        loc = r.get("file_path") or r.get("source") or "-"
        score = float(r.get("final_score", r.get("fusion_score", 0.0)))
        module = r.get("module") or "-"
        out.append(f"  - [{module}] {head}  (score={score:.3f}, {loc})")
    return "\n".join(out)
