"""PostToolUse hook: work-aware injection.

After a write/action tool finishes, search the index against BOTH what
Claude touched (tool_input) and what the result revealed (tool_response),
and surface a single high-relevance, novel-to-this-session prior-work
match as a system reminder. Stage A of the librarian design: synchronous,
no LLM reasoning, advisory (always exit 0; never blocks the turn).

Reuses the PreToolUse nudge's gates/renderers (tool_use.py) for DRY.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from claude_mem.config import ProjectConfig
from claude_mem.embed import EmbeddingClient
from claude_mem.intent import has_existing_subsystem_intent
from claude_mem.search import Searcher
from claude_mem.hooks.tool_use import (
    MAX_QUERY_TEXT_CHARS,
    MIN_NUDGE_SCORE,
    NUDGE_TOOLS,
    _format_matches,
    _is_self_referential,
)

POST_NUDGE_TEMPLATE = """NUDGE (claude-mem) -- you just used {tool}; existing work may overlap with what you are doing:

{subsystems}

Soft signal, not a block. If this prior work is relevant and you did not already have it in mind, consider it before continuing. Surface to the user if there is a meaningful conflict you cannot resolve yourself."""


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_stringify(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_stringify(v) for v in value)
    return ""


def _extract_text_post(
    tool_name: str, tool_input: Dict[str, Any], tool_response: Any
) -> str:
    """Pull actionable text from the finished tool's input AND its result."""
    if not isinstance(tool_input, dict):
        tool_input = {}
    parts: list[str] = []
    if tool_name == "Edit":
        parts += [tool_input.get("file_path", ""),
                  tool_input.get("old_string", ""),
                  tool_input.get("new_string", "")]
    elif tool_name == "Write":
        parts += [tool_input.get("file_path", ""), tool_input.get("content", "")]
    elif tool_name == "Bash":
        parts += [tool_input.get("description", ""), tool_input.get("command", "")]
    elif tool_name == "Agent":
        parts += [tool_input.get("subagent_type", ""),
                  tool_input.get("description", ""),
                  tool_input.get("prompt", "")]
    elif tool_name == "Grep":
        parts += [tool_input.get("pattern", ""), tool_input.get("path", "")]
    elif tool_name == "NotebookEdit":
        parts += [tool_input.get("notebook_path", ""),
                  tool_input.get("new_source", "")]
    # The result of any tool can carry relevant identifiers (paths, symbols).
    parts.append(_stringify(tool_response))
    # Bounded per the Stage-A contract — see MAX_QUERY_TEXT_CHARS in
    # tool_use.py for the 2026-07-06 unbounded-payload incident this caps.
    joined = "\n".join(p for p in parts if isinstance(p, str) and p)
    return joined[:MAX_QUERY_TEXT_CHARS]


def _build_searcher(cfg: ProjectConfig) -> Searcher:
    embedder = EmbeddingClient(
        model=cfg.values["embedding_model"],
        fallback_model=cfg.values["embedding_fallback"],
        endpoint=cfg.values["ollama_endpoint"],
        keep_alive=cfg.values["embedding_keep_alive"],
    )
    return Searcher(db_path=cfg.db_path, embedder=embedder)


def run(
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_response: Any,
    session_id: str,
    project_root: Path,
) -> str:
    """Return a single novel, relevant nudge for a finished tool call, or ''."""
    if tool_name not in NUDGE_TOOLS:
        return ""
    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""
    text = _extract_text_post(tool_name, tool_input, tool_response)
    if not text:
        return ""
    if _is_self_referential(text):
        return ""
    if not has_existing_subsystem_intent(text):
        return ""
    searcher = _build_searcher(cfg)
    try:
        rows = searcher.search(text, top_k=5, filter_operator_vetted=True)
    finally:
        searcher.close()
    rows = [
        r for r in rows
        if float(r.get("final_score", r.get("fusion_score", 0.0)))
        >= MIN_NUDGE_SCORE
    ]
    if not rows:
        return ""
    from claude_mem.capture import CaptureStore
    store = CaptureStore(cfg.db_path)
    try:
        for r in rows:
            chunk_id = str(r.get("id") or r.get("chunk_id") or "")
            if not chunk_id or store.was_delivered(session_id, chunk_id):
                continue
            store.record_delivered(session_id, chunk_id, tool_name)
            return POST_NUDGE_TEMPLATE.format(
                tool=tool_name, subsystems=_format_matches([r]),
            )
    finally:
        store.close()
    return ""
