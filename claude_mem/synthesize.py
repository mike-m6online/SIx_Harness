"""Thread -> dated lineage synthesis (anti-recurrence Rung 3).

build_lineage_prompt assembles a thread's decisions + dead-ends into an
instruction; synthesize_lineage runs it through an injected generator (the
GenerationClient.generate in production); structured_fallback renders the rows
deterministically when generation is unavailable; get_or_regenerate_lineage is
the per-thread cache gate keyed on threads.last_updated. Generation runs off the
prompt hot path only (SessionEnd + capture-synthesize)."""
from __future__ import annotations

import json
from typing import Callable, List


def _rejected(raw) -> str:
    try:
        items = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        items = []
    return "; ".join(items) if items else ""


def build_lineage_prompt(thread: dict, decisions: List[dict],
                         dead_ends: List[dict]) -> str:
    lines = [
        "You are tracing the lineage of an engineering decision thread. "
        "From the dated decisions and dead-ends below, write a SHORT dated "
        "A -> B -> C lineage (3-6 sentences) showing how the thinking evolved "
        "and what was tried and shelved. Be concrete; cite dates. Do not invent "
        "facts beyond what is given.",
        "",
        f"THREAD: {thread.get('name', '')}",
        f"SUMMARY: {thread.get('summary', '')}",
        "",
        "DECISIONS (oldest to newest):",
    ]
    for d in sorted(decisions, key=lambda x: x.get("date") or ""):
        rej = _rejected(d.get("options_rejected"))
        rej_s = f" | rejected: {rej}" if rej else ""
        lines.append(f"  - [{d.get('date')}] {d.get('title')}{rej_s}")
        if d.get("rationale"):
            lines.append(f"      because: {d['rationale']}")
    lines.append("")
    lines.append("DEAD-ENDS:")
    for e in sorted(dead_ends, key=lambda x: x.get("date") or ""):
        sup = e.get("superseded_by")
        sup_s = f" -> superseded by: {sup}" if sup else ""
        lines.append(f"  - [{e.get('date')}] {e.get('approach')}{sup_s}")
        if e.get("why_shelved"):
            lines.append(f"      shelved because: {e['why_shelved']}")
    return "\n".join(lines)


def synthesize_lineage(thread: dict, decisions: List[dict],
                       dead_ends: List[dict],
                       generate: Callable[[str], str]) -> str:
    prompt = build_lineage_prompt(thread, decisions, dead_ends)
    return generate(prompt).strip()


def structured_fallback(thread: dict, decisions: List[dict],
                        dead_ends: List[dict]) -> str:
    lines = [f"thread: {thread.get('name', '')} [{thread.get('state', '')}]"]
    if thread.get("summary"):
        lines.append(f"  {thread['summary']}")
    for d in sorted(decisions, key=lambda x: x.get("date") or "", reverse=True):
        tag = " (Mike-approved)" if d.get("mike_approved") else ""
        lines.append(f"  - DECISION [{d.get('date')}] {d.get('title')}{tag}")
        rej = _rejected(d.get("options_rejected"))
        if rej:
            lines.append(f"      rejected: {rej}")
    for e in sorted(dead_ends, key=lambda x: x.get("date") or "", reverse=True):
        sup = e.get("superseded_by")
        sup_s = f" (superseded_by: {sup})" if sup else ""
        lines.append(f"  - DEAD-END [{e.get('date')}] {e.get('approach')}{sup_s}")
    return "\n".join(lines)


def get_or_regenerate_lineage(store, thread_id: str,
                              generate: Callable[[str], str]) -> str:
    """Cache gate keyed on threads.last_updated. Returns the cached lineage if
    fresh; otherwise regenerates via `generate`, caches, and returns it. Returns
    "" for an unknown thread. Generation exceptions propagate to the caller
    (which decides whether to fall back)."""
    thread = store.get_thread(thread_id)
    if not thread:
        return ""
    text, key = store.get_cached_lineage(thread_id)
    if text and key == thread.get("last_updated"):
        return text
    decisions = store.list_decisions(thread_id=thread_id)
    dead_ends = store.list_dead_ends(thread_id=thread_id)
    new_text = synthesize_lineage(thread, decisions, dead_ends, generate)
    store.set_cached_lineage(thread_id, new_text, thread.get("last_updated"))
    return new_text
