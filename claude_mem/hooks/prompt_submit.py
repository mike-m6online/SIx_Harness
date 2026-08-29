"""UserPromptSubmit hook implementation.

Detects build/investigation/decision intent and emits up to three block
types, each behind a precision gate (Task B, 2026-08-19):

  - DO-NOT-REBUILD warning: only lists retrieved items that pass the
    distinctive-token per-item relevance filter (relevance.dnr_item_gate,
    B3); if no item passes, NOTHING is emitted -- the old
    ``if subs or corrs:`` gate fired on any hit because search.py has no
    relevance floor.
  - STALE-CLAIM reminder: only fires when the DO-NOT-REBUILD block fired,
    and cites only the chunks that passed the B3 filter (B4) -- it used
    to cite ANY narrative chunk in an unfiltered top-10 (~79% fire rate
    on arbitrary intent prompts).
  - DECISION LINEAGE (Rung 3): thread injection is gated by word-boundary
    matched-token IDF (relevance.lineage_gate, B1) instead of the old
    substring-overlap>=2 heuristic that let generic words inject.
    Cache-lookup-only; never calls Ollama inline; fully guarded.

Per-session damping (B2, reusing the delivered_nudges mechanism the
tool-use nudge path already writes): a given thread injects at most
twice per session; the DNR and stale-claim blocks at most 3 times per
session each. Requires ``session_id``; ``None`` disables damping
(backward-compatible).

Every invocation past the index-exists guard writes one
``wrapper_invocations`` telemetry row (B5) -- the shakedown's
measurement instrument: intent flags, which blocks were emitted,
injected thread ids, matched-token/IDF evidence, and damping verdicts.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from claude_mem.capture import CaptureStore
from claude_mem.config import ProjectConfig
from claude_mem.embed import EmbeddingClient
from claude_mem.intent import (
    has_build_intent, has_decision_intent, has_investigation_intent,
)
from claude_mem.relevance import (
    CorpusIdf, dnr_item_gate, lineage_gate, prompt_relevance_tokens,
)
from claude_mem.search import Searcher
from claude_mem.stale_check import check_for_stale_claims
from claude_mem.synthesize import structured_fallback
from claude_mem.telemetry import record_wrapper_invocation
from claude_mem.textutil import clip


WARNING_TEMPLATE = """WARNING: DO NOT REBUILD -- existing-subsystem detection fired

Build-intent or investigation-intent in this prompt. The following
subsystems / past findings already exist in this project. Do NOT propose,
spec, design, rebuild, or re-investigate any of them unless explicitly
told the existing implementation is being replaced:

{subsystems}

User corrections relevant to this topic (highest signal):
{corrections}

If you are about to propose something in these areas: STOP. First check
whether the existing implementation covers the need. If you believe a
new implementation or investigation is genuinely required, state
explicitly why the existing one cannot be extended."""


# Per-session damping caps (Task B2). Keys live in delivered_nudges'
# chunk_id column, namespaced so they cannot collide with the tool-use
# path's sha256-hex chunk ids.
THREAD_INJECTION_CAP = 2
DNR_INJECTION_CAP = 3
STALE_INJECTION_CAP = 3

_DNR_ITEM_KEY = "dnr"
_STALE_ITEM_KEY = "stale"
_DELIVERY_TOOL = "prompt_submit"


def _thread_key(thread_id: str) -> str:
    """delivered_nudges item key for a lineage thread injection."""
    return f"thread:{thread_id}"


def _thread_haystack(store: "CaptureStore", thread: dict) -> str:
    # Match on the decision-identifying text (name + decision titles + dead-end
    # approaches), NOT the verbose summary -- a single common summary word like
    # "test" must not trigger a thread injection.
    parts = [thread.get("name", "")]
    for d in store.list_decisions(thread_id=thread["id"]):
        parts.append(d.get("title") or "")
    for e in store.list_dead_ends(thread_id=thread["id"]):
        parts.append(e.get("approach") or "")
    return " ".join(parts).lower()


def _item_text(row: Dict[str, Any]) -> str:
    """Lowercased matching surface for the DNR per-item filter: the
    chunk's content plus its identity metadata (module, aliases, path)."""
    parts = [
        row.get("content") or "",
        row.get("module") or "",
        row.get("aliases") or "",
        row.get("file_path") or "",
    ]
    return " ".join(p for p in parts if p).lower()


def _thread_lineage_block(
    prompt: str,
    db_path: Path,
    idf: Optional[CorpusIdf] = None,
    session_id: Optional[str] = None,
) -> Tuple[str, List[str], Dict[str, Dict[str, float]], List[str]]:
    """Cache-lookup-only lineage block for the most relevant thread(s).

    Gate (B1): relevance.lineage_gate -- word-boundary matched-token IDF,
    not substring overlap. Damping (B2): at most THREAD_INJECTION_CAP
    injections per (session, thread); disabled when ``session_id`` is
    None. When ``idf`` is None a CorpusIdf is opened (and closed) here;
    run() passes its own in to share the per-prompt cache.

    Returns ``(block_text, injected_thread_ids, matched_token_summary,
    suppressed_item_keys)``. Never calls Ollama; never intentionally
    raises (callers still wrap defensively).
    """
    store = None
    own_idf = idf is None
    try:
        store = CaptureStore(db_path)
        if own_idf:
            idf = CorpusIdf(db_path)
        toks = prompt_relevance_tokens(prompt)
        if not toks:
            return ("", [], {}, [])
        hits = []
        for thread in store.list_threads():
            hay = _thread_haystack(store, thread)
            result = lineage_gate(toks, hay, idf)
            if result.passed:
                hits.append((result, thread))
        if not hits:
            return ("", [], {}, [])
        # IDF regime ranks by matched-token idf sum; the small-corpus
        # fallback (all sums 0.0) falls through to match count -- the
        # ordering the old overlap sort produced.
        hits.sort(
            key=lambda x: (x[0].idf_sum, len(x[0].matched)), reverse=True,
        )
        blocks: List[str] = []
        injected: List[str] = []
        summary: Dict[str, Dict[str, float]] = {}
        suppressed: List[str] = []
        for result, thread in hits[:2]:
            key = _thread_key(thread["id"])
            if (
                session_id is not None
                and store.delivery_count(session_id, key)
                >= THREAD_INJECTION_CAP
            ):
                suppressed.append(key)
                continue
            cached, _key = store.get_cached_lineage(thread["id"])
            if cached:
                body = cached
            else:
                decs = store.list_decisions(thread_id=thread["id"])
                des = store.list_dead_ends(thread_id=thread["id"])
                body = structured_fallback(thread, decs, des)
            blocks.append(
                "DECISION LINEAGE (claude-mem thread) -- prior reasoning on "
                "this topic; do not re-decide a settled question without "
                "reading it:\n" + body
            )
            injected.append(thread["id"])
            summary[thread["id"]] = {
                t: round(v, 3) for t, v in result.matched.items()
            }
            if session_id is not None:
                store.record_delivery(session_id, key, _DELIVERY_TOOL)
        return ("\n\n".join(blocks), injected, summary, suppressed)
    finally:
        if own_idf and idf is not None:
            idf.close()
        if store is not None:
            store.close()


# System-generated turn shapes (2026-08-29, day-9 shakedown review).
# Claude Code delivers agent-completion callbacks, local-command echoes,
# and reminder blocks through the same UserPromptSubmit hook as human
# prompts. Nine days of wrapper_invocations telemetry showed 2 of the 3
# real-use injections landing on task-notification turns -- topically
# matched, but spent tokens with no reader decision to inform. A turn
# whose first non-whitespace characters are one of these markers gets no
# retrieval work at all; its telemetry row still lands (the instrument
# sees every invocation) with the exemption named.
_SYSTEM_TURN_OPENERS: Tuple[str, ...] = (
    "<task-notification>",
    "<local-command-stdout>",
    "<local-command-caveat>",
    "<command-name>",
    "<system-reminder>",
)


def _system_turn_marker(prompt: str) -> Optional[str]:
    """The matched opener when `prompt` is a system-generated turn, else
    None. Matches only at the head of the text: a human prompt that
    merely QUOTES one of these tags mid-sentence still gets the full
    pipeline."""
    head = prompt.lstrip()
    for opener in _SYSTEM_TURN_OPENERS:
        if head.startswith(opener):
            return opener
    return None


def run(
    prompt: str, project_root: Path, session_id: Optional[str] = None,
) -> str:
    """Run the UserPromptSubmit hook body; returns the injection text
    (possibly '') for the caller to echo.

    ``session_id`` enables the per-session damping caps (B2) and stamps
    the telemetry row (B5); ``None`` (the pre-B2 call shape) disables
    damping and records a NULL session.
    """
    # Sanitize C0 control characters that break SQLite FTS5 (a null byte
    # raises "unterminated string"); ordinary whitespace is retained.
    prompt = "".join(
        " " if ord(c) < 0x20 and c not in "\t\n\r" else c for c in prompt
    )
    cfg = ProjectConfig(project_root=project_root)
    if not cfg.db_path.is_file():
        return ""

    sys_marker = _system_turn_marker(prompt)
    if sys_marker is not None:
        try:
            record_wrapper_invocation(
                cfg.telemetry_path,
                prompt_hash=hashlib.sha256(
                    prompt.encode("utf-8")).hexdigest(),
                prompt_truncated=prompt,
                session_id=session_id,
                matched_token_summary={"system_turn_exempt": sys_marker},
            )
        except Exception:
            # The measurement instrument must never break the hook.
            pass
        return ""

    build_fired = has_build_intent(prompt)
    invest_fired = has_investigation_intent(prompt)
    decision_fired = has_decision_intent(prompt)

    # Telemetry accumulator (B5). One row per invocation, written in the
    # finally below so a mid-body exception still leaves its trace.
    tele: Dict[str, Any] = {
        "build_intent_fired": build_fired,
        "investigation_intent_fired": invest_fired,
        "decision_intent_fired": decision_fired,
        "do_not_rebuild_warning_emitted": False,
        "stale_claim_warning_emitted": False,
        "lineage_block_emitted": False,
        "lineage_thread_ids": [],
        "matched_token_summary": {},
        "suppressed_by_damping": [],
        "retrieved_chunk_count": 0,
        "retrieved_chunk_topics": [],
        "retrieval_latency_ms": 0,
    }

    parts: List[str] = []
    try:
        if build_fired or invest_fired or decision_fired:
            _emit_blocks(
                prompt, cfg, session_id, parts, tele,
                build_fired=build_fired, decision_fired=decision_fired,
            )
    finally:
        try:
            record_wrapper_invocation(
                cfg.telemetry_path,
                prompt_hash=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                prompt_truncated=prompt,
                session_id=session_id,
                **tele,
            )
        except Exception:
            # The measurement instrument must never break the hook.
            pass

    return "\n\n".join(parts) if parts else ""


def _emit_blocks(
    prompt: str,
    cfg: ProjectConfig,
    session_id: Optional[str],
    parts: List[str],
    tele: Dict[str, Any],
    *,
    build_fired: bool,
    decision_fired: bool,
) -> None:
    """Build the DO-NOT-REBUILD (+stale) and lineage blocks into
    ``parts``, updating the ``tele`` accumulator in place."""
    tokens = prompt_relevance_tokens(prompt)
    idf = CorpusIdf(cfg.db_path)
    store = CaptureStore(cfg.db_path)
    try:
        embedder = EmbeddingClient(
            model=cfg.values["embedding_model"],
            fallback_model=cfg.values["embedding_fallback"],
            endpoint=cfg.values["ollama_endpoint"],
            keep_alive=cfg.values["embedding_keep_alive"],
        )
        searcher = Searcher(db_path=cfg.db_path, embedder=embedder)
        t0 = time.monotonic()
        try:
            # filter_operator_vetted (DNR OR is_correction OR is_decision
            # OR signal_weight>=50) -- broader than the prior
            # filter_do_not_rebuild=True gate. A real 2,793-chunk corpus
            # has 0 explicit do_not_rebuild=1 chunks but 956 decisions /
            # 146 corrections / 1215 signal_weight>=50; the strict DNR-only
            # filter silenced this hook against any real bulk-backfilled
            # corpus. The broader gate surfaces operator-vetted signal
            # which is the actual signal Mike's "DO NOT REBUILD" intent
            # was reaching for; the B3 per-item relevance filter below
            # supplies the topical precision the search itself lacks.
            subs = searcher.search(
                prompt, top_k=10, filter_operator_vetted=True,
            )
            corrs = searcher.search(
                prompt, top_k=5, filter_is_correction=True,
            )
        finally:
            searcher.close()
        tele["retrieval_latency_ms"] = int((time.monotonic() - t0) * 1000)
        tele["retrieved_chunk_count"] = len(subs) + len(corrs)

        # B3b: per-item distinctive-token relevance filter. Only items
        # word-boundary-matching >= 2 distinctive prompt tokens are
        # listed; if no item passes, the block is not emitted at all.
        dnr_matched: Dict[str, float] = {}

        def _passing(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            kept = []
            for row in rows:
                result = dnr_item_gate(tokens, _item_text(row), idf)
                if result.passed:
                    kept.append(row)
                    dnr_matched.update(result.matched)
            return kept

        filtered_subs = _passing(subs)
        filtered_corrs = _passing(corrs)
        topics: List[str] = []
        for row in filtered_subs + filtered_corrs:
            module = row.get("module")
            if module and module not in topics:
                topics.append(module)
        tele["retrieved_chunk_topics"] = topics
        if dnr_matched:
            tele["matched_token_summary"][_DNR_ITEM_KEY] = {
                t: round(v, 3) for t, v in dnr_matched.items()
            }

        if filtered_subs or filtered_corrs:
            if (
                session_id is not None
                and store.delivery_count(session_id, _DNR_ITEM_KEY)
                >= DNR_INJECTION_CAP
            ):
                tele["suppressed_by_damping"].append(_DNR_ITEM_KEY)
            else:
                block = WARNING_TEMPLATE.format(
                    subsystems=_format(filtered_subs)
                    or "  (none on this topic)",
                    corrections=_format(filtered_corrs)
                    or "  (none on this topic)",
                )
                tele["do_not_rebuild_warning_emitted"] = True
                # B4 (supersedes the Phase 4.6 unfiltered-top-10 wiring):
                # the stale-claim reminder rides ONLY on a fired DNR block
                # and cites ONLY the chunks that passed the B3 filter --
                # stale_check keeps its symbol-extraction role on that
                # reduced set.
                stale = check_for_stale_claims(filtered_subs + filtered_corrs)
                if stale:
                    if (
                        session_id is not None
                        and store.delivery_count(session_id, _STALE_ITEM_KEY)
                        >= STALE_INJECTION_CAP
                    ):
                        tele["suppressed_by_damping"].append(_STALE_ITEM_KEY)
                    else:
                        block = block + "\n\n" + stale
                        tele["stale_claim_warning_emitted"] = True
                        if session_id is not None:
                            store.record_delivery(
                                session_id, _STALE_ITEM_KEY, _DELIVERY_TOOL,
                            )
                parts.append(block)
                if session_id is not None:
                    store.record_delivery(
                        session_id, _DNR_ITEM_KEY, _DELIVERY_TOOL,
                    )

        # Rung 3: decision-intent OR build-intent prompts get the relevant
        # thread's lineage. Build-intent is included so "what if we use the
        # FM head..." prompts surface the rejection verdict (the B1 IDF
        # gate protects precision). Lookup-only (no inline Ollama); fully
        # guarded so it cannot break the hook.
        if decision_fired or build_fired:
            try:
                lineage, injected, summary, suppressed = _thread_lineage_block(
                    prompt, cfg.db_path, idf=idf, session_id=session_id,
                )
            except Exception:
                lineage, injected, summary, suppressed = "", [], {}, []
            if lineage:
                parts.append(lineage)
                tele["lineage_block_emitted"] = True
            tele["lineage_thread_ids"] = injected
            if summary:
                tele["matched_token_summary"]["lineage"] = summary
            tele["suppressed_by_damping"].extend(suppressed)
    finally:
        idf.close()
        store.close()


def _format(rows: List[dict]) -> str:
    out = []
    for r in rows:
        head = clip(r["content"], 160)
        loc = r.get("file_path") or r.get("source")
        out.append(f"  - [{r.get('module') or '-'}] {head}  ({loc})")
    return "\n".join(out)
