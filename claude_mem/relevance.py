"""Distinctive-token relevance machinery (Task B1).

Shared by the prompt_submit hook's thread-lineage gate and its
DO-NOT-REBUILD per-item filter. Replaces the old substring-overlap
heuristic (any >=4-char prompt token appearing as a SUBSTRING of a
thread haystack, gate = overlap>=2) which let shopping-list-grade
prompts inject decision lineage: generic words ("build", "test",
"action") matched everything.

Design:
  - Tokens match on WORD BOUNDARIES (regex ``\\b``), not substrings.
  - Each token gets an inverse-document-frequency weight over the
    project's ``chunks`` corpus: ``idf = ln((N + 1) / (df + 1))`` where
    ``df`` is counted with one FTS5 MATCH per token (the same porter-
    stemmed inverted index the BM25 search side uses, so morphological
    variants collapse the way they do at search time).
  - The lineage gate requires >= 2 word-boundary matches AND a matched-
    token IDF sum >= ``LINEAGE_IDF_SUM_THRESHOLD`` AND at least one
    matched token with idf >= ``DISTINCTIVE_IDF``.
  - The DNR per-item filter requires >= 2 DISTINCTIVE (idf >=
    ``DISTINCTIVE_IDF``) prompt tokens word-boundary-matching the item.

Calibration (2026-08-19, origin project's live corpus N=18799, read-only):
  generic tokens measured idf 0.9-3.0 (build 2.05, dialogue 2.11,
  module 2.07, plan 2.01, list 2.98); distinctive tokens 3.3-8.7
  (gw 3.42, rewire 5.08, homeostasis 5.99, metabolism 4.67). The three
  observed off-topic prompt classes (kit packaging / database-paper
  review / installer building) word-boundary-match at most ONE token of
  any live thread haystack (max sum 2.05); the weakest genuinely
  on-topic prompt class (homeostasis wording) matches at sum 9.51 with
  max-token idf 5.99. ``LINEAGE_IDF_SUM_THRESHOLD = 7.0`` sits between
  those bands with margins of +4.95 over the strongest off-topic sum
  and -2.51 under the weakest on-topic sum.

Small-corpus fallback: IDF statistics over a corpus of fewer than
``MIN_IDF_CORPUS`` chunks are noise (a brand-new project can hold
capture threads before its first ingest), so below that floor the gates
degrade to plain word-boundary overlap >= 2 -- already strictly more
precise than the substring heuristic they replace.
"""
from __future__ import annotations

import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from claude_mem.intent import DECISION_INTENT_VERBS


# Tokens whose grammatical role dominates their topical signal.
STOPWORDS = {
    "should", "would", "could", "this", "that", "with", "from", "have",
    "about", "into", "what", "when", "where", "which", "your", "ours",
    "keep", "going", "change", "look", "want", "need", "make", "does",
    "the", "and", "for", "are", "but", "not", "you", "our", "out", "now",
}

# Project-specific short tokens that carry strong identity signal.
# The >= 4 char filter is intentionally bypassed for these.
ALLOWED_SHORT = {"gw", "mlm", "f8", "pe", "sp", "c2", "c3", "ddx", "slm", "hss", "mlp"}

# IDF floor separating distinctive from generic vocabulary. Measured on
# the live corpus: generic engineering words land at 0.9-3.0, domain
# identifiers at 3.3-8.7 (see module docstring).
DISTINCTIVE_IDF = 3.0

# Matched-token IDF sum a thread haystack must reach for lineage
# injection (see calibration numbers in the module docstring).
LINEAGE_IDF_SUM_THRESHOLD = 7.0

# Minimum word-boundary token matches for ANY injection (both regimes).
MIN_MATCHES = 2

# Minimum DISTINCTIVE prompt-token matches for a DNR item to be listed.
DNR_MIN_DISTINCTIVE_MATCHES = 2

# Below this corpus size IDF is statistically meaningless: fall back to
# word-boundary overlap >= MIN_MATCHES.
MIN_IDF_CORPUS = 50

# Cap on per-token df lookups per prompt (one FTS5 COUNT each). Ordered
# first-seen, so the head of the prompt wins the budget -- mirrors the
# _BM25_MAX_TERMS discipline in search.py. Tokens beyond the cap score
# idf 0.0 (never distinctive, no sum contribution).
MAX_IDF_TOKENS = 32


def significant_tokens(text: str) -> List[str]:
    """Split ``text`` into deduplicated, ordered significant tokens.

    Lowercases, splits on every non-alphanumeric run (so hyphens,
    underscores, slashes, and punctuation are all boundaries), drops
    stopwords, and keeps tokens that are >= 4 chars or in the
    ALLOWED_SHORT project-identifier set. First-seen order preserved.
    """
    out: List[str] = []
    seen: set[str] = set()
    for w in re.split(r"[^a-z0-9]+", text.lower()):
        if not w or w in STOPWORDS or w in seen:
            continue
        if len(w) >= 4 or w in ALLOWED_SHORT:
            seen.add(w)
            out.append(w)
    return out


# Decision-intent META-vocabulary, derived from intent.py's phrase list
# (single source of truth -- extending the verb list automatically
# extends this set). Words like "double", "down", "pivot", "revisit",
# "decision", "lineage" mark THAT the user is deciding, not WHAT the
# topic is -- yet in a corpus full of decision narratives they score
# high IDF ("double" 4.19 / "down" 3.73 on the live corpus, measured
# 2026-08-19) and leaked 2/200 vetted items into the DNR block for the
# off-topic database-paper-review prompt class. Intent vocabulary gates
# WHETHER the hook looks; it must never count as topical evidence.
META_DECISION_TOKENS = frozenset(
    t
    for phrase in DECISION_INTENT_VERBS
    for t in re.split(r"[^a-z0-9]+", phrase.lower())
    if t and t not in STOPWORDS and (len(t) >= 4 or t in ALLOWED_SHORT)
)


def prompt_relevance_tokens(prompt: str) -> List[str]:
    """Significant tokens of a PROMPT for relevance gating: the output
    of :func:`significant_tokens` minus the decision-intent
    meta-vocabulary (see META_DECISION_TOKENS). Haystacks/items are
    matched as raw text, so this filter applies to the prompt side only.
    """
    return [t for t in significant_tokens(prompt) if t not in META_DECISION_TOKENS]


def word_boundary_match(token: str, haystack: str) -> bool:
    """True when ``token`` appears in ``haystack`` on word boundaries.

    ``haystack`` must already be lowercased (tokens are produced
    lowercase by :func:`significant_tokens`).
    """
    return re.search(rf"\b{re.escape(token)}\b", haystack) is not None


class CorpusIdf:
    """Per-token IDF over a claude-mem project's ``chunks`` corpus.

    ``df`` is one FTS5 MATCH COUNT per token against ``chunks_fts`` --
    the exact inverted index the BM25 side queries -- so a token's
    document frequency reflects the same porter-stemmed vocabulary that
    ranking sees. Results are memoized per instance.

    All queries are reads; the connection never writes.
    """

    def __init__(self, db_path: Path) -> None:
        self._conn: Optional[sqlite3.Connection] = None
        self._cache: Dict[str, float] = {}
        self.corpus_size: int = 0
        try:
            self._conn = sqlite3.connect(db_path)
            self.corpus_size = int(self._conn.execute(
                "SELECT COUNT(*) FROM chunks"
            ).fetchone()[0])
        except sqlite3.Error:
            # No chunks table / unreadable DB: corpus_size stays 0, which
            # puts every gate in the small-corpus overlap fallback.
            self.corpus_size = 0

    @property
    def usable(self) -> bool:
        """True when the corpus is large enough for IDF to be meaningful."""
        return self.corpus_size >= MIN_IDF_CORPUS

    def _df(self, token: str) -> int:
        if self._conn is None:
            return 0
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH ?",
                (f'"{token}"',),
            ).fetchone()
            return int(row[0])
        except sqlite3.Error:
            # Broken/absent FTS index: report df = corpus_size so idf
            # collapses toward 0 -- "cannot establish distinctiveness"
            # must fail CLOSED (no injection), never open.
            return self.corpus_size

    def idf(self, token: str) -> float:
        """Return ``ln((N + 1) / (df + 1))`` for ``token`` (memoized)."""
        if token in self._cache:
            return self._cache[token]
        value = math.log((self.corpus_size + 1) / (self._df(token) + 1))
        value = max(value, 0.0)
        self._cache[token] = value
        return value

    def close(self) -> None:
        """Close the read connection (idempotent)."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "CorpusIdf":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


@dataclass
class GateResult:
    """Outcome of a relevance gate evaluation.

    ``matched`` maps each word-boundary-matched token to its idf
    (0.0 for every token in the small-corpus fallback regime, where idf
    is not computed). ``idf_sum`` is the sum over ``matched``.
    """
    passed: bool
    matched: Dict[str, float]
    idf_sum: float


def _capped(tokens: List[str]) -> List[str]:
    return tokens[:MAX_IDF_TOKENS]


def lineage_gate(tokens: List[str], haystack: str, idf: CorpusIdf) -> GateResult:
    """Evaluate the thread-lineage injection gate (B1).

    ``tokens`` are the prompt's significant tokens; ``haystack`` is the
    thread's lowercased decision-identifying text. IDF regime: >= 2
    word-boundary matches AND matched idf sum >= LINEAGE_IDF_SUM_THRESHOLD
    AND >= 1 matched token with idf >= DISTINCTIVE_IDF. Fallback regime
    (corpus < MIN_IDF_CORPUS): >= 2 word-boundary matches.
    """
    matched_tokens = [
        t for t in _capped(tokens) if word_boundary_match(t, haystack)
    ]
    if not idf.usable:
        matched = {t: 0.0 for t in matched_tokens}
        return GateResult(
            passed=len(matched_tokens) >= MIN_MATCHES,
            matched=matched, idf_sum=0.0,
        )
    matched = {t: idf.idf(t) for t in matched_tokens}
    idf_sum = sum(matched.values())
    passed = (
        len(matched) >= MIN_MATCHES
        and idf_sum >= LINEAGE_IDF_SUM_THRESHOLD
        and any(v >= DISTINCTIVE_IDF for v in matched.values())
    )
    return GateResult(passed=passed, matched=matched, idf_sum=idf_sum)


def dnr_item_gate(tokens: List[str], item_text: str, idf: CorpusIdf) -> GateResult:
    """Evaluate the DO-NOT-REBUILD per-item relevance filter (B3b).

    IDF regime: the item passes when >= DNR_MIN_DISTINCTIVE_MATCHES of
    the prompt's DISTINCTIVE tokens (idf >= DISTINCTIVE_IDF) word-
    boundary-match ``item_text``. Generic-token matches ("build",
    "test") carry no weight here at all -- this is deliberately stricter
    than the lineage gate because the DNR search has no relevance floor
    of its own (search.py returns SOMETHING for nearly any query).
    Fallback regime (corpus < MIN_IDF_CORPUS): >= MIN_MATCHES word-
    boundary matches of any significant token.

    ``item_text`` must already be lowercased.
    """
    capped = _capped(tokens)
    if not idf.usable:
        matched_tokens = [t for t in capped if word_boundary_match(t, item_text)]
        matched = {t: 0.0 for t in matched_tokens}
        return GateResult(
            passed=len(matched_tokens) >= MIN_MATCHES,
            matched=matched, idf_sum=0.0,
        )
    distinctive = [t for t in capped if idf.idf(t) >= DISTINCTIVE_IDF]
    matched = {
        t: idf.idf(t) for t in distinctive if word_boundary_match(t, item_text)
    }
    return GateResult(
        passed=len(matched) >= DNR_MIN_DISTINCTIVE_MATCHES,
        matched=matched, idf_sum=sum(matched.values()),
    )
