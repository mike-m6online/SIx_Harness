"""Curated bidirectional acronym <-> expansion map for retrieval aliasing.

Audit-verified (2026-06-04). A chunk's derived alias string lets BM25 retrieve
it by EITHER the acronym or its expansion. Precision rules:
  - acronym -> expansion (Case A): whole-token match; case-INsensitive for
    distinctive acronyms, case-SENSITIVE against `surface` forms for the
    false-friend-prone short ones (ACC vs accuracy, FM, GW, ToM).
  - expansion -> acronym (Case B): contiguous case-insensitive phrase match;
    always safe (the expansions are distinctive multi-word phrases).
Deliberately excluded (would inject noise): PE/"prediction error" (the
expansion is the single most common term in the corpus); the stray
"cognitive world model" / "masked language model" usages (the canonical
concepts are CAUSAL world model and MOTIVATED language model)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class Acronym:
    acronym: str
    expansions: List[str]
    surface: Optional[List[str]] = None


ACRONYMS: List[Acronym] = [
    Acronym("cwm", ["causal world model"]),
    Acronym("ddx", ["differential diagnosis"]),
    Acronym("mlm", ["motivated language model"]),
    Acronym("slm", ["small language model"]),
    Acronym("r-slm", ["right small language model", "right-hemisphere small language model"]),
    Acronym("hss", ["hidden state server"]),
    Acronym("gw", ["global workspace"], surface=["GW"]),
    Acronym("kmi", ["knowledge-mode interoception"]),
    Acronym("rsi", ["recursive self-improvement"]),
    Acronym("acc", ["anterior cingulate cortex"], surface=["ACC"]),
    Acronym("lsh", ["locality-sensitive hashing"]),
    Acronym("ema", ["exponential moving average"]),
    Acronym("jepa", ["joint embedding predictive architecture"]),
    Acronym("tom", ["theory of mind"], surface=["ToM"]),
    Acronym("fm", ["forward model"], surface=["FM"]),
    Acronym("gnn", ["graph neural network"]),
]


def _whole_token_ci(token: str, text_lower: str) -> bool:
    return re.search(
        rf"(?<![a-z0-9]){re.escape(token.lower())}(?![a-z0-9])", text_lower
    ) is not None


def _whole_token_cs(token: str, text: str) -> bool:
    return re.search(
        rf"(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])", text
    ) is not None


def derive_aliases(text: str) -> str:
    """Return a ' | '-joined, sorted, deduped alias string for a chunk.
    Bidirectional: acronym-in-text -> expansions; expansion-phrase-in-text ->
    acronym. Acronym detection is whole-token (case-sensitive for the
    false-friend-prone entries via `surface`)."""
    if not text:
        return ""
    low = text.lower()
    out = set()
    for entry in ACRONYMS:
        if entry.surface is None:
            present = _whole_token_ci(entry.acronym, low)
        else:
            present = any(_whole_token_cs(sf, text) for sf in entry.surface)
        if present:
            out.update(entry.expansions)
        for exp in entry.expansions:
            if exp in low:
                out.add(entry.acronym)
    return " | ".join(sorted(out))
