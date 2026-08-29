"""Build-intent + investigation-intent detection.

Broadened from the origin project's upstream memory-system spec to catch the actual
failure pattern: real-world bad-rebuild prompts use investigate / trace
/ diagnose verbs more often than the explicit build / create / implement
construction verbs.
"""
from __future__ import annotations


# Task B3a precision fix (2026-08-19): the bare verbs "make" and
# "plan ", plus the naked openers "let's " and "we should ", matched
# virtually every imperative sentence ("Make a shopping list", "plan
# the meals", "let's grab lunch") and were the top drivers of the
# DO-NOT-REBUILD block's measured off-topic fire rate. Construction
# verbs (build/implement/create/design/add/write/wire/refactor) and the
# specific multiword proposal forms remain; "let's build ..." still
# fires via "build" itself.
BUILD_INTENT_VERBS = [
    "build", "create", "add ", "add a ", "add the ",
    "implement", "design", "architect",
    "develop", "write", "wire ", "refactor", "introduce", "propose",
    "spec ", "draft", "we could ",
    "what if we", "i think we need", "we need to add",
]

INVESTIGATION_INTENT_VERBS = [
    "investigate", "trace why", "understand why", "diagnose", "explore",
    "look at the", "look into", "why does", "what causes", "find out",
    "figure out", "dig into", "deep dive",
]

DECISION_INTENT_VERBS = [
    "should we", "should i", "double-down", "double down", "step back",
    "step-back", "do we pivot", "should we pivot", "or pivot",
    "or keep going", "or should we", "decision lineage", "design lineage",
    "which approach", "which direction", "double down or",
    "keep this approach", "revisit the decision", "revisiting the decision",
    "reconsider",
]

# Past-tense disqualifiers: when these phrases precede a build verb,
# the prompt is describing existing work, not proposing new work.
PAST_TENSE_DISQUALIFIERS = [
    "was built", "were built", "already built",
    "already implemented", "was implemented",
    "already added", "was added",
    "already created", "was created",
]


def has_build_intent(prompt: str) -> bool:
    p = prompt.lower()
    if any(d in p for d in PAST_TENSE_DISQUALIFIERS):
        return False
    return any(v in p for v in BUILD_INTENT_VERBS)


def has_investigation_intent(prompt: str) -> bool:
    p = prompt.lower()
    return any(v in p for v in INVESTIGATION_INTENT_VERBS)


def has_decision_intent(prompt: str) -> bool:
    p = prompt.lower()
    return any(v in p for v in DECISION_INTENT_VERBS)


def has_existing_subsystem_intent(prompt: str) -> bool:
    return (
        has_build_intent(prompt)
        or has_investigation_intent(prompt)
        or has_decision_intent(prompt)
    )
