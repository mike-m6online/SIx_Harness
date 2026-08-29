from datetime import datetime, timedelta, timezone

from claude_mem.stale_check import (
    check_for_stale_claims, extract_symbol_references,
)


def test_extract_symbol_references_finds_qualified_names():
    content = (
        "the fix lives in scripts/knowledge_bridge.py at the "
        "_register_corrected_mechanism_as_new_hypothesis helper; "
        "the use_apollo flag controls dispatch."
    )
    syms = extract_symbol_references(content)
    assert "scripts/knowledge_bridge.py" in syms
    assert "use_apollo" in syms
    # Snake-case function name
    assert any("_register_corrected_mechanism" in s for s in syms)


def test_extract_symbol_references_empty_when_no_symbols():
    syms = extract_symbol_references("just a prose paragraph here")
    assert syms == []


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _days_ago_iso(n: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=n)).isoformat()


def test_check_for_stale_claims_fires_on_narrative_chunk_with_symbol():
    retrieved = [{
        "source": "memory",
        "date": _days_ago_iso(60),
        "ingested_at": _days_ago_iso(60),
        "content": "scripts/knowledge_bridge.py:1234 defines _foo_bar",
        "file_path": "memory/checkpoint_2026_03_14.md",
    }]
    warning = check_for_stale_claims(retrieved)
    assert warning is not None
    assert "STALE-CLAIM" in warning
    assert "knowledge_bridge.py" in warning


def test_check_for_stale_claims_silent_on_code_chunks():
    """A chunk whose source is `doc` (i.e. a regular project doc, not
    a Claude-authored narrative) does not trigger the warning."""
    retrieved = [{
        "source": "doc",
        "date": _days_ago_iso(60),
        "ingested_at": _days_ago_iso(60),
        "content": "scripts/knowledge_bridge.py:1234 defines _foo_bar",
        "file_path": "docs/api_reference.md",
    }]
    warning = check_for_stale_claims(retrieved)
    assert warning is None


def test_check_for_stale_claims_silent_when_no_symbols():
    """Narrative source with no symbol references -> no warning."""
    retrieved = [{
        "source": "memory",
        "date": _days_ago_iso(60),
        "ingested_at": _days_ago_iso(60),
        "content": "just narrative prose with no specific symbols",
        "file_path": "memory/snapshot.md",
    }]
    warning = check_for_stale_claims(retrieved)
    assert warning is None
