import json
import tempfile
from pathlib import Path

from claude_mem.bulk import (
    is_correction, is_decision, parse_claude_code_jsonl, parse_markdown_doc,
    parse_memory_md, parse_progress_jsonl, signal_weight,
)


def test_signal_weight_user_correction_is_100():
    msg = "no, we already built that. check the existing apollo loop."
    assert signal_weight(msg, role="user") == 100


def test_signal_weight_user_decision_is_50():
    msg = "going forward, the design is to use the kmi flag."
    assert signal_weight(msg, role="user") == 50


def test_signal_weight_user_investigation_is_30():
    msg = "let's investigate why the verifier stops firing"
    assert signal_weight(msg, role="user") == 30


def test_signal_weight_user_other_is_10():
    assert signal_weight("status?", role="user") == 10


def test_signal_weight_assistant_filler_is_0():
    assert signal_weight("ok let me think about that", role="assistant") == 0


def test_signal_weight_assistant_decision_is_20():
    msg = "the design is to apply Fix E after the prior is drained."
    assert signal_weight(msg, role="assistant") == 20


def test_is_correction_and_is_decision_predicates():
    assert is_correction("we already built that", role="user")
    assert not is_correction("status?", role="user")
    assert is_decision("the design is X")
    assert not is_decision("status?")


def test_is_correction_requires_user_role():
    # A correction is the HUMAN correcting the assistant. Assistant prose
    # echoing a correction phrase ("you're right -- we already built X")
    # must never carry the flag: the pre-fix content-only grader marked
    # those, and RECENT CORRECTIONS surfaced assistant essays instead of
    # the operator's words.
    assert not is_correction(
        "you're right -- we already built the apollo loop", role="assistant")
    assert not is_correction("we already built that", role="system")
    assert is_correction("we already built that", role="user")


def test_parse_claude_code_jsonl_yields_messages():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        p.write_text(
            json.dumps({
                "type": "user",
                "message": {"role": "user", "content": "hello"},
                "timestamp": "2026-05-24T12:00:00Z",
            }) + "\n" +
            json.dumps({
                "type": "assistant",
                "message": {"role": "assistant", "content": "world"},
                "timestamp": "2026-05-24T12:00:01Z",
            }) + "\n",
            encoding="utf-8",
        )
        msgs = list(parse_claude_code_jsonl(p))
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"


def test_parse_claude_code_jsonl_flattens_content_blocks():
    """Anthropic content can be a list of blocks; parser should flatten
    the text-only ones into a single string."""
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "session.jsonl"
        p.write_text(
            json.dumps({
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "first "},
                        {"type": "text", "text": "second"},
                    ],
                },
                "timestamp": "2026-05-24T12:00:00Z",
            }) + "\n",
            encoding="utf-8",
        )
        msgs = list(parse_claude_code_jsonl(p))
        assert len(msgs) == 1
        assert msgs[0]["content"] == "first second"


def test_parse_markdown_doc_returns_list_of_section_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "doc.md"
        p.write_text("# heading\n\nsome content", encoding="utf-8")
        chunks = parse_markdown_doc(p)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        assert "heading" in chunks[0]["content"]
        assert chunks[0]["file_path"] == str(p)
        assert isinstance(chunks[0]["file_mtime"], float)
        assert chunks[0]["line_start"] >= 1
        assert chunks[0]["line_end"] >= chunks[0]["line_start"]


def test_parse_memory_md_strips_frontmatter():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "memory.md"
        p.write_text(
            "---\n"
            "name: feedback_test\n"
            "type: feedback\n"
            "---\n"
            "actual memory content lives here\n",
            encoding="utf-8",
        )
        chunks = parse_memory_md(p)
        assert len(chunks) == 1
        assert "actual memory content" in chunks[0]["content"]
        assert "frontmatter" not in chunks[0]["content"]


def test_parse_progress_jsonl_summarizes_last_row():
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "progress.jsonl"
        rows = [
            {"tick": 1000, "n_completed_tasks": 5, "coverage_total": 3,
             "ddx_intents_total": 100},
            {"tick": 6000, "n_completed_tasks": 200, "coverage_total": 12,
             "ddx_intents_total": 350},
        ]
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        chunks = parse_progress_jsonl(p)
        assert len(chunks) == 1
        assert "final_tick=6000" in chunks[0]["content"]
        assert "n_completed_tasks=200" in chunks[0]["content"]
