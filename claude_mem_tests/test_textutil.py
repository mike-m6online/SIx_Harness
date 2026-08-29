"""claude_mem.textutil: the shared whole-word clip + sentence helpers.

One clip definition replaces five independent bare-slice truncations
(session-start corrections, both hook formatters, CLI search echo, and
the candidate miner's permanently-stored titles).
"""
from claude_mem.textutil import ELLIPSIS, clip, collapse_ws, first_sentence


def test_short_text_passes_through_unchanged():
    assert clip("a short line", 40) == "a short line"


def test_exact_limit_not_clipped():
    text = "x" * 40
    assert clip(text, 40) == text


def test_clip_never_exceeds_limit():
    text = "word " * 100
    for limit in (10, 37, 120, 240):
        assert len(clip(text, limit)) <= limit


def test_clip_ends_on_word_boundary_with_ellipsis():
    text = "the quick brown fox jumps over the lazy dog repeatedly"
    out = clip(text, 20)
    assert out.endswith(ELLIPSIS)
    assert len(out) <= 20
    head = out[:-1]
    # Every token in the head is a whole source token -- no mid-word cut
    # ("the quick brown fox" fits in 19 chars; "jumps" would overflow).
    assert head == "the quick brown fox"


def test_clip_collapses_whitespace_and_newlines():
    text = "line one\nline   two\t\tline three"
    assert clip(text, 100) == "line one line two line three"


def test_clip_multiline_truncated_has_no_newlines():
    text = ("first line of a correction\nsecond line with more detail\n"
            "third line beyond any budget here")
    out = clip(text, 40)
    assert "\n" not in out
    assert out.endswith(ELLIPSIS)


def test_single_long_token_hard_cut():
    token = "a" * 100
    out = clip(token, 20)
    assert len(out) == 20
    assert out == "a" * 19 + ELLIPSIS


def test_zero_or_negative_limit_returns_empty():
    assert clip("anything", 0) == ""
    assert clip("anything", -5) == ""


def test_empty_text():
    assert clip("", 10) == ""
    assert clip("   \n\t ", 10) == ""


def test_collapse_ws():
    assert collapse_ws("  a\n\nb\t c  ") == "a b c"


def test_first_sentence_basic():
    assert first_sentence("First point. Second point.") == "First point."


def test_first_sentence_no_boundary_returns_all():
    assert first_sentence("no terminal punctuation here") == (
        "no terminal punctuation here"
    )


def test_first_sentence_collapses_whitespace():
    assert first_sentence("One\nsentence   spanning lines. Two.") == (
        "One sentence spanning lines."
    )


def test_first_sentence_empty():
    assert first_sentence("") == ""
    assert first_sentence("  \n ") == ""
