"""Tests for acronym alias map + derive_aliases (Task 1, acronym-resilience system).
Written FIRST (TDD) — all tests must FAIL with ImportError before acronyms.py exists.
"""
import pytest

from claude_mem.acronyms import derive_aliases


class TestDeriveAliasesBasic:
    def test_acronym_in_text_expands(self):
        result = derive_aliases("the CWM kernel")
        parts = result.split(" | ")
        assert "causal world model" in parts

    def test_expansion_in_text_gives_acronym(self):
        result = derive_aliases("a causal world model graph")
        parts = result.split(" | ")
        assert "cwm" in parts

    def test_empty_string_returns_empty(self):
        assert derive_aliases("") == ""


class TestCaseSensitiveFalseFriends:
    def test_acc_false_friend_accuracy_not_matched(self):
        result = derive_aliases("we tuned the accuracy and pred_acc metric")
        parts = result.split(" | ") if result else []
        assert "anterior cingulate cortex" not in parts
        assert "acc" not in parts

    def test_acc_surface_form_matched(self):
        result = derive_aliases("ACC conflict monitoring")
        parts = result.split(" | ")
        assert "anterior cingulate cortex" in parts

    def test_fm_surface_matched(self):
        result = derive_aliases("the FM head")
        parts = result.split(" | ")
        assert "forward model" in parts

    def test_fm_lowercase_not_matched(self):
        result = derive_aliases("fm radio confm")
        parts = result.split(" | ") if result else []
        assert "forward model" not in parts

    def test_gw_surface_matched(self):
        result = derive_aliases("the GW broadcast")
        parts = result.split(" | ")
        assert "global workspace" in parts

    def test_gw_lowercase_not_matched(self):
        result = derive_aliases("the gw broadcast")
        parts = result.split(" | ") if result else []
        assert "global workspace" not in parts

    def test_tom_surface_matched(self):
        result = derive_aliases("ToM reasoning")
        parts = result.split(" | ")
        assert "theory of mind" in parts

    def test_tom_lowercase_not_matched(self):
        result = derive_aliases("tom is a name")
        parts = result.split(" | ") if result else []
        assert "theory of mind" not in parts


class TestWholeTokenBoundaries:
    def test_cwm_underscore_bounded_matches(self):
        """Underscores are not [a-z0-9], so use_cwm_graph should match cwm."""
        result = derive_aliases("use_cwm_graph flag")
        parts = result.split(" | ")
        assert "causal world model" in parts

    def test_cwm_embedded_in_word_no_match(self):
        result = derive_aliases("scwmx nonsense")
        assert result == ""

    def test_phrase_direction_theory_of_mind(self):
        result = derive_aliases("the theory of mind module")
        parts = result.split(" | ")
        assert "tom" in parts


class TestDeterminismAndSorting:
    def test_deterministic_across_calls(self):
        text = "CWM and MLM and causal world model"
        assert derive_aliases(text) == derive_aliases(text)

    def test_result_is_sorted(self):
        text = "CWM and DDX and causal world model and differential diagnosis"
        result = derive_aliases(text)
        if result:
            parts = result.split(" | ")
            assert parts == sorted(parts)


class TestBidirectionalMultiEntry:
    def test_mlm_acronym_expands(self):
        result = derive_aliases("the MLM training step")
        parts = result.split(" | ")
        assert "motivated language model" in parts

    def test_mlm_expansion_gives_acronym(self):
        result = derive_aliases("the motivated language model trains")
        parts = result.split(" | ")
        assert "mlm" in parts

    def test_ddx_roundtrip(self):
        r1 = derive_aliases("DDX pipeline")
        r2 = derive_aliases("differential diagnosis module")
        assert "differential diagnosis" in r1.split(" | ")
        assert "ddx" in r2.split(" | ")

    def test_lsh_ci_acronym(self):
        """LSH has no surface list so should be case-insensitive."""
        result_upper = derive_aliases("the LSH buckets")
        result_lower = derive_aliases("the lsh buckets")
        parts_upper = result_upper.split(" | ")
        parts_lower = result_lower.split(" | ")
        assert "locality-sensitive hashing" in parts_upper
        assert "locality-sensitive hashing" in parts_lower

    def test_r_slm_acronym(self):
        result = derive_aliases("the R-SLM self-trains on dialogue")
        parts = result.split(" | ")
        assert "right small language model" in parts or "right-hemisphere small language model" in parts

    def test_no_false_expansion_from_common_words(self):
        """Ensure no spurious expansions from generic prose."""
        result = derive_aliases("we ran an experiment to measure accuracy")
        parts = result.split(" | ") if result else []
        assert "anterior cingulate cortex" not in parts
