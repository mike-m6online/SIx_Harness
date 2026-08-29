"""Tests for the harness-content filter (spec R1).

claude-mem's corpus was contaminated by harness-injected pseudo-user
content -- skill-file bodies, task-notifications, system reminders, hook
stdout, compaction summaries -- ingested as role="user" and phrase-graded
as if it were genuine user prose. `is_harness_content` / `harness_reason`
are the root fix; this module also verifies the grading path (bulk.py's
session-message ingestion in cli.py) and extract_decisions.py's candidate
mining both honor the filter.
"""
import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.filters import is_harness_content, harness_reason
from claude_mem.extract_decisions import scan_candidates


# ---------------------------------------------------------------------------
# is_harness_content / harness_reason -- marker coverage
# ---------------------------------------------------------------------------

def test_skill_file_body_is_harness_content():
    text = (
        "Base directory for this skill: "
        "C:\\Users\\someuser\\.claude\\plugins\\cache\\claude-plugins-official"
        "\\superpowers\\5.0.7\\skills\\brainstorming\n"
        "# Brainstorming Ideas Into Designs\n"
        "Help turn ideas into fully formed designs and specs through "
        "natural collaborative dialogue."
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_task_notification_is_harness_content():
    text = (
        "<task-notification>\n"
        "<task-id>be3rnmevh</task-id>\n"
        '<summary>Monitor event: "Cell 1 progress (FP16+neuralink)"</summary>\n'
        "<event>[Monitor timed out -- re-arm if needed.]</event>\n"
        "</task-notification>"
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_system_reminder_is_harness_content():
    text = (
        "<system-reminder>Respond with just the action or changes and "
        "without a thinking block, unless this is a redesign or requires "
        "fresh reasoning.</system-reminder>"
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_command_name_is_harness_content():
    text = (
        "<command-name>/compact</command-name>\n"
        "<command-message>compact</command-message>\n"
        "<command-args></command-args>"
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_local_command_caveat_is_harness_content():
    text = (
        "<local-command-caveat>Caveat: The messages below were generated "
        "by the user while running local commands. DO NOT respond to "
        "these messages or otherwise consider them in your response "
        "unless the user explicitly asks you to.</local-command-caveat>"
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_hook_success_prefix_is_harness_content():
    text = "hook success: memory context loaded (3 checkpoints, 2 corrections)"
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_user_prompt_submit_hook_success_is_harness_content():
    text = "UserPromptSubmit hook success: injected 4 relevant memories"
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_session_start_hook_prefix_is_harness_content():
    text = "SessionStart hook: restored checkpoint from 2026-06-30"
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_compaction_summary_is_harness_content():
    text = (
        "This session is being continued from a previous conversation "
        "that ran out of context. The conversation is summarized below:\n"
        "We decided to keep the MLM and no shortcuts were taken."
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


def test_ide_selection_is_harness_content():
    text = (
        "<ide_selection>The user selected the following text in "
        "bulk.py:\ndef signal_weight(...):\n</ide_selection>"
    )
    assert is_harness_content(text) is True
    assert harness_reason(text) is not None


# ---------------------------------------------------------------------------
# Negative cases -- the phrase-grading bait. Genuine user prose containing
# the exact correction/decision trigger phrases must NOT be caught by the
# harness filter (it is a content-origin filter, not a phrase filter).
# ---------------------------------------------------------------------------

def test_genuine_user_correction_with_no_shortcuts_phrase_is_not_harness():
    text = (
        "No shortcuts here -- we already built the apollo loop, go read "
        "engine.cpp before you propose a new kernel."
    )
    assert is_harness_content(text) is False
    assert harness_reason(text) is None


def test_genuine_user_decision_with_the_approach_phrase_is_not_harness():
    text = (
        "The approach going forward is to gate graphify behind the "
        "health check -- that's the design we're locking in."
    )
    assert is_harness_content(text) is False
    assert harness_reason(text) is None


def test_plain_status_question_is_not_harness():
    assert is_harness_content("status?") is False
    assert harness_reason("status?") is None


def test_ordinary_assistant_prose_is_not_harness():
    text = (
        "I read bulk.py and cli.py; the grading path lives in the bulk "
        "CLI command, not in bulk.py itself."
    )
    assert is_harness_content(text) is False
    assert harness_reason(text) is None


def test_prose_mentioning_hook_by_name_is_not_harness():
    """A genuine message that talks ABOUT a hook (doesn't open with the
    harness prefix) must not be caught -- guards against over-broad
    substring matching on 'hook' anywhere in the text."""
    text = (
        "Honest evaluation: the SessionStart hook design is better than "
        "mine in one key way, and the gap is the right place to focus."
    )
    assert is_harness_content(text) is False
    assert harness_reason(text) is None


# ---------------------------------------------------------------------------
# bulk.py grading path (wired via cli.py's `bulk` command): harness-True
# text must get sw=0 / is_correction=0 / is_decision=0 regardless of
# phrase content, and the chunk must not reach the index.
# ---------------------------------------------------------------------------

def test_bulk_grader_zeroes_harness_content_regardless_of_phrases(monkeypatch):
    """A session JSONL containing a skill-file-body message (with corr-
    ection/decision bait phrases baked in) must be filtered out entirely
    -- it must not appear in the bulk_sample.json review file."""
    from claude_mem.cli import cli

    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024

    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])

        proj_slug = str(Path(tmp).resolve()).replace(":", "-").replace(
            "\\", "-"
        ).replace("/", "-")
        home = Path(tmp) / "fakehome"
        proj_dir = home / ".claude" / "projects" / f"session-{proj_slug}-x"
        proj_dir.mkdir(parents=True)
        jsonl = proj_dir / "sess.jsonl"
        harness_text = (
            "Base directory for this skill: /some/skills/example\n"
            "we already built that. no shortcuts. the approach is locked."
        )
        genuine_text = (
            "no, we already built that -- check the existing apollo loop."
        )
        jsonl.write_text(
            "\n".join([
                json.dumps({"message": {"role": "user", "content": harness_text}}),
                json.dumps({"message": {"role": "user", "content": genuine_text}}),
            ]) + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient", lambda **kw: _ConstEmbedder()
        )
        result = runner.invoke(
            cli,
            ["bulk", "--project-root", tmp, "--sample", "50",
             "--no-include-git"],
        )
        assert result.exit_code == 0, result.output
        sample = Path(tmp) / ".claude-mem" / "bulk_sample.json"
        data = json.loads(sample.read_text())
        heads = [r["head"] for r in data]
        assert not any("Base directory for this skill" in h for h in heads), heads
        assert any("check the existing apollo loop" in h for h in heads), heads


def test_bulk_summary_reports_filtered_harness_count(monkeypatch):
    """The bulk command's summary line reports how many chunks the
    harness filter dropped, so the count is auditable without a DB query."""
    from claude_mem.cli import cli

    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024

    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])

        proj_slug = str(Path(tmp).resolve()).replace(":", "-").replace(
            "\\", "-"
        ).replace("/", "-")
        home = Path(tmp) / "fakehome"
        proj_dir = home / ".claude" / "projects" / f"session-{proj_slug}-x"
        proj_dir.mkdir(parents=True)
        jsonl = proj_dir / "sess.jsonl"
        jsonl.write_text(
            json.dumps({
                "message": {
                    "role": "user",
                    "content": "<system-reminder>be terse</system-reminder>",
                }
            }) + "\n",
            encoding="utf-8",
        )

        monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient", lambda **kw: _ConstEmbedder()
        )
        result = runner.invoke(
            cli, ["bulk", "--project-root", tmp, "--no-include-git"]
        )
        assert result.exit_code == 0, result.output
        assert "harness" in result.output.lower(), result.output


# ---------------------------------------------------------------------------
# extract_decisions.py candidate mining honors the filter
# ---------------------------------------------------------------------------

def test_scan_candidates_skips_harness_content_sentences():
    """A skill-file body containing decision/dead-end cue phrases must
    not produce candidates; a genuine adjacent message still does."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = Path(tmp) / "session.jsonl"
        harness_msg = (
            "Base directory for this skill: /skills/foo\n"
            "We decided to use this skill. This approach is rejected "
            "elsewhere but locked in here."
        )
        genuine_msg = ("We decided to go with the GW causal-write path "
                       "for the dialogue substrate.")
        jsonl.write_text(
            "\n".join([
                json.dumps({"message": {"role": "user", "content": harness_msg}}),
                json.dumps({"message": {"role": "user", "content": genuine_msg}}),
            ]) + "\n",
            encoding="utf-8",
        )
        decisions, dead_ends, _offset, _skips = scan_candidates(jsonl)
        titles = [d.title for d in decisions]
        assert not any("skill" in t.lower() for t in titles), titles
        assert any("GW causal-write path" in t for t in titles), titles
