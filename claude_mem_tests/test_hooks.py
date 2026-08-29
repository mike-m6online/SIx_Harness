import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.cli import cli
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _seed(db: Path):
    init_db(db)
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content="ddx_differential_intent_emit_kernel handles differential rivals",
        source="doc", module="ddx_differential",
        status="PRODUCTION", do_not_rebuild=True, signal_weight=50,
    ))
    ing.close()


def test_prompt_submit_emits_warning_on_build_intent(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        # B3: the per-item relevance filter requires >= 2 word-boundary
        # prompt-token matches per listed item, so the on-topic prompt
        # names two tokens of the seeded chunk (differential + rivals).
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "let's build a new differential rivals dispatcher",
        ])
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" in result.output
        assert "differential" in result.output.lower()


def test_prompt_submit_emits_warning_on_investigation_intent(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "let's investigate why the differential rivals gate stopped firing",
        ])
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" in result.output


def test_prompt_submit_silent_on_status_question(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp, "status?",
        ])
        assert result.exit_code == 0
        assert "DO NOT REBUILD" not in result.output


def test_prompt_submit_silent_when_no_index():
    """No .claude-mem/index.db -> hook returns nothing (does not crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "let's build a new thing",
        ])
        assert result.exit_code == 0
        assert "DO NOT REBUILD" not in result.output


def test_prompt_submit_fires_on_decision_only_corpus(monkeypatch):
    """Production-realistic regression: corpus seeded with is_decision=1
    (NOT do_not_rebuild=1) chunks must still trigger the DO NOT
    REBUILD warning. The strict DNR-only filter that prompt-submit
    used historically silenced this hook against any bulk-backfilled
    corpus; the broader filter_operator_vetted gate fixes it."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        # Seed an is_decision=1 / signal_weight=80 chunk (NOT DNR)
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content=(
                "decision: ddx_differential_intent_emit_kernel is the "
                "production path for differential rivals -- do not add "
                "parallel infrastructure"
            ),
            source="memory", module="ddx_differential",
            is_decision=True, signal_weight=80,
        ))
        ing.close()
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "let's build a new differential rivals dispatcher",
        ])
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" in result.output
        assert "differential" in result.output.lower()


def test_session_start_emits_top_signal_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        # Seed a high-signal correction chunk
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="we already built the apollo loop; do not rebuild",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        result = runner.invoke(cli, [
            "session-start", "--project-root", tmp,
        ])
        assert result.exit_code == 0
        assert "claude-mem" in result.output
        assert "apollo" in result.output.lower()


def test_session_end_noop_when_session_id_not_found():
    """No matching session JSONL -> session-end returns silently."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = runner.invoke(cli, [
            "session-end", "--project-root", tmp,
            "--session-id", "does_not_exist_xyz",
        ])
        assert result.exit_code == 0
        assert result.output.strip() == ""


def test_session_end_noop_when_session_id_empty():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = runner.invoke(cli, [
            "session-end", "--project-root", tmp,
        ])
        assert result.exit_code == 0
        assert result.output.strip() == ""


def test_prompt_submit_reads_prompt_from_stdin_json(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        stdin_json = json.dumps({
            "session_id": "test-sid",
            "cwd": tmp,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "let's build a new differential rivals dispatcher",
        })
        result = runner.invoke(
            cli, ["prompt-submit", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" in result.output


def test_prompt_submit_stdin_silent_on_status_question(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        stdin_json = json.dumps({
            "cwd": tmp,
            "hook_event_name": "UserPromptSubmit",
            "prompt": "status?",
        })
        result = runner.invoke(
            cli, ["prompt-submit", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        assert "DO NOT REBUILD" not in result.output


def test_prompt_submit_stdin_empty_json_is_noop():
    """Empty JSON / missing prompt -> hook exits silently, not a crash."""
    runner = CliRunner()
    result = runner.invoke(cli, ["prompt-submit", "--stdin"], input="")
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_session_start_uses_cwd_from_stdin_json():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="we already built the apollo loop; do not rebuild",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        stdin_json = json.dumps({
            "session_id": "test-sid",
            "cwd": tmp,
            "hook_event_name": "SessionStart",
            "source": "startup",
        })
        result = runner.invoke(
            cli, ["session-start", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        assert "apollo" in result.output.lower()


def test_session_start_derives_session_from_transcript_path():
    """A compact/resume-source SessionStart may omit session_id but still
    carries transcript_path (<...>/<session_id>.jsonl). The hook must derive
    the session id from its stem so the frozen-render guard (memory_health
    check 9) ARMS instead of sitting in the disarmed 'session unknown'
    fallback (the 2026-07-04 dormant-guard gap)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a correction so the render is non-empty",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        stdin_json = json.dumps({
            "transcript_path": str(Path(tmp) / "sess-from-transcript.jsonl"),
            "cwd": tmp,
            "hook_event_name": "SessionStart",
            "source": "compact",
        })
        result = runner.invoke(
            cli, ["session-start", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        from claude_mem.capture import CaptureStore
        store = CaptureStore(db)
        try:
            assert (store.get_meta("last_sessionstart_session")
                    == "sess-from-transcript")
        finally:
            store.close()


def test_session_start_prefers_explicit_session_id_over_transcript():
    """When both are present, the explicit session_id wins (transcript_path is
    only the fallback)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="a correction so the render is non-empty",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        stdin_json = json.dumps({
            "session_id": "explicit-sid",
            "transcript_path": str(Path(tmp) / "other-stem.jsonl"),
            "cwd": tmp,
            "hook_event_name": "SessionStart",
            "source": "startup",
        })
        result = runner.invoke(
            cli, ["session-start", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        from claude_mem.capture import CaptureStore
        store = CaptureStore(db)
        try:
            assert store.get_meta("last_sessionstart_session") == "explicit-sid"
        finally:
            store.close()


def test_session_start_records_session_id_from_stdin_json():
    """The receiving-session meta key (read by the health gate's check 9)
    must carry the hook payload's session_id end-to-end."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        db = Path(tmp) / ".claude-mem" / "index.db"
        init_db(db)
        ing = Ingester(db_path=db, embedder=_ConstEmbedder())
        ing.add(Chunk(
            content="we already built the apollo loop; do not rebuild",
            source="claude_code", role="user",
            signal_weight=100, is_correction=True,
        ))
        ing.close()
        stdin_json = json.dumps({
            "session_id": "sid-e2e",
            "cwd": tmp,
            "hook_event_name": "SessionStart",
            "source": "startup",
        })
        result = runner.invoke(
            cli, ["session-start", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        from claude_mem.capture import CaptureStore
        store = CaptureStore(db)
        try:
            assert store.get_meta("last_sessionstart_session") == "sid-e2e"
        finally:
            store.close()


def test_session_end_uses_session_id_from_stdin_json():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        stdin_json = json.dumps({
            "session_id": "does-not-exist-xyz",
            "cwd": tmp,
            "hook_event_name": "Stop",
        })
        result = runner.invoke(
            cli, ["session-end", "--stdin"], input=stdin_json,
        )
        assert result.exit_code == 0
        assert result.output.strip() == ""


# ---- system-generated turn exemption (day-9 shakedown fix) -----------------

def test_prompt_submit_task_notification_exempt(monkeypatch):
    """A <task-notification> turn gets no injection even when its text
    would trip every intent + relevance gate -- and the telemetry
    denominator still records the invocation with the exemption named."""
    import sqlite3
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "<task-notification>let's build a new differential rivals "
            "dispatcher</task-notification>",
        ])
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" not in result.output
        assert result.output.strip() == ""
        conn = sqlite3.connect(Path(tmp) / ".claude-mem" / "telemetry.db")
        row = conn.execute(
            "SELECT matched_token_summary, retrieved_chunk_count, "
            "build_intent_fired FROM wrapper_invocations"
        ).fetchone()
        conn.close()
        assert row is not None
        assert "system_turn_exempt" in (row[0] or "")
        assert "task-notification" in (row[0] or "")
        assert row[1] == 0
        assert not row[2]


def test_prompt_submit_quoted_tag_mid_text_not_exempt(monkeypatch):
    """Only a turn that STARTS with a system marker is exempt; a human
    prompt quoting the tag mid-sentence keeps the full pipeline."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.prompt_submit.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(cli, [
            "prompt-submit", "--project-root", tmp,
            "let's build a new differential rivals dispatcher like the "
            "<task-notification> handler does",
        ])
        assert result.exit_code == 0, result.output
        assert "DO NOT REBUILD" in result.output
