"""Hook heartbeat coverage (spec R4 + gate prereq, Task 5).

Every hook entry point writes a heartbeat row to telemetry.db on BOTH
success and failure -- Task 7's memory-health gate (check #3) reads
these to detect a hook that silently died (the cp1252 gen_decisions_state
crash class of bug). record_hook_heartbeat() must never raise even when
its own write fails (wrapped in its own try/except), and must never break
the calling hook.
"""
import sqlite3
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_mem.cli import cli
from claude_mem.schema import init_db
from claude_mem.telemetry import init_telemetry_db, record_hook_heartbeat


def _rows(telemetry_db: Path):
    conn = sqlite3.connect(telemetry_db)
    conn.row_factory = sqlite3.Row
    try:
        return [
            dict(r) for r in conn.execute(
                "SELECT hook, ok, detail, timestamp FROM hook_heartbeat "
                "ORDER BY id"
            ).fetchall()
        ]
    finally:
        conn.close()


def test_record_hook_heartbeat_writes_row():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        record_hook_heartbeat(db, hook="session_start", ok=True, detail="rendered 5 lines")
        rows = _rows(db)
        assert len(rows) == 1
        assert rows[0]["hook"] == "session_start"
        assert rows[0]["ok"] == 1
        assert rows[0]["detail"] == "rendered 5 lines"
        assert rows[0]["timestamp"]


def test_record_hook_heartbeat_records_failure():
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        init_telemetry_db(db)
        record_hook_heartbeat(db, hook="prompt_submit", ok=False, detail="RuntimeError: boom")
        rows = _rows(db)
        assert rows[0]["ok"] == 0
        assert "boom" in rows[0]["detail"]


def test_record_hook_heartbeat_creates_table_if_missing():
    """init_telemetry_db is idempotent and the heartbeat writer is
    self-sufficient -- calling it against a telemetry.db that was never
    explicitly init'd must not raise."""
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "telemetry.db"
        record_hook_heartbeat(db, hook="tool_use", ok=True, detail="")
        rows = _rows(db)
        assert len(rows) == 1


def test_record_hook_heartbeat_swallows_write_errors(monkeypatch, tmp_path):
    """A forced failure inside the heartbeat write itself must never
    propagate -- the calling hook's correctness always outranks
    heartbeat bookkeeping."""
    import claude_mem.telemetry as telemetry_mod

    db = tmp_path / "telemetry.db"
    init_telemetry_db(db)

    def _explode(*_a, **_k):
        raise sqlite3.OperationalError("simulated heartbeat failure")

    monkeypatch.setattr(telemetry_mod.sqlite3, "connect", _explode)
    # Must not raise.
    record_hook_heartbeat(db, hook="session_end", ok=True, detail="fine")


@pytest.mark.parametrize(
    "subcmd,args,stdin_json",
    [
        ("session-start", ["--project-root"], None),
        ("prompt-submit", ["--project-root"], None),
        ("session-end", ["--project-root"], None),
    ],
)
def test_hook_cli_entry_point_writes_heartbeat_on_success(subcmd, args, stdin_json):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        cmd = [subcmd] + args + [tmp]
        if subcmd == "prompt-submit":
            cmd = cmd + ["status?"]
        result = runner.invoke(cli, cmd)
        assert result.exit_code == 0, result.output
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        assert telemetry_db.is_file()
        rows = _rows(telemetry_db)
        hooks_seen = {r["hook"] for r in rows}
        expected_hook = subcmd.replace("-", "_")
        assert expected_hook in hooks_seen
        assert all(r["ok"] == 1 for r in rows if r["hook"] == expected_hook)


def test_tool_use_hook_writes_heartbeat():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = runner.invoke(cli, [
            "tool-use", "--project-root", tmp, "--tool-name", "Read",
        ])
        assert result.exit_code == 0
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        assert any(r["hook"] == "tool_use" and r["ok"] == 1 for r in rows)


def test_tool_use_post_hook_writes_heartbeat():
    import json as _json

    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        stdin_json = _json.dumps({
            "cwd": tmp, "tool_name": "Read",
            "tool_input": {"file_path": "x.py"}, "tool_response": "",
            "session_id": "sid",
        })
        result = runner.invoke(cli, ["tool-use-post", "--stdin"], input=stdin_json)
        assert result.exit_code == 0, result.output
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        assert any(r["hook"] == "tool_use_post" and r["ok"] == 1 for r in rows)


def test_capture_extract_hook_writes_heartbeat():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = runner.invoke(cli, [
            "capture-extract", "--project-root", tmp, "--skip-incremental",
        ])
        assert result.exit_code == 0, result.output
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        hooks_seen = {r["hook"] for r in rows}
        # Both the candidate-mining step and the incremental-ingest step
        # (or its skip) get their own heartbeat identity.
        assert "capture_extract" in hooks_seen


def test_capture_synthesize_hook_writes_heartbeat(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = runner.invoke(cli, [
            "capture-synthesize", "--project-root", tmp,
        ])
        assert result.exit_code == 0, result.output
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        assert any(r["hook"] == "capture_synthesize" for r in rows)


def test_hook_survives_corrupted_config_yaml(tmp_path):
    """Task-5 review carry-item: _heartbeat() must construct ProjectConfig
    INSIDE its try-guard. A corrupted config.yaml makes ProjectConfig raise
    (yaml.safe_load ParserError) while resolving telemetry_path; the hook
    must still exit 0 (degrade to no-heartbeat), never crash. Regression for
    the never-crash contract."""
    runner = CliRunner()
    runner.invoke(cli, ["init", "--project-root", str(tmp_path)])
    # Corrupt the config so ProjectConfig(...).telemetry_path raises.
    cfg_path = tmp_path / ".claude-mem" / "config.yaml"
    cfg_path.write_text("key: [unclosed\n  bad: : :\n", encoding="utf-8")
    # session-start goes through _heartbeat(); must not crash.
    result = runner.invoke(cli, ["session-start", "--project-root", str(tmp_path)])
    assert result.exit_code == 0, result.output


def test_hook_echo_absorbs_dead_pipe_oserror(monkeypatch):
    """Windows dead-pipe regression (memory-health RED, 2026-07-21): when
    the Claude Code process that spawned a hook exits before the hook body
    finishes, a stdout write raises OSError(22, 'Invalid argument')
    (POSIX: BrokenPipeError, an OSError subclass). _hook_echo must absorb
    it -- the reader is gone, the message has no destination, and the
    remaining hook work must proceed."""
    import claude_mem.cli as cli_mod

    def _dead_pipe(*_a, **_k):
        raise OSError(22, "Invalid argument")

    monkeypatch.setattr(cli_mod.click, "echo", _dead_pipe)
    cli_mod._hook_echo("late hook output")  # must not raise


def test_hook_echo_absorbs_closed_stdout_valueerror(monkeypatch):
    """A within-process closed stdout raises ValueError from write();
    same informational-write seam, same absorption contract."""
    import claude_mem.cli as cli_mod

    def _closed(*_a, **_k):
        raise ValueError("I/O operation on closed file")

    monkeypatch.setattr(cli_mod.click, "echo", _closed)
    cli_mod._hook_echo("late hook output")  # must not raise


def test_capture_extract_ingest_runs_when_stdout_dead(monkeypatch):
    """THE work-loss regression behind the 2026-07-21 RED gate: capture-
    extract's first stdout write sits BETWEEN candidate mining and
    incremental ingest. When that write raised (dead pipe), the ingest
    stage never ran and the heartbeat recorded ok=0 -- every SessionEnd
    from 2026-07-17 onward lost its incremental ingest. With _hook_echo,
    the ingest must run and the heartbeat must record ok=1 even with
    stdout dead for the whole command."""
    import claude_mem.cli as cli_mod
    import claude_mem.hooks.session_end as se_mod

    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])

        stages = []
        monkeypatch.setattr(
            se_mod, "run_candidates",
            lambda _sid, _root: stages.append("mine") or "mined 1 candidate",
        )
        monkeypatch.setattr(
            se_mod, "run_incremental_ingest",
            lambda _root, **_k: stages.append("ingest") or "ingested 2 chunks",
        )

        def _dead_pipe(*_a, **_k):
            raise OSError(22, "Invalid argument")

        monkeypatch.setattr(cli_mod.click, "echo", _dead_pipe)
        result = runner.invoke(cli, ["capture-extract", "--project-root", tmp])

        assert result.exit_code == 0
        assert stages == ["mine", "ingest"]
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        assert any(
            r["hook"] == "capture_extract" and r["ok"] == 1 for r in rows
        )


def test_session_start_hook_writes_failure_heartbeat_on_exception(monkeypatch):
    """A forced exception inside the session-start render must still leave
    a failure heartbeat row (ok=0) behind -- the whole point of the
    heartbeat is to make a hook crash visible the NEXT session."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        import claude_mem.hooks.session_start as ss_mod

        def _explode(_root):
            raise RuntimeError("forced session-start failure")

        monkeypatch.setattr(ss_mod, "run", _explode)
        result = runner.invoke(cli, ["session-start", "--project-root", tmp])
        # Fail-loud-but-never-block: hook still exits 0 for Claude Code,
        # but the heartbeat records the failure.
        assert result.exit_code == 0
        telemetry_db = Path(tmp) / ".claude-mem" / "telemetry.db"
        rows = _rows(telemetry_db)
        assert any(
            r["hook"] == "session_start" and r["ok"] == 0 for r in rows
        )
