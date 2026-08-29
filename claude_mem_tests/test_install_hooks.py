import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.cli import cli


def test_install_hooks_writes_shims_to_specified_dir():
    with tempfile.TemporaryDirectory() as tmp:
        hook_dir = Path(tmp)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["install-hooks", "--hook-dir", str(hook_dir)]
        )
        assert result.exit_code == 0, result.output
        for name in ("SessionStart.sh", "UserPromptSubmit.sh", "Stop.sh"):
            p = hook_dir / name
            assert p.is_file(), f"missing {name}"
            content = p.read_text(encoding="utf-8")
            assert "claude-mem" in content
            # Fail-loud + exit-0 invariants
            assert "exit 0" in content
            assert "WARN" in content


def test_install_hooks_creates_dir_if_missing():
    with tempfile.TemporaryDirectory() as tmp:
        nested = Path(tmp) / "fresh" / "hooks"
        runner = CliRunner()
        result = runner.invoke(
            cli, ["install-hooks", "--hook-dir", str(nested)]
        )
        assert result.exit_code == 0, result.output
        assert (nested / "SessionStart.sh").is_file()


def test_install_hooks_patches_settings_when_path_provided():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / ".claude" / "settings.local.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--hook-dir", tmp,
            "--settings-path", str(settings),
            "--exe-path", "/abs/path/to/claude-mem",
        ])
        assert result.exit_code == 0, result.output
        assert settings.is_file()
        data = json.loads(settings.read_text(encoding="utf-8"))
        assert "hooks" in data
        for event in ("SessionStart", "UserPromptSubmit"):
            assert event in data["hooks"]
            cmds = [
                h["command"]
                for entry in data["hooks"][event]
                for h in entry.get("hooks", [])
            ]
            assert any("/abs/path/to/claude-mem" in c for c in cmds), event
            assert any("--stdin" in c for c in cmds), event
        # Stop is intentionally NOT registered (slow + blocks Claude Code)
        assert "Stop" not in data["hooks"] or not any(
            "claude-mem" in h["command"]
            for entry in data["hooks"].get("Stop", [])
            for h in entry.get("hooks", [])
        ), "Stop hook should not be registered in settings"


def test_install_hooks_settings_preserves_existing_entries():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "SessionStart": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": "python scripts/gen_project_state.py --print",
                    }],
                }],
            },
        }), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(settings.read_text(encoding="utf-8"))
        cmds = [
            h["command"]
            for entry in data["hooks"]["SessionStart"]
            for h in entry.get("hooks", [])
        ]
        assert any("gen_project_state" in c for c in cmds)
        assert any("claude-mem" in c for c in cmds)


def test_install_hooks_settings_is_idempotent():
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        for _ in range(2):
            runner.invoke(cli, [
                "install-hooks", "--no-shims",
                "--settings-path", str(settings),
                "--exe-path", "/abs/claude-mem",
            ])
        data = json.loads(settings.read_text(encoding="utf-8"))
        for event in ("SessionStart", "UserPromptSubmit"):
            cmds = [
                h["command"]
                for entry in data["hooks"][event]
                for h in entry.get("hooks", [])
            ]
            cm_cmds = [c for c in cmds if "claude-mem" in c]
            assert len(cm_cmds) == 1, f"{event}: {cm_cmds}"


def test_install_hooks_no_shims_skips_shim_files():
    with tempfile.TemporaryDirectory() as tmp:
        hook_dir = Path(tmp) / "hooks"
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--hook-dir", str(hook_dir),
            "--settings-path", str(settings),
            "--no-shims",
            "--exe-path", "/abs/claude-mem",
        ])
        assert result.exit_code == 0, result.output
        assert not (hook_dir / "SessionStart.sh").is_file()
        assert settings.is_file()


def test_install_hooks_shims_use_stdin_flag():
    with tempfile.TemporaryDirectory() as tmp:
        hook_dir = Path(tmp)
        runner = CliRunner()
        result = runner.invoke(
            cli, ["install-hooks", "--hook-dir", str(hook_dir)],
        )
        assert result.exit_code == 0, result.output
        for name in ("SessionStart.sh", "UserPromptSubmit.sh", "Stop.sh"):
            content = (hook_dir / name).read_text(encoding="utf-8")
            assert "--stdin" in content, name


def test_install_hooks_prunes_stale_stop_entry_on_reinstall():
    """A settings.json with a previously-registered claude-mem Stop hook
    gets it pruned on re-install (Stop is no longer in the default set)."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        settings.write_text(json.dumps({
            "hooks": {
                "Stop": [{
                    "matcher": "",
                    "hooks": [{
                        "type": "command",
                        "command": "/old/claude-mem session-end --stdin",
                        "timeout": 10000,
                    }],
                }],
            },
        }), encoding="utf-8")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(settings.read_text(encoding="utf-8"))
        stop_cmds = [
            h["command"]
            for entry in data["hooks"].get("Stop", [])
            for h in entry.get("hooks", [])
        ]
        assert not any("claude-mem" in c for c in stop_cmds), \
            f"stale Stop entry should be pruned, got: {stop_cmds}"


def test_install_hooks_settings_timeouts_are_short():
    """Per-turn UserPromptSubmit timeout must be <= 5s so a dead Ollama
    does not block Claude Code for minutes."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
        ])
        data = json.loads(settings.read_text(encoding="utf-8"))
        ups = data["hooks"]["UserPromptSubmit"][0]["hooks"][0]
        assert ups["timeout"] <= 5000, ups


def test_install_hooks_registers_session_end_capture_pipeline():
    """SessionEnd carries the capture pipeline (session-end,
    capture-extract, capture-synthesize) — load-bearing for the
    memory-health gate's hook-heartbeat checks (a project wired without
    them reports RED heartbeats). capture-synthesize takes no --stdin;
    with --project-root given, the pipeline bakes it explicitly so the
    wiring is cwd-independent."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
            "--project-root", "e:/someproj",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(settings.read_text(encoding="utf-8"))
        hooks = [
            h
            for entry in data["hooks"]["SessionEnd"]
            for h in entry.get("hooks", [])
        ]
        by_subcmd = {h["command"].split()[1]: h for h in hooks}
        assert set(by_subcmd) == {
            "session-end", "capture-extract", "capture-synthesize",
        }, by_subcmd
        assert "--stdin" in by_subcmd["session-end"]["command"]
        assert "--stdin" in by_subcmd["capture-extract"]["command"]
        assert "--stdin" not in by_subcmd["capture-synthesize"]["command"]
        for h in by_subcmd.values():
            assert "--project-root e:/someproj" in h["command"], h
        assert by_subcmd["capture-synthesize"]["timeout"] == 120000


def test_install_hooks_session_end_idempotent_three_entries():
    """Re-running install-hooks leaves exactly the three SessionEnd
    entries (prune-and-readd, no duplicates)."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        for _ in range(2):
            runner.invoke(cli, [
                "install-hooks", "--no-shims",
                "--settings-path", str(settings),
                "--exe-path", "/abs/claude-mem",
            ])
        data = json.loads(settings.read_text(encoding="utf-8"))
        cmds = [
            h["command"]
            for entry in data["hooks"]["SessionEnd"]
            for h in entry.get("hooks", [])
        ]
        assert len([c for c in cmds if "claude-mem" in c]) == 3, cmds


def test_install_hooks_without_project_root_omits_flag():
    """Without --project-root the capture pipeline registers cwd-derived
    (no baked --project-root), matching the pre-extension behavior for
    the four original events."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
        ])
        data = json.loads(settings.read_text(encoding="utf-8"))
        all_cmds = [
            h["command"]
            for event_entries in data["hooks"].values()
            for entry in event_entries
            for h in entry.get("hooks", [])
        ]
        assert not any("--project-root" in c for c in all_cmds), all_cmds
