"""Tests for the claude-mem tool-use (PreToolUse) hook.

Mirrors the test_hooks.py pattern (in-memory seeded index + CliRunner
invocation) and exercises the four extraction paths (Edit / Write /
Bash / Agent) plus the silent paths (no-intent, self-referential,
read-only tools)."""
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


def _seed(db: Path) -> None:
    init_db(db)
    ing = Ingester(db_path=db, embedder=_ConstEmbedder())
    ing.add(Chunk(
        content=(
            "ddx_differential_intent_emit_kernel handles differential "
            "rivals via the cb dispatcher. Already production-wired."
        ),
        source="doc", module="ddx_differential",
        status="PRODUCTION", do_not_rebuild=True, signal_weight=80,
    ))
    ing.close()


def _invoke_tool_use(runner: CliRunner, tmp: str, payload: dict):
    return runner.invoke(
        cli, ["tool-use", "--project-root", tmp, "--stdin"],
        input=json.dumps(payload),
    )


def test_agent_dispatch_with_build_intent_emits_nudge(monkeypatch):
    """The bullseye case: Mike's exact spec -- assistant dispatches an
    Agent for 'build the differential dispatcher' and the hook nudges."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.tool_use.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Agent",
            "tool_input": {
                "subagent_type": "general-purpose",
                "description": "build differential dispatcher",
                "prompt": "Implement a new differential dispatcher for cb rivals.",
            },
        })
        assert result.exit_code == 0, result.output
        assert "NUDGE" in result.output
        assert "differential" in result.output.lower()
        # Soft framing assertions
        assert "not a block" in result.output.lower()


def test_write_tool_with_build_intent_emits_nudge(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.tool_use.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Write",
            "tool_input": {
                "file_path": "scripts/ddx_new.py",
                "content": "# Let's build a differential dispatcher",
            },
        })
        assert result.exit_code == 0, result.output
        assert "NUDGE" in result.output


def test_bash_tool_with_investigation_intent_emits_nudge(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.tool_use.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Bash",
            "tool_input": {
                "description": "investigate why the differential gate stopped firing",
                "command": "grep -rn differential scripts/",
            },
        })
        assert result.exit_code == 0, result.output
        assert "NUDGE" in result.output


def test_edit_tool_with_build_intent_emits_nudge(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        monkeypatch.setattr(
            "claude_mem.hooks.tool_use.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "scripts/cb_dispatcher.py",
                "old_string": "existing_thing",
                "new_string": (
                    "# Let's add a new differential dispatcher branch here"
                ),
            },
        })
        assert result.exit_code == 0, result.output
        assert "NUDGE" in result.output


def test_silent_on_read_class_tool():
    """Read / Grep / Glob / TodoWrite never fire the hook (lookup-only,
    no rebuild risk)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        for tool in ("Read", "Grep", "Glob", "TodoWrite"):
            result = _invoke_tool_use(runner, tmp, {
                "tool_name": tool,
                "tool_input": {"description": "build a new dispatcher"},
            })
            assert result.exit_code == 0, result.output
            assert result.output.strip() == "", (
                f"tool {tool} should be silent; got: {result.output}"
            )


def test_silent_when_no_build_intent():
    """A descriptive Bash command without build/investigation verbs
    does not fire the hook (intent gate is the noise filter)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Bash",
            "tool_input": {
                "description": "list files",
                "command": "ls -la",
            },
        })
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


def test_silent_when_self_referential():
    """Loop-back guard: editing claude-mem itself does not nudge on
    claude-mem chunks (otherwise meta-recursion fires on every config
    edit)."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        _seed(Path(tmp) / ".claude-mem" / "index.db")
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Edit",
            "tool_input": {
                "file_path": "tools/claude_mem/claude_mem/cli.py",
                "old_string": "old",
                "new_string": "build the differential dispatcher inside claude-mem",
            },
        })
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


def test_silent_when_no_db_present():
    """If .claude-mem/ is not initialized, the hook is a no-op (does
    not crash)."""
    with tempfile.TemporaryDirectory() as tmp:
        # NO init -- no db_path file
        runner = CliRunner()
        result = _invoke_tool_use(runner, tmp, {
            "tool_name": "Agent",
            "tool_input": {
                "prompt": "build a new differential dispatcher",
            },
        })
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


def test_missing_tool_name_is_noop():
    """No tool_name in payload -> silent no-op."""
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        result = _invoke_tool_use(runner, tmp, {
            "tool_input": {"prompt": "build the differential dispatcher"},
        })
        assert result.exit_code == 0, result.output
        assert result.output.strip() == ""


def test_install_hooks_registers_pretooluse_with_matcher():
    """install-hooks should register the PreToolUse hook with the
    write-class matcher so Read / Grep / Glob calls never trigger the
    hook invocation."""
    with tempfile.TemporaryDirectory() as tmp:
        settings = Path(tmp) / "settings.json"
        runner = CliRunner()
        result = runner.invoke(cli, [
            "install-hooks", "--no-shims",
            "--settings-path", str(settings),
            "--exe-path", "/abs/claude-mem",
        ])
        assert result.exit_code == 0, result.output
        data = json.loads(settings.read_text())
        assert "PreToolUse" in data["hooks"]
        entries = data["hooks"]["PreToolUse"]
        # Find the entry whose matcher includes the write-class tools
        matcher_entry = next(
            (e for e in entries if "Edit" in (e.get("matcher") or "")),
            None,
        )
        assert matcher_entry is not None
        assert "Edit" in matcher_entry["matcher"]
        assert "Write" in matcher_entry["matcher"]
        assert "Bash" in matcher_entry["matcher"]
        assert "Agent" in matcher_entry["matcher"]
        cmds = [h["command"] for h in matcher_entry["hooks"]]
        assert any("tool-use" in c for c in cmds)
        assert any("--stdin" in c for c in cmds)
