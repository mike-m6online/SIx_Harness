import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.cli import cli


def test_init_creates_state_dir():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--project-root", tmp])
        assert result.exit_code == 0, result.output
        state = Path(tmp) / ".claude-mem"
        assert state.is_dir()
        assert (state / "config.yaml").is_file()
        assert (state / "index.db").is_file()


def test_init_isolate_flag_writes_config():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["init", "--project-root", tmp, "--isolate"]
        )
        assert result.exit_code == 0, result.output
        cfg_text = (Path(tmp) / ".claude-mem" / "config.yaml").read_text()
        assert "isolate_from_cross_project: true" in cfg_text


def test_search_on_empty_index_returns_no_results(monkeypatch):
    """`claude-mem search` on a freshly-init'd empty index returns
    'no results'. Substitute the embedder so the test does not need
    Ollama running."""
    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["search", "--project-root", tmp, "anything"]
        )
        assert result.exit_code == 0
        assert "no results" in result.output.lower()


def test_search_without_init_exits_error():
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        result = runner.invoke(
            cli, ["search", "--project-root", tmp, "anything"]
        )
        assert result.exit_code == 1
        assert "claude-mem init" in result.output
