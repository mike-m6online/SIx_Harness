import json
import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.bulk import detect_module
from claude_mem.cli import cli


def test_detect_module_word_boundary():
    modules = ["apollo", "kmi", "drives"]
    assert detect_module("use_apollo is the master switch", modules) == "apollo"
    assert detect_module("nothing here", modules) is None
    # "apollomania" should NOT match "apollo" (word boundary)
    assert detect_module("apollomania project", modules) is None


def test_bulk_sample_writes_review_file(monkeypatch):
    """`claude-mem bulk --sample 10` writes a review file Mike can inspect."""
    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        # Add a small doc tree so the bulk command has something to scan
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "alpha.md").write_text(
            "# alpha doc\nuse_apollo is the master switch", encoding="utf-8"
        )
        (docs / "beta.md").write_text(
            "# beta doc\nthe design is X going forward", encoding="utf-8"
        )
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli,
            ["bulk", "--project-root", tmp, "--sample", "10",
             "--no-include-git"],
        )
        assert result.exit_code == 0, result.output
        sample = Path(tmp) / ".claude-mem" / "bulk_sample.json"
        assert sample.is_file()
        data = json.loads(sample.read_text())
        assert isinstance(data, list)
        assert len(data) > 0
        assert any("apollo" in r["head"].lower() for r in data)


def test_bulk_full_indexes_chunks(monkeypatch):
    """`claude-mem bulk` (no sample) actually writes chunks to the index."""
    import sqlite3

    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024
    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        docs = Path(tmp) / "docs"
        docs.mkdir()
        (docs / "doc.md").write_text(
            "# doc\nthe design is to use the apollo loop", encoding="utf-8"
        )
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["bulk", "--project-root", tmp, "--no-include-git"]
        )
        assert result.exit_code == 0, result.output
        db = Path(tmp) / ".claude-mem" / "index.db"
        conn = sqlite3.connect(db)
        try:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        assert count >= 1


def test_bulk_sets_aliases_from_content(monkeypatch):
    """bulk doc loop calls derive_aliases(content) and stores result in aliases column."""
    import sqlite3

    class _ConstEmbedder:
        def embed(self, text: str) -> list[float]:
            return [0.1] * 1024

    with tempfile.TemporaryDirectory() as tmp:
        runner = CliRunner()
        runner.invoke(cli, ["init", "--project-root", tmp])
        docs = Path(tmp) / "docs"
        docs.mkdir()
        # "CWM" acronym -> derive_aliases should produce "causal world model"
        (docs / "cwm_doc.md").write_text(
            "# CWM design\nthe CWM kernel is the causal backbone",
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "claude_mem.cli.EmbeddingClient",
            lambda **kw: _ConstEmbedder(),
        )
        result = runner.invoke(
            cli, ["bulk", "--project-root", tmp, "--no-include-git"]
        )
        assert result.exit_code == 0, result.output
        db = Path(tmp) / ".claude-mem" / "index.db"
        conn = sqlite3.connect(db)
        try:
            rows = conn.execute(
                "SELECT aliases FROM chunks WHERE source = 'doc'"
            ).fetchall()
        finally:
            conn.close()
        # At least one doc chunk must carry the alias for "causal world model"
        aliases_values = [r[0] or "" for r in rows]
        assert any(
            "causal world model" in v for v in aliases_values
        ), f"No chunk aliases contained 'causal world model'; got: {aliases_values}"
