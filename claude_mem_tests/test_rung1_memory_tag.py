"""Rung-1 completeness: the memory ingest loop tags module + do_not_rebuild.

Two tests:
  1. Unit: detect_module works on a realistic memory-body string
     (proves the tagging call in the memory loop will fire correctly).
  2. Integration: full bulk CLI run over a tmp project whose memory dir
     contains a .md that mentions use_apollo; the DB row for that chunk
     must have module='use_apollo' and do_not_rebuild=1.
"""
import json
import sqlite3
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from claude_mem.bulk import detect_module
from claude_mem.cli import cli


# ---------------------------------------------------------------------------
# 1. Unit: detect_module fires on a realistic memory body
# ---------------------------------------------------------------------------

def test_detect_module_on_memory_body():
    """The memory loop calls detect_module(content, modules).
    Prove it returns the right name given a real memory-style body."""
    body = (
        "# Apollo hypothesis loop\n"
        "The use_apollo flag is the master switch for the Apollo C.1-C.5 substrate. "
        "do not rebuild this subsystem -- it is production.\n"
    )
    modules = ["use_apollo", "use_kmi", "use_drives"]
    assert detect_module(body, modules) == "use_apollo"


def test_detect_module_returns_none_when_no_match_in_memory_body():
    body = "# Session summary\nGeneral discussion about architecture.\n"
    modules = ["use_apollo", "use_kmi"]
    assert detect_module(body, modules) is None


# ---------------------------------------------------------------------------
# 2. Integration: bulk CLI writes module + do_not_rebuild into the DB row
# ---------------------------------------------------------------------------

class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def _write_project(root: Path) -> None:
    """Populate a minimal tmp project for the memory-tagging test."""
    # module_states YAML so collect_module_names returns "use_apollo"
    # and collect_do_not_rebuild_modules returns {"use_apollo"}.
    state_dir = root / "docs" / "marathon" / "module_states"
    state_dir.mkdir(parents=True)
    (state_dir / "use_apollo.state.yaml").write_text(
        "config_flag: use_apollo\ndo_not_rebuild: true\n",
        encoding="utf-8",
    )


def _write_memory_file(mem_dir: Path) -> None:
    """Write a memory .md whose body mentions use_apollo."""
    mem_dir.mkdir(parents=True, exist_ok=True)
    (mem_dir / "test_memory.md").write_text(
        "# Apollo hypothesis loop\n"
        "The use_apollo flag is the master switch for the Apollo C.1-C.5 substrate. "
        "do not rebuild this subsystem -- it is production.\n",
        encoding="utf-8",
    )


def test_bulk_memory_loop_tags_module_and_do_not_rebuild(monkeypatch, tmp_path):
    """Run the full bulk CLI over a tmp project; assert the memory chunk
    has module='use_apollo' and do_not_rebuild=1 in the DB."""
    root = tmp_path
    _write_project(root)

    # The memory dir the bulk command reads is
    #   ~/.claude/projects/<proj_slug>/memory/
    # We monkeypatch Path.home() so it resolves under our tmp tree.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()

    proj_slug = (
        str(root).replace(":", "-").replace("\\", "-").replace("/", "-")
    )
    mem_dir = fake_home / ".claude" / "projects" / proj_slug / "memory"
    _write_memory_file(mem_dir)

    monkeypatch.setattr("claude_mem.cli.Path.home", staticmethod(lambda: fake_home))
    monkeypatch.setattr("claude_mem.cli.EmbeddingClient", lambda **kw: _ConstEmbedder())

    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--project-root", str(root)])
    assert result.exit_code == 0, result.output

    result = runner.invoke(
        cli, ["bulk", "--project-root", str(root), "--no-include-git"]
    )
    assert result.exit_code == 0, result.output

    db_path = root / ".claude-mem" / "index.db"
    assert db_path.is_file(), "index.db was not created"

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT module, do_not_rebuild FROM chunks WHERE source='memory'"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "No memory chunks were indexed"
    # Every memory chunk that mentions use_apollo must be tagged.
    tagged = [r for r in rows if r[0] == "use_apollo"]
    assert tagged, (
        f"Expected at least one memory chunk with module='use_apollo'; "
        f"got rows: {rows}"
    )
    # do_not_rebuild must be 1 (use_apollo is in dnr_modules AND content
    # contains 'do not rebuild').
    assert all(r[1] == 1 for r in tagged), (
        f"Expected do_not_rebuild=1 for use_apollo chunks; got: {tagged}"
    )
