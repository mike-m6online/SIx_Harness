import tempfile
from pathlib import Path

from claude_mem.config import ProjectConfig
from claude_mem.cross_project import (
    cross_project_db_path, mirror_chunk_to_global, search_cross_project,
)
from claude_mem.ingest import Chunk, Ingester
from claude_mem.schema import init_db


class _ConstEmbedder:
    def embed(self, text: str) -> list[float]:
        return [0.1] * 1024


def test_cross_project_db_path_uses_home_default(monkeypatch, tmp_path):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setenv("CLAUDE_MEM_HOME", str(fake_home))
    p = cross_project_db_path()
    assert str(fake_home) in str(p)
    assert p.name == "cross-project-index.db"


def test_mirror_chunk_to_global_when_not_isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MEM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj_a"
    proj.mkdir()
    cfg = ProjectConfig(project_root=proj)
    cfg.write()
    init_db(cfg.db_path)
    # isolate_from_cross_project default is False
    chunk = Chunk(
        content="proj_a apollo chunk",
        source="doc", module="apollo",
        signal_weight=50, is_decision=True,
    )
    mirror_chunk_to_global(chunk, cfg)
    global_db = cross_project_db_path()
    assert global_db.is_file()
    import sqlite3
    conn = sqlite3.connect(global_db)
    try:
        rows = conn.execute(
            "SELECT origin_project, content FROM chunks"
        ).fetchall()
    finally:
        conn.close()
    assert len(rows) == 1
    assert rows[0][0] == "proj_a"
    assert "apollo" in rows[0][1]


def test_isolated_project_does_not_mirror(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MEM_HOME", str(tmp_path / "home"))
    proj = tmp_path / "proj_b"
    proj.mkdir()
    cfg = ProjectConfig(project_root=proj)
    cfg.values["isolate_from_cross_project"] = True
    cfg.write()
    init_db(cfg.db_path)
    chunk = Chunk(content="proj_b secret", source="doc")
    mirror_chunk_to_global(chunk, cfg)
    global_db = cross_project_db_path()
    if global_db.is_file():
        import sqlite3
        conn = sqlite3.connect(global_db)
        try:
            n = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        finally:
            conn.close()
        assert n == 0


def test_search_cross_project_returns_origin_tagged_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAUDE_MEM_HOME", str(tmp_path / "home"))
    # Seed project A
    proj_a = tmp_path / "proj_a"
    proj_a.mkdir()
    cfg_a = ProjectConfig(project_root=proj_a)
    cfg_a.write()
    init_db(cfg_a.db_path)
    chunk_a = Chunk(
        content="apollo loop master switch",
        source="doc", module="apollo",
        signal_weight=50, is_decision=True,
    )
    mirror_chunk_to_global(chunk_a, cfg_a)
    # Seed project B
    proj_b = tmp_path / "proj_b"
    proj_b.mkdir()
    cfg_b = ProjectConfig(project_root=proj_b)
    cfg_b.write()
    init_db(cfg_b.db_path)
    chunk_b = Chunk(
        content="apollo loop notes",
        source="doc", module="apollo",
        signal_weight=30, is_decision=True,
    )
    mirror_chunk_to_global(chunk_b, cfg_b)
    # Cross-project search hits both
    results = search_cross_project("apollo", _ConstEmbedder(), top_k=10)
    origins = {r.get("origin_project") for r in results}
    assert "proj_a" in origins
    assert "proj_b" in origins
