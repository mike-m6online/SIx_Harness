"""Tests for the one-command installer (install.py).

The venv/pip/ingest stages are exercised for real by the operator's
end-to-end pilot (they cost minutes); these tests pin the orchestration
contract: stage commands, ordering, flag plumbing, dry-run inertness,
and operator-error handling.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

KIT_ROOT = Path(__file__).resolve().parents[1]

_spec = importlib.util.spec_from_file_location(
    "kit_install", KIT_ROOT / "install.py"
)
kit_install = importlib.util.module_from_spec(_spec)
sys.modules["kit_install"] = kit_install
_spec.loader.exec_module(kit_install)


def _run_dry(capsys, *extra: str) -> str:
    rc = kit_install.main(["E:/definitely-not-a-real-project",
                           "--dry-run", *extra])
    assert rc == 0
    return capsys.readouterr().out


def test_dry_run_prints_all_stages_in_order(capsys):
    out = _run_dry(capsys)
    idx_install = out.index("venv (install kit)")
    idx_init = out.index("harness init")
    idx_cm_init = out.index("claude-mem init")
    idx_bulk = out.index("claude-mem bulk")
    assert idx_install < idx_init < idx_cm_init < idx_bulk


def test_dry_run_writes_nothing(tmp_path, capsys):
    before = sorted(p.name for p in KIT_ROOT.iterdir())
    _run_dry(capsys)
    after = sorted(p.name for p in KIT_ROOT.iterdir())
    assert before == after  # no .venv creation, nothing else either


def test_dry_run_bakes_venv_claude_mem_exe(capsys):
    out = _run_dry(capsys)
    paths = kit_install.venv_paths(KIT_ROOT)
    assert f"--claude-mem-exe {paths['claude_mem_exe']}" in out


def test_skip_ingest_omits_stage_three(capsys):
    out = _run_dry(capsys, "--skip-ingest")
    assert "claude-mem bulk" not in out
    assert "skipped (--skip-ingest)" in out


def test_optional_flags_are_plumbed(capsys):
    out = _run_dry(
        capsys, "--memory-dir", "X:/mem", "--with-decisions",
        "--settings", "shared",
    )
    assert "--memory-dir X:/mem" in out
    assert "--with-decisions" in out
    assert "--settings shared" in out


def test_missing_project_root_is_operator_error(capsys):
    rc = kit_install.main(["E:/definitely-not-a-real-project"])
    assert rc == 2
    assert "OPERATOR ERROR" in capsys.readouterr().err


def test_venv_paths_platform_layout():
    paths = kit_install.venv_paths(Path("Q:/kit"))
    assert paths["venv"] == Path("Q:/kit/.venv")
    if sys.platform.startswith("win"):
        assert paths["python"].name == "python.exe"
        assert paths["claude_mem_exe"].name == "claude-mem.exe"
    else:
        assert paths["python"].name == "python"
        assert paths["claude_mem_exe"].name == "claude-mem"
