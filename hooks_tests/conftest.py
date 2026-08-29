"""Shared fixtures for the hooks test suite.

Every ported hook is exercised the way Claude Code runs it: as a subprocess
(`python -X utf8 hooks/<script>.py <flags>`) with stdin/stdout captured.
Unit-level assertions import the hook modules directly (the repo root is
put on sys.path below).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOKS_DIR = REPO_ROOT / "hooks"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RunHook = Callable[..., subprocess.CompletedProcess]


@pytest.fixture
def run_hook() -> RunHook:
    """Run a hook script as a subprocess, mirroring the baked hook command
    (`python -X utf8 <script> <flags>`). Returns the CompletedProcess with
    text stdout/stderr decoded as UTF-8."""

    def _run(
        script: str,
        *args: object,
        stdin_text: str = "",
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-X", "utf8", str(HOOKS_DIR / script)]
            + [str(a) for a in args],
            input=stdin_text,
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(cwd if cwd is not None else REPO_ROOT),
            timeout=120,
        )

    return _run
