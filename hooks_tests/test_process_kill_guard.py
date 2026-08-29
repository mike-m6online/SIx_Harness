"""Tests for hooks/process_kill_guard.py -- every dangerous pattern blocks,
targeted kills pass, non-shell tools are ignored, and malformed stdin never
fails the hook."""
from __future__ import annotations

import json

import pytest

from hooks import process_kill_guard as guard


def _payload(tool_name: str, command: str) -> dict:
    return {"tool_name": tool_name, "tool_input": {"command": command}}


@pytest.mark.parametrize("command", [
    "taskkill /f /im python.exe",
    "taskkill /IM python.exe /F",
    "pkill python",
    "pkill -f python",
    "killall python",
    "kill -9 -1",
])
def test_broad_kills_blocked_in_bash(command):
    reason = guard.evaluate(_payload("Bash", command))
    assert reason == guard.SAFE_MESSAGE


@pytest.mark.parametrize("command", [
    "Stop-Process -Name python -Force",
    "stop-process -name 'python'",
    "taskkill /f /im python.exe",
])
def test_broad_kills_blocked_in_powershell(command):
    reason = guard.evaluate(_payload("PowerShell", command))
    assert reason == guard.SAFE_MESSAGE


@pytest.mark.parametrize("command", [
    "taskkill /PID 1234 /F",
    "kill 1234",
    "Stop-Process -Id 1234",
    "python train.py --config x.yaml",
    "echo taskkill-free command",
])
def test_targeted_and_benign_commands_pass(command):
    assert guard.evaluate(_payload("Bash", command)) is None
    assert guard.evaluate(_payload("PowerShell", command)) is None


def test_non_shell_tools_ignored():
    assert guard.evaluate(_payload("Read", "pkill python")) is None
    assert guard.evaluate(_payload("Edit", "taskkill /f /im python.exe")) is None


def test_malformed_tool_input_ignored():
    assert guard.evaluate({"tool_name": "Bash", "tool_input": "pkill python"}) is None
    assert guard.evaluate({"tool_name": "Bash"}) is None
    assert guard.evaluate({}) is None


def test_subprocess_blocks_with_json_decision(run_hook):
    proc = run_hook(
        "process_kill_guard.py",
        stdin_text=json.dumps(_payload("Bash", "pkill -f python")),
    )
    assert proc.returncode == 0
    out = json.loads(proc.stdout)
    assert out["decision"] == "block"
    assert "BLOCKED" in out["reason"]
    # The generic message must carry no origin-project artifacts.
    for origin_token in ("agent-army", "Virgil", "restart_dashboard", "5555"):
        assert origin_token not in out["reason"]


def test_subprocess_silent_on_safe_command(run_hook):
    proc = run_hook(
        "process_kill_guard.py",
        stdin_text=json.dumps(_payload("Bash", "taskkill /PID 42 /F")),
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_subprocess_garbage_stdin_never_fails(run_hook):
    proc = run_hook("process_kill_guard.py", stdin_text="{{{ not json")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_subprocess_empty_stdin_never_fails(run_hook):
    proc = run_hook("process_kill_guard.py", stdin_text="")
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""
