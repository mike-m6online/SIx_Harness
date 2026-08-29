#!/usr/bin/env python3
"""PreToolUse hook: blocks broad Python process kills.

Claude Code will block the tool call if this script outputs:
  {"decision": "block", "reason": "..."}

A broad "kill every python" command takes down the harness's OWN
background Python processes -- MCP servers, hook workers, watchers, and
any long-running experiment -- along with whatever the caller meant to
target. On the source project this destroyed the session's memory tooling
mid-run; the guard exists so the mistake is caught before execution.

Patterns blocked:
  - taskkill /f /im python.exe        (kills ALL python on Windows)
  - pkill python / pkill -f python    (kills ALL python on Unix)
  - killall python
  - Stop-Process -Name python         (kills ALL python via PowerShell)
  - kill -9 -1                        (kills every process the user can)

Safe alternatives (in the block message):
  - taskkill /PID <pid> /F            (Windows, one process by PID)
  - kill <pid> / Stop-Process -Id     (Unix / PowerShell, one PID)

Harness-kit deltas from the origin-project original:
  - The block message no longer names origin-project artifacts (the agent-army MCP
    server, scripts/restart_dashboard.sh, port 5555); it states the
    generic hazard and generic targeted-kill alternatives, in plain ASCII
    (the original's bullet characters are a cp1252 console hazard).
  - The guard also inspects the PowerShell tool (the original checked only
    Bash) and adds the PowerShell-native broad kill `Stop-Process -Name
    python` to the pattern list -- on a Windows harness the PowerShell
    tool is the primary shell, so a Bash-only guard leaves the exact
    hazard it exists for unguarded.
  - No path flags: the guard reads only the tool payload from stdin and
    touches no project files, so there is nothing to parameterize.
"""
from __future__ import annotations

import json
import re
import sys
from typing import Optional, Sequence

# Tool names whose commands are inspected. Both shell tools can issue
# process kills; every other tool is ignored.
GUARDED_TOOLS = ("Bash", "PowerShell")

DANGEROUS_PATTERNS = (
    # Windows broad kills
    r"taskkill\s+.*?/im\s+python",
    r"taskkill\s+.*?python.*?/f",
    # Unix broad kills
    r"pkill\s+python",
    r"pkill\s+-f\s+python",
    r"killall\s+python",
    # PowerShell-native broad kill
    r"stop-process\s+.*?-name\s+[\"']?python",
    # Very broad kill-all
    r"kill\s+-9\s+-1",
)

SAFE_MESSAGE = (
    "BLOCKED: This broad Python kill would take down the harness's own "
    "background Python processes (MCP servers, hook workers, running "
    "experiments) along with the target.\n\n"
    "Use targeted kills instead:\n"
    "  - taskkill /PID <specific_pid> /F   (Windows, one process by PID)\n"
    "  - Stop-Process -Id <specific_pid>   (PowerShell, one process by PID)\n"
    "  - kill <specific_pid>               (Unix, one process by PID)\n"
    "Find the PID first: netstat -ano | findstr <port>  (Windows)  or  "
    "ps aux | grep <name>  (Unix)"
)


def evaluate(hook_input: dict) -> Optional[str]:
    """Return the block reason when the payload carries a dangerous shell
    command, else None. Pure function -- the testable core of the guard."""
    if hook_input.get("tool_name") not in GUARDED_TOOLS:
        return None
    tool_input = hook_input.get("tool_input", {})
    if not isinstance(tool_input, dict):
        return None
    command = tool_input.get("command", "")
    if not isinstance(command, str):
        return None
    command_lower = command.lower()
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, command_lower):
            return SAFE_MESSAGE
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError, UnicodeDecodeError, ValueError):
        return 0
    if not isinstance(hook_input, dict):
        return 0
    reason = evaluate(hook_input)
    if reason is not None:
        print(json.dumps({
            "decision": "block",
            "reason": reason,
        }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
