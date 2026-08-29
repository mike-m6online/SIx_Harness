"""The CLI must force UTF-8 stdio regardless of the platform code page.

Claude Code invokes every hook as `claude-mem.exe <cmd> --stdin` with
UTF-8 JSON piped to stdin and the rendered block read from stdout.
Windows Python defaults PIPED (non-console) streams to the ANSI code
page (cp1252), which caused two live hook deaths:

  (a) session_start crashed with UnicodeEncodeError('charmap') the
      first time the curated render contained a non-cp1252 char (the
      '→' in the ground-truth-hierarchy invariant title) -- the
      render was never injected into that session;
  (b) tool_use mis-decoded UTF-8 stdin (the '⏯' resume-anchor
      emoji's 0x8f byte is undefined in cp1252) into mojibake + a lone
      surrogate that exploded later at encode time.

Importing claude_mem.cli must therefore reconfigure stdin/stdout/stderr
to UTF-8 before any hook I/O. These tests run the real interpreter with
piped stdio and PYTHONIOENCODING/PYTHONUTF8 stripped, so on Windows
they exercise the exact cp1252 default the hooks see.
"""
import os
import subprocess
import sys


def _clean_env() -> dict:
    env = dict(os.environ)
    for key in ("PYTHONIOENCODING", "PYTHONUTF8"):
        env.pop(key, None)
    return env


def test_cli_import_forces_utf8_stdio_on_pipes():
    code = (
        "import claude_mem.cli, sys;"
        "print(sys.stdout.encoding.lower(), sys.stdin.encoding.lower());"
        "sys.stdout.write('\\u2192\\n')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, env=_clean_env(), timeout=120,
    )
    assert proc.returncode == 0, proc.stderr.decode("utf-8", "replace")
    out = proc.stdout.decode("utf-8", "replace")
    assert "utf-8 utf-8" in out
    assert "→" in out


def test_stdin_json_with_utf8_content_decodes_correctly():
    # The comparison MUST happen inside the child: cp1252 mojibake
    # round-trips byte-identically through a cp1252 stdout, so checking
    # the parent-side bytes would pass vacuously without the fix.
    code = (
        "import claude_mem.cli, sys, json;"
        "data = json.loads(sys.stdin.read());"
        "sys.exit(0 if data['t'] == '\\u2192 \\u23ef' else 3)"
    )
    payload = '{"t": "→ ⏯"}'.encode("utf-8")
    proc = subprocess.run(
        [sys.executable, "-c", code],
        input=payload, capture_output=True, env=_clean_env(), timeout=120,
    )
    assert proc.returncode == 0, (
        f"exit {proc.returncode} (3 = stdin mis-decoded): "
        + proc.stderr.decode("utf-8", "replace")
    )
