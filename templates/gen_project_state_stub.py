#!/usr/bin/env python3
"""gen_project_state -- SessionStart hook: render '## PROJECT STATE'.

WHAT THIS IS
    A working, per-project extension-point stub from the SIx Harness kit.
    The origin project's real version (scripts/gen_project_state.py in
    its repo) parses its CUDA engine's config header and module
    sources at session start and renders auto-derived facts: every config
    flag's ON/DORMANT state, per-module `dispatch_wired` status, and the
    DORMANT list of built-but-not-enabled subsystems. Those facts are
    "tier 2" in the ground-truth hierarchy -- regenerated from the actual
    code each session, so agents TRUST them instead of re-grepping.

    Each project supplies its own version: copy this file into your
    project's scripts/, keep the CONTRACT below, and replace the generic
    sections with facts derived from YOUR code (parse your config, your
    feature-flag registry, your service manifest -- whatever "what is
    enabled right now" means here). The value of this hook is exactly
    proportional to how much of it is DERIVED FROM CODE rather than
    hand-written.

THE CONTRACT (keep all of it)
    * Pure stdlib -- it runs inside a SessionStart hook with a tight
      timeout (the origin project's wiring gives it 15s).
    * Prints a markdown block starting with '## PROJECT STATE' to stdout;
      Claude Code injects hook stdout into the session context.
    * ALWAYS exits 0. A state renderer that can crash the harness becomes
      the silent-death it exists to prevent -- degrade to a partial render,
      never raise out of main().
    * Accepts --project-root (where to derive from) and --print (the
      hook-convention flag; output goes to stdout either way -- the flag
      exists so the same script can later grow a write-to-file mode
      without rewiring the hook).

WIRING (after adapting it)
    Add to .claude/settings.local.json under hooks.SessionStart:
        {"type": "command",
         "command": "<python> -X utf8 <project>/scripts/gen_project_state.py --project-root <project> --print",
         "timeout": 15000,
         "statusMessage": "Loading PROJECT_STATE.md..."}
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


def _git(args: List[str], cwd: Path) -> Optional[str]:
    """Run a git command; return stripped stdout, or None on any failure.

    Failure is normal (not a git repo, git absent) -- the render degrades
    to '(unavailable)' rather than crashing the hook.
    """
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def render_state(project_root: Path) -> str:
    """Render the '## PROJECT STATE' markdown block.

    The generic version derives what it can from any repo: git branch,
    last commit, dirty-file count. ADAPT: replace/extend the AUTO-DERIVED
    FACTS section with facts parsed from your project's own sources of
    truth (config files, flag registries, deployment manifests). Keep each
    fact on one line and name the file it was derived from -- that is what
    lets a session trust the line without re-deriving it.
    """
    lines: List[str] = ["## PROJECT STATE"]
    lines.append(
        f"(auto-generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} "
        f"from {project_root.as_posix()} -- tier-2 ground truth: regenerated "
        "this session, trust it, do not re-derive)"
    )
    lines.append("")

    lines.append("### Repository")
    branch = _git(["rev-parse", "--abbrev-ref", "HEAD"], project_root)
    last_commit = _git(["log", "-1", "--format=%h %ad %s", "--date=short"], project_root)
    status = _git(["status", "--porcelain"], project_root)
    dirty = len([ln for ln in status.splitlines() if ln.strip()]) if status else 0
    lines.append(f"- branch: {branch or '(unavailable)'}")
    lines.append(f"- last commit: {last_commit or '(unavailable)'}")
    if status is None:
        lines.append("- working tree: (unavailable)")
    else:
        lines.append(f"- working tree: {dirty} modified/untracked file(s)")
    lines.append("")

    lines.append("### Auto-derived facts")
    lines.append(
        "- (ADAPT ME: parse your project's config / flag registry here and "
        "emit one line per fact, e.g. `- use_feature_x: ON "
        "(src/config.yaml)` -- the origin project's version renders every engine flag's "
        "ON/DORMANT state and the DORMANT list of built-but-not-enabled "
        "subsystems from its config header)"
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    """CLI: --project-root plus the hook-convention --print flag."""
    parser = argparse.ArgumentParser(
        description="Render '## PROJECT STATE' for SessionStart injection."
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="project directory to derive state from (default: cwd)",
    )
    parser.add_argument(
        "--print",
        dest="print_stdout",
        action="store_true",
        help=(
            "hook-convention flag: render to stdout (the default and only "
            "behavior of this stub; kept so a future write-to-file mode "
            "does not require rewiring the hook)"
        ),
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. ALWAYS returns 0 -- see THE CONTRACT."""
    try:
        args = build_arg_parser().parse_args(argv)
        project_root = Path(args.project_root).expanduser().resolve()
        print(render_state(project_root))
    except Exception as exc:  # noqa: BLE001 -- hook must never crash the turn
        print(f"## PROJECT STATE\n(render failed: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
