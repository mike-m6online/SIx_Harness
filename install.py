#!/usr/bin/env python3
"""One-command installer: wire the harness onto a project, end to end.

    python C:\\six-harness\\install.py <project-root> [options]

does everything the README's quick start lists, in order:

  1. VENV    — create the kit's own virtualenv at <kit>/.venv (once) and
               ``pip install -e`` the kit into it. The venv is deliberate:
               the kit ships a ``claude_mem`` package under the same import
               name as a pre-existing user-site claude-mem installation
               (the origin project's, which is live and load-bearing) — a
               --user install would shadow it. The venv isolates the kit
               completely; every hook command bakes the venv's absolute
               executable paths, so nothing needs to be on PATH and no
               existing installation is disturbed.
  2. INIT    — run ``harness init`` against the project (idempotent,
               merge-only; never overwrites CLAUDE.md/MEMORY.md; skips
               anything already wired).
  3. INGEST  — bootstrap the project's memory index:
               ``claude-mem init`` + ``claude-mem bulk`` (docs + memory +
               sessions + git history). Skippable with --skip-ingest;
               bulk can take minutes on a large project.

Idempotent end to end: re-running re-uses the venv, re-installs cheaply,
merges nothing new, and refreshes the index incrementally.

Pure stdlib — this script must run before anything is installed.

Exit codes: 0 = success; 1 = a stage failed (stderr says which);
2 = operator error (bad arguments/paths).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence

KIT_ROOT = Path(__file__).resolve().parent


def venv_paths(kit_root: Path) -> dict:
    """Absolute paths inside the kit venv (Windows and POSIX layouts)."""
    venv = kit_root / ".venv"
    if sys.platform.startswith("win"):
        scripts = venv / "Scripts"
        return {
            "venv": venv,
            "python": scripts / "python.exe",
            "claude_mem_exe": scripts / "claude-mem.exe",
        }
    bindir = venv / "bin"
    return {
        "venv": venv,
        "python": bindir / "python",
        "claude_mem_exe": bindir / "claude-mem",
    }


def run_stage(
    label: str, cmd: Sequence[str], dry_run: bool,
    timeout_s: Optional[int] = None,
) -> bool:
    """Run one installer stage, streaming output. Returns success."""
    printable = " ".join(str(c) for c in cmd)
    if dry_run:
        print(f"[dry-run] {label}: {printable}")
        return True
    print(f"== {label} ==")
    print(f"   {printable}")
    try:
        completed = subprocess.run(list(cmd), timeout=timeout_s)
    except FileNotFoundError as exc:
        print(f"FAILED ({label}): {exc}", file=sys.stderr)
        return False
    except subprocess.TimeoutExpired:
        print(f"FAILED ({label}): timed out after {timeout_s}s", file=sys.stderr)
        return False
    if completed.returncode != 0:
        print(
            f"FAILED ({label}): exit code {completed.returncode}",
            file=sys.stderr,
        )
        return False
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="install.py",
        description=(
            "One-command harness installer: kit venv + harness init + "
            "claude-mem index bootstrap for one project."
        ),
    )
    parser.add_argument(
        "project_root",
        help="the project to wire (e.g. E:\\myproject)",
    )
    parser.add_argument(
        "--memory-dir", default=None,
        help="override the memory dir (default: the Claude Code per-project "
             "slug dir, computed by harness init)",
    )
    parser.add_argument(
        "--settings", choices=("local", "shared"), default="local",
        help="which settings file harness init merges into (default local)",
    )
    parser.add_argument(
        "--skip-ingest", action="store_true",
        help="skip the claude-mem init/bulk index bootstrap (stage 3)",
    )
    parser.add_argument(
        "--with-decisions", action="store_true",
        help="also wire the gen_decisions_state SessionStart hook "
             "(useful once the index is populated)",
    )
    parser.add_argument(
        "--recreate-venv", action="store_true",
        help="delete and rebuild the kit venv before installing",
    )
    parser.add_argument(
        "--python", dest="python_exe", default=sys.executable,
        help="base interpreter used to create the venv (default: this one)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="print every stage command without running anything",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    project_root = Path(args.project_root).expanduser()
    if not args.dry_run and not project_root.is_dir():
        print(
            f"OPERATOR ERROR: project root does not exist or is not a "
            f"directory: {project_root}",
            file=sys.stderr,
        )
        return 2

    paths = venv_paths(KIT_ROOT)
    venv: Path = paths["venv"]
    venv_python: Path = paths["python"]
    claude_mem_exe: Path = paths["claude_mem_exe"]

    # ---- stage 1: kit venv + editable install -----------------------------
    if args.recreate_venv and venv.exists() and not args.dry_run:
        import shutil

        shutil.rmtree(venv)
        print(f"removed existing venv {venv}")
    if venv_python.exists():
        print(f"== venv == reusing {venv}")
    else:
        if not run_stage(
            "venv (create)",
            [args.python_exe, "-m", "venv", str(venv)],
            args.dry_run,
            timeout_s=300,
        ):
            return 1
    if not run_stage(
        "venv (install kit)",
        [str(venv_python), "-X", "utf8", "-m", "pip", "install", "-q",
         "-e", str(KIT_ROOT)],
        args.dry_run,
        timeout_s=900,
    ):
        return 1

    # ---- stage 2: harness init --------------------------------------------
    init_cmd: List[str] = [
        str(venv_python), "-X", "utf8", "-m", "harness", "init",
        "--project-root", str(project_root),
        "--settings", args.settings,
        # Bake the venv's own claude-mem executable: absolute, PATH-free,
        # and guaranteed to be the kit's code rather than any pre-existing
        # user-site installation.
        "--claude-mem-exe", str(claude_mem_exe),
    ]
    if args.memory_dir:
        init_cmd += ["--memory-dir", args.memory_dir]
    if args.with_decisions:
        init_cmd += ["--with-decisions"]
    if not run_stage("harness init", init_cmd, args.dry_run):
        return 1

    # ---- stage 3: memory index bootstrap ----------------------------------
    if args.skip_ingest:
        print("== ingest == skipped (--skip-ingest)")
    else:
        if not run_stage(
            "claude-mem init",
            [str(claude_mem_exe), "init",
             "--project-root", str(project_root)],
            args.dry_run,
            timeout_s=120,
        ):
            return 1
        # bulk = the canonical ingest (docs + memory + sessions + git).
        # No timeout: large corpora legitimately take minutes.
        if not run_stage(
            "claude-mem bulk",
            [str(claude_mem_exe), "bulk",
             "--project-root", str(project_root)],
            args.dry_run,
        ):
            return 1

    print()
    print("== done ==")
    print(f"  project : {project_root}")
    print(f"  venv    : {venv}")
    print("  Open a Claude Code session in the project to see the "
          "SessionStart injection.")
    print("  Remaining by hand: fill the <FILL-IN> slots in CLAUDE.md, "
          "write PRINCIPLES/invariants (templates/memory_file_examples/), "
          "and adapt templates/gen_project_state_stub.py if the project "
          "wants a state hook.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
