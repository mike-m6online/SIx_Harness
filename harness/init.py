"""``harness init`` -- bootstrap the SIx Harness onto a project.

This module is the kit's console entry point (``harness init ...``). It
reproduces, for a NEW project, the wiring the live origin project
carries by hand:

* the Claude Code per-project memory directory
  (``~/.claude/projects/<slug>/memory/``) seeded with a ``MEMORY.md``
  index from the kit template,
* the append-only arc ledger at ``.superpowers/sdd/progress.md``,
* a ``CLAUDE.md`` constitution from the kit template -- ONLY when the
  project has none (an existing CLAUDE.md is never overwritten),
* hook entries MERGED into the project's Claude Code settings JSON
  (``.claude/settings.local.json``, or ``settings.json`` with
  ``--settings shared``) mirroring the origin project's production wiring:

  ============== ======================================= =============
  Event          Command                                 Matcher
  ============== ======================================= =============
  SessionStart   hooks/memory_decay.py (kit)             ``""``
  SessionStart   hooks/memory_health.py (kit)            ``""``
  SessionStart   hooks/gen_decisions_state.py (optional, ``""``
                 only with ``--with-decisions``)
  SessionStart   claude-mem session-start --stdin        ``""``
  UserPromptSub. claude-mem prompt-submit --stdin        ``""``
  PreToolUse     claude-mem tool-use --stdin             Edit|Write|Bash|Agent|NotebookEdit
  PostToolUse    claude-mem tool-use-post --stdin        Edit|Write|Bash|Agent
  SessionEnd     claude-mem session-end --stdin          ``""``
  SessionEnd     claude-mem capture-extract --stdin      ``""``
  SessionEnd     claude-mem capture-synthesize           ``""``
  ============== ======================================= =============

Merge contract (never clobber):

* Every existing settings key (``permissions``, custom hooks, anything
  else) is preserved verbatim.
* A hook the kit would add is SKIPPED (and reported) when an equivalent
  hook is already wired for that event -- equivalence is judged by
  semantic identity (same kit script filename, or same claude-mem
  subcommand, regardless of the executable path), so re-runs are
  idempotent and a project already carrying its own claude-mem
  installation is never double-wired.
* An unparseable settings file aborts the whole run BEFORE any write.

``.claude-mem/`` is deliberately NOT created here -- ``claude-mem init``
owns that directory's layout.

``--dry-run`` prints every action that would be taken and writes nothing.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------
# Kit layout (resolved relative to this file so the baked hook paths always
# point at THIS kit checkout, wherever it lives).
# --------------------------------------------------------------------------

KIT_ROOT: Path = Path(__file__).resolve().parent.parent
TEMPLATES_DIR: Path = KIT_ROOT / "templates"
HOOKS_DIR: Path = KIT_ROOT / "hooks"

CLAUDE_MD_TEMPLATE = "CLAUDE.md.template"
MEMORY_MD_TEMPLATE = "MEMORY.md.template"
LEDGER_TEMPLATE = "ledger_progress.md.template"

SETTINGS_FILENAMES: Dict[str, str] = {
    "local": "settings.local.json",
    "shared": "settings.json",
}
LEDGER_RELPATH: Path = Path(".superpowers") / "sdd" / "progress.md"

# Kit hook scripts wired at SessionStart (shipped in <kit>/hooks/ by the
# hooks part of this kit). gen_decisions_state is opt-in: it renders a
# decisions/threads digest out of the claude-mem index, so it is only
# useful once `claude-mem init` + `claude-mem bulk` have populated one.
DECAY_SCRIPT = "memory_decay.py"
HEALTH_SCRIPT = "memory_health.py"
DECISIONS_SCRIPT = "gen_decisions_state.py"
KIT_SCRIPT_NAMES: Tuple[str, ...] = (DECAY_SCRIPT, HEALTH_SCRIPT, DECISIONS_SCRIPT)

# THE canonical claude-mem hook set (events, subcommands, matchers,
# timeouts, flag shapes) lives in the claude_mem package itself so it is
# never duplicated across registrars; this module contributes only the
# preserve-existing merge semantics. See claude_mem/cli.py.
from claude_mem.cli import SETTINGS_HOOK_TABLE

# claude-mem hook subcommands the kit wires, derived from the canonical
# table. Order matters only for documentation; identity matching uses
# token boundaries so `tool-use` never swallows `tool-use-post`.
CLAUDE_MEM_SUBCOMMANDS: Tuple[str, ...] = tuple(
    spec["subcmd"]
    for cfg in SETTINGS_HOOK_TABLE.values()
    for spec in cfg["subcmds"]
)

# Matches any spelling of the claude-mem executable or module:
# `claude-mem`, `claude-mem.exe`, `claude_mem.cli`, `-m claude_mem.cli`.
_CLAUDE_MEM_MARKER = re.compile(r"claude[-_]mem", re.IGNORECASE)

# Event order used when reporting; mirrors the reference settings file.
HOOK_EVENT_ORDER: Tuple[str, ...] = (
    "SessionStart",
    "UserPromptSubmit",
    "PreToolUse",
    "PostToolUse",
    "SessionEnd",
)


# --------------------------------------------------------------------------
# Small pure helpers (unit-tested directly).
# --------------------------------------------------------------------------


def project_slug(absolute_path: str) -> str:
    """Compute the Claude Code project slug for an absolute project path.

    Claude Code derives the per-project directory name by lowercasing the
    absolute path and replacing the drive colon and every path separator
    with ``-``:

    * ``d:\\myproject``        -> ``d--myproject``
    * ``D:\\Myproject``        -> ``d--myproject`` (case-insensitive)
    * ``/home/someuser/proj``  -> ``-home-someuser-proj``

    The caller is responsible for passing an ABSOLUTE, already-resolved
    path string; this function performs no filesystem access.
    """
    slug = absolute_path.lower()
    for char in (":", "\\", "/"):
        slug = slug.replace(char, "-")
    return slug


def default_memory_dir(project_root: Path) -> Path:
    """Default Claude Code memory dir for ``project_root``.

    ``<home>/.claude/projects/<slug>/memory`` with the home directory
    derived from the running user (``Path.home()``) -- never hardcoded.
    """
    slug = project_slug(str(project_root))
    return Path.home() / ".claude" / "projects" / slug / "memory"


def _quote(path_text: str) -> str:
    """Quote a command path iff it contains whitespace (JSON-safe)."""
    if any(ch.isspace() for ch in path_text):
        return f'"{path_text}"'
    return path_text


def _cmd_path(path: Path) -> str:
    """Render a path for use inside a hook command string.

    Forward slashes (the style the origin project's settings file uses) work in
    every shell Claude Code spawns on Windows and avoid JSON backslash
    noise; the result is quoted when it contains spaces.
    """
    return _quote(path.as_posix())


def resolve_claude_mem_command(
    explicit_exe: Optional[str], python_exe: str
) -> Tuple[str, str]:
    """Resolve the claude-mem invocation prefix for hook commands.

    Resolution order (first hit wins):

    1. ``--claude-mem-exe`` flag -- used verbatim (absolute path
       recommended; the origin project's live wiring bakes the absolute .exe path
       because the Scripts dir is not on bash PATH).
    2. ``shutil.which("claude-mem")`` -- the executable found on PATH,
       baked as an absolute path so the hooks survive PATH changes.
    3. Module fallback: ``<python> -X utf8 -m claude_mem.cli`` -- works
       whenever ``claude_mem`` is importable by ``<python>`` even without
       a console script.

    Returns ``(prefix, how)`` where ``prefix`` is the command prefix to
    prepend to a subcommand and ``how`` describes the resolution for the
    summary report.
    """
    if explicit_exe:
        return _cmd_path(Path(explicit_exe)), "explicit --claude-mem-exe"
    found = shutil.which("claude-mem")
    if found:
        return _cmd_path(Path(found)), "found on PATH"
    return (
        f"{_quote(python_exe)} -X utf8 -m claude_mem.cli",
        "module fallback (<python> -X utf8 -m claude_mem.cli)",
    )


def command_identity(command: str) -> Optional[Tuple[str, str]]:
    """Semantic identity of a hook command string, or ``None``.

    Two hook commands are "the same hook" when they run the same kit
    script (by filename) or the same claude-mem subcommand -- regardless
    of interpreter, executable path, or argument spelling. This is what
    makes re-runs idempotent AND keeps ``harness init`` from double-wiring
    a project that already carries its own claude-mem installation.
    """
    for script in KIT_SCRIPT_NAMES:
        if script in command:
            return ("script", script)
    if _CLAUDE_MEM_MARKER.search(command):
        for sub in CLAUDE_MEM_SUBCOMMANDS:
            # Token-boundary match: `tool-use` must not match inside
            # `tool-use-post`, and `session-start` not inside a path.
            if re.search(rf"(?<![\w-]){re.escape(sub)}(?![\w-])", command):
                return ("claude-mem", sub)
    return None


# --------------------------------------------------------------------------
# Hook plan.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class HookCommand:
    """One hook entry the kit wants present in the settings file."""

    event: str
    matcher: str
    command: str
    timeout: int
    identity: Tuple[str, str]
    status_message: Optional[str] = None

    def to_settings_obj(self) -> Dict[str, Any]:
        """Render as the dict Claude Code expects inside a hooks group."""
        obj: Dict[str, Any] = {
            "type": "command",
            "command": self.command,
            "timeout": self.timeout,
        }
        if self.status_message is not None:
            obj["statusMessage"] = self.status_message
        return obj


def build_hook_plan(
    project_root: Path,
    memory_dir: Path,
    claude_mem_prefix: str,
    python_exe: str,
    with_decisions: bool,
) -> List[HookCommand]:
    """Build the full ordered list of hook entries the kit wires.

    Mirrors the origin project's production wiring, with the project-specific
    ``gen_project_state`` slot left to the project (see the
    ``templates/gen_project_state_stub.py`` extension point) and
    ``gen_decisions_state`` opt-in via ``--with-decisions``.
    """
    py = _quote(python_exe)
    proj = _cmd_path(project_root)
    mem = _cmd_path(memory_dir)
    decay = _cmd_path(HOOKS_DIR / DECAY_SCRIPT)
    health = _cmd_path(HOOKS_DIR / HEALTH_SCRIPT)
    decisions = _cmd_path(HOOKS_DIR / DECISIONS_SCRIPT)

    # The kit hook scripts share a UNIFORM interface: every one REQUIRES
    # both --project-root and --memory-dir (each script documents which of
    # the pair it actually consumes); harness init bakes concrete absolute
    # paths for both into every installed command.
    plan: List[HookCommand] = [
        HookCommand(
            event="SessionStart",
            matcher="",
            command=(
                f"{py} -X utf8 {decay} --project-root {proj} "
                f"--memory-dir {mem} --print"
            ),
            timeout=8000,
            identity=("script", DECAY_SCRIPT),
            status_message="Decaying MEMORY.md...",
        ),
        HookCommand(
            event="SessionStart",
            matcher="",
            command=(
                f"{py} -X utf8 {health} --project-root {proj} --memory-dir {mem}"
            ),
            timeout=8000,
            identity=("script", HEALTH_SCRIPT),
            status_message="Checking memory health...",
        ),
    ]
    if with_decisions:
        plan.append(
            HookCommand(
                event="SessionStart",
                matcher="",
                command=(
                    f"{py} -X utf8 {decisions} --project-root {proj} "
                    f"--memory-dir {mem} --print"
                ),
                timeout=10000,
                identity=("script", DECISIONS_SCRIPT),
                status_message="Loading DECISIONS_STATE...",
            )
        )
    # claude-mem entries are rendered from the package's own canonical
    # table (claude_mem.cli.SETTINGS_HOOK_TABLE): the hook SET is
    # single-sourced there, while the preserve-existing merge semantics
    # stay here in merge_hook_plan.
    for event, cfg in SETTINGS_HOOK_TABLE.items():
        for spec in cfg["subcmds"]:
            command = f"{claude_mem_prefix} {spec['subcmd']}"
            if spec.get("stdin", True):
                command += " --stdin"
            if spec.get("project_root", False):
                command += f" --project-root {proj}"
            plan.append(
                HookCommand(
                    event=event,
                    matcher=cfg.get("matcher", ""),
                    command=command,
                    timeout=spec["timeout"],
                    identity=("claude-mem", spec["subcmd"]),
                )
            )
    return plan


def merge_hook_plan(
    settings: Dict[str, Any], plan: Sequence[HookCommand]
) -> Tuple[List[HookCommand], List[Tuple[HookCommand, str]]]:
    """Merge ``plan`` into ``settings`` in place. Returns (added, skipped).

    * Existing keys and hook groups are preserved verbatim.
    * A planned command is skipped when its exact command string OR its
      semantic identity is already wired anywhere under the same event
      (any matcher group) -- this is what makes re-runs idempotent and
      protects projects with pre-existing claude-mem wiring.
    * New commands are appended to the first existing group with the same
      matcher, else a new ``{"matcher": ..., "hooks": [...]}`` group is
      appended -- never inserted into someone else's group.
    """
    hooks_cfg = settings.setdefault("hooks", {})
    added: List[HookCommand] = []
    skipped: List[Tuple[HookCommand, str]] = []

    events = []
    for hc in plan:
        if hc.event not in events:
            events.append(hc.event)

    for event in events:
        raw = hooks_cfg.setdefault(event, [])
        if not isinstance(raw, list):
            # Malformed but user-owned: leave it alone, skip our additions.
            for hc in plan:
                if hc.event == event:
                    skipped.append(
                        (hc, f"existing '{event}' hooks value is not a list; left untouched")
                    )
            continue

        existing_identities = set()
        existing_commands = set()
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            for hook in entry.get("hooks", []) or []:
                if not isinstance(hook, dict):
                    continue
                cmd = hook.get("command", "")
                if not isinstance(cmd, str):
                    continue
                existing_commands.add(cmd)
                ident = command_identity(cmd)
                if ident is not None:
                    existing_identities.add(ident)

        for hc in plan:
            if hc.event != event:
                continue
            if hc.command in existing_commands:
                skipped.append((hc, "exact command already wired"))
                continue
            if hc.identity in existing_identities:
                skipped.append(
                    (hc, f"equivalent hook already wired ({hc.identity[0]}:{hc.identity[1]})")
                )
                continue
            group: Optional[Dict[str, Any]] = None
            for entry in raw:
                if isinstance(entry, dict) and entry.get("matcher", "") == hc.matcher:
                    group = entry
                    break
            if group is None:
                group = {"matcher": hc.matcher, "hooks": []}
                raw.append(group)
            group_hooks = group.setdefault("hooks", [])
            if isinstance(group_hooks, list):
                group_hooks.append(hc.to_settings_obj())
            else:  # user put a non-list there; do not clobber it
                skipped.append(
                    (hc, "matcher group 'hooks' value is not a list; left untouched")
                )
                continue
            added.append(hc)
            existing_commands.add(hc.command)
            existing_identities.add(hc.identity)

    return added, skipped


# --------------------------------------------------------------------------
# Filesystem actions with dry-run support.
# --------------------------------------------------------------------------


@dataclass
class Report:
    """Accumulates the summary the operator sees at the end."""

    dry_run: bool
    created: List[str] = field(default_factory=list)
    merged: List[str] = field(default_factory=list)
    skipped: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def _prefix(self) -> str:
        return "[dry-run] would create " if self.dry_run else "created "

    def note_created(self, what: str) -> None:
        self.created.append(self._prefix() + what)

    def note_merged(self, what: str) -> None:
        prefix = "[dry-run] would add " if self.dry_run else "added "
        self.merged.append(prefix + what)

    def note_skipped(self, what: str) -> None:
        self.skipped.append(what)

    def note_warning(self, what: str) -> None:
        self.warnings.append(what)


def install_template_file(
    dest: Path, template: Path, label: str, report: Report
) -> None:
    """Copy ``template`` to ``dest`` unless ``dest`` exists (never overwrite)."""
    if dest.exists():
        report.note_skipped(f"{label}: {dest} already exists (never overwritten)")
        return
    content = template.read_text(encoding="utf-8")
    report.note_created(f"{label}: {dest} (from templates/{template.name})")
    if report.dry_run:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8", newline="\n")


def load_settings(settings_path: Path) -> Dict[str, Any]:
    """Load the settings JSON, or ``{}`` when the file does not exist.

    Raises ``ValueError`` (with a path-bearing message) when the file
    exists but cannot be parsed or is not a JSON object -- the caller
    aborts before ANY write in that case, so a hand-edited-but-broken
    settings file is never clobbered.
    """
    if not settings_path.exists():
        return {}
    text = settings_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"cannot parse {settings_path}: {exc}. Fix the JSON by hand and "
            "re-run; harness init never overwrites an unparseable settings file."
        ) from exc
    if not isinstance(data, dict):
        raise ValueError(
            f"{settings_path} does not contain a JSON object; refusing to touch it."
        )
    return data


def write_settings(settings_path: Path, settings: Dict[str, Any]) -> None:
    """Pretty-print ``settings`` to ``settings_path`` (UTF-8, 2-space indent)."""
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


# --------------------------------------------------------------------------
# The init command.
# --------------------------------------------------------------------------


def _check_kit_layout(report: Report, with_decisions: bool) -> Optional[str]:
    """Verify templates exist (fatal) and hook scripts exist (warning).

    Returns an error message when a required template is missing, else
    ``None``. Hook-script absence is only a warning because the settings
    entries are still correct once the kit's hooks land -- but the
    operator must know the SessionStart hooks will no-op until then.
    """
    missing = [
        name
        for name in (CLAUDE_MD_TEMPLATE, MEMORY_MD_TEMPLATE, LEDGER_TEMPLATE)
        if not (TEMPLATES_DIR / name).is_file()
    ]
    if missing:
        return (
            f"kit templates missing under {TEMPLATES_DIR}: {', '.join(missing)} "
            "(broken kit checkout?)"
        )
    wanted_scripts = [DECAY_SCRIPT, HEALTH_SCRIPT]
    if with_decisions:
        wanted_scripts.append(DECISIONS_SCRIPT)
    for script in wanted_scripts:
        if not (HOOKS_DIR / script).is_file():
            report.note_warning(
                f"kit hook script not present yet: {HOOKS_DIR / script} -- the "
                "settings entry is wired but will no-op until the kit's hooks/ "
                "directory ships it"
            )
    return None


def run_init(args: argparse.Namespace) -> int:
    """Execute ``harness init``. Returns a process exit code."""
    project_root = Path(args.project_root).expanduser()
    if not project_root.is_dir():
        print(f"ERROR: --project-root {project_root} is not an existing directory")
        return 1
    project_root = project_root.resolve()

    memory_dir = (
        Path(args.memory_dir).expanduser().resolve()
        if args.memory_dir
        else default_memory_dir(project_root)
    )
    python_exe = args.python or Path(sys.executable).as_posix()
    report = Report(dry_run=args.dry_run)

    layout_error = _check_kit_layout(report, args.with_decisions)
    if layout_error:
        print(f"ERROR: {layout_error}")
        return 1

    if args.claude_mem_exe and not Path(args.claude_mem_exe).exists():
        report.note_warning(
            f"--claude-mem-exe {args.claude_mem_exe} does not exist (baking it "
            "into the hooks anyway; verify the path)"
        )

    claude_mem_prefix, claude_mem_how = resolve_claude_mem_command(
        args.claude_mem_exe, python_exe
    )

    # ---- settings: load FIRST so a broken file aborts before any write.
    settings_path = (
        project_root / ".claude" / SETTINGS_FILENAMES[args.settings]
    )
    try:
        settings = load_settings(settings_path)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1
    if "hooks" in settings and not isinstance(settings["hooks"], dict):
        print(
            f"ERROR: {settings_path} has a non-object 'hooks' value; refusing "
            "to touch it. Fix the file by hand and re-run."
        )
        return 1

    # ---- memory skeleton + ledger + CLAUDE.md (create-if-absent only).
    if memory_dir.exists():
        report.note_skipped(f"memory dir: {memory_dir} already exists")
    else:
        report.note_created(f"memory dir: {memory_dir}")
        if not args.dry_run:
            memory_dir.mkdir(parents=True, exist_ok=True)
    install_template_file(
        memory_dir / "MEMORY.md", TEMPLATES_DIR / MEMORY_MD_TEMPLATE,
        "MEMORY.md", report,
    )
    install_template_file(
        project_root / LEDGER_RELPATH, TEMPLATES_DIR / LEDGER_TEMPLATE,
        "arc ledger", report,
    )
    install_template_file(
        project_root / "CLAUDE.md", TEMPLATES_DIR / CLAUDE_MD_TEMPLATE,
        "CLAUDE.md", report,
    )
    # .claude-mem/ is claude-mem's own to create (claude-mem init).

    # ---- hooks merge.
    plan = build_hook_plan(
        project_root=project_root,
        memory_dir=memory_dir,
        claude_mem_prefix=claude_mem_prefix,
        python_exe=python_exe,
        with_decisions=args.with_decisions,
    )
    merge_target = copy.deepcopy(settings) if args.dry_run else settings
    added, skipped = merge_hook_plan(merge_target, plan)
    for hc in added:
        report.note_merged(f"{hc.event}: {hc.command}")
    for hc, why in skipped:
        report.note_skipped(f"{hc.event} hook ({hc.identity[1]}): {why}")
    if added:
        verb = "[dry-run] would write" if args.dry_run else "wrote"
        report.merged.append(f"{verb} {settings_path}")
        if not args.dry_run:
            write_settings(settings_path, settings)
    else:
        report.note_skipped(
            f"settings: no hook changes needed in {settings_path}"
        )

    _print_summary(
        report=report,
        project_root=project_root,
        memory_dir=memory_dir,
        settings_path=settings_path,
        claude_mem_prefix=claude_mem_prefix,
        claude_mem_how=claude_mem_how,
        with_decisions=args.with_decisions,
    )
    return 0


def _print_summary(
    report: Report,
    project_root: Path,
    memory_dir: Path,
    settings_path: Path,
    claude_mem_prefix: str,
    claude_mem_how: str,
    with_decisions: bool,
) -> None:
    """Print the operator-facing summary block."""
    mode = " (DRY RUN -- nothing was written)" if report.dry_run else ""
    proj = project_root.as_posix()
    print(f"== harness init summary{mode} ==")
    print(f"  project root : {proj}")
    print(f"  memory dir   : {memory_dir.as_posix()}")
    print(f"  settings     : {settings_path.as_posix()}")
    print(f"  claude-mem   : {claude_mem_prefix}  [{claude_mem_how}]")
    print(f"  kit hooks dir: {HOOKS_DIR.as_posix()}")

    def _section(title: str, lines: List[str]) -> None:
        print(f"\n{title}:")
        if lines:
            for line in lines:
                print(f"  - {line}")
        else:
            print("  (none)")

    _section("Created", report.created)
    _section("Hook entries merged", report.merged)
    _section("Skipped (already present / unchanged)", report.skipped)
    if report.warnings:
        _section("WARNINGS", report.warnings)

    print("\nNext steps:")
    print(
        "  1. Install the kit so `claude-mem` resolves: in a venv, "
        f"`pip install -e {KIT_ROOT.as_posix()}` -- or `pipx install "
        f"{KIT_ROOT.as_posix()}`."
    )
    print(
        f"  2. Bootstrap the memory index: `claude-mem init --project-root {proj}` "
        f"(creates .claude-mem/), then `claude-mem bulk --project-root {proj}` "
        "(the canonical ingest: docs + memory + sessions + git)."
    )
    print(
        "  3. Fill in every <FILL-IN: ...> slot in CLAUDE.md; write your "
        "PRINCIPLES.yaml and the first invariant_*.md files in the memory dir "
        f"(examples: {TEMPLATES_DIR.as_posix()}/memory_file_examples/)."
    )
    print(
        "  4. Optional per-project SessionStart state hook: copy "
        f"{TEMPLATES_DIR.as_posix()}/gen_project_state_stub.py into your "
        "project's scripts/, adapt it, and wire it as a SessionStart hook."
    )
    if not with_decisions:
        print(
            "  5. Once the claude-mem index is populated, re-run with "
            "--with-decisions to add the gen_decisions_state SessionStart hook "
            "(it renders a digest FROM the index, so it is useless before "
            "ingestion)."
        )


# --------------------------------------------------------------------------
# CLI plumbing.
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the ``harness`` argument parser (subcommand: ``init``)."""
    parser = argparse.ArgumentParser(
        prog="harness",
        description=(
            "SIx Harness kit CLI -- bootstrap the memory stack, hooks, and "
            "conventions onto a project."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    init_p = sub.add_parser(
        "init",
        help="wire a project: memory skeleton, arc ledger, CLAUDE.md, hook entries",
        description=(
            "Create the memory skeleton / ledger / CLAUDE.md (never "
            "overwriting existing files) and MERGE the harness hook entries "
            "into the project's Claude Code settings JSON."
        ),
    )
    init_p.add_argument(
        "--project-root",
        required=True,
        help="the project directory to wire (must exist)",
    )
    init_p.add_argument(
        "--memory-dir",
        default=None,
        help=(
            "memory directory override (default: "
            "<home>/.claude/projects/<slug>/memory, where <slug> is the "
            "Claude Code project slug, e.g. d:\\myproject -> d--myproject)"
        ),
    )
    init_p.add_argument(
        "--settings",
        choices=sorted(SETTINGS_FILENAMES),
        default="local",
        help=(
            "which settings file to merge hooks into: 'local' -> "
            ".claude/settings.local.json (default), 'shared' -> "
            ".claude/settings.json"
        ),
    )
    init_p.add_argument(
        "--claude-mem-exe",
        default=None,
        help=(
            "absolute path to the claude-mem executable to bake into hook "
            "commands (default: shutil.which('claude-mem'), else the "
            "'<python> -X utf8 -m claude_mem.cli' module form)"
        ),
    )
    init_p.add_argument(
        "--python",
        default=None,
        help=(
            "python executable to bake into hook commands (default: the "
            "interpreter running harness init)"
        ),
    )
    init_p.add_argument(
        "--with-decisions",
        action="store_true",
        help=(
            "also wire the gen_decisions_state SessionStart hook (opt-in: it "
            "needs a populated claude-mem index to be useful)"
        ),
    )
    init_p.add_argument(
        "--dry-run",
        action="store_true",
        help="print every action without writing anything",
    )
    init_p.set_defaults(func=run_init)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Console entry point: ``harness init ...``. Returns an exit code."""
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
