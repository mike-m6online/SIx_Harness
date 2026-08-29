"""Tests for ``harness init`` (harness/init.py).

Covers the behavior contract:

* skeleton creation (memory dir + MEMORY.md, arc ledger, CLAUDE.md,
  settings hooks) on a fresh project,
* idempotent re-runs (no duplicate hook entries, byte-identical settings),
* merge that preserves pre-existing settings content,
* CLAUDE.md never overwritten,
* Claude Code project-slug computation (d:\\myproject -> d--myproject),
* ``--dry-run`` writing nothing,
* semantic-identity skip of pre-existing claude-mem wiring,
* ``--settings shared`` targeting, broken-JSON abort, and
  ``--with-decisions`` opt-in.

All init runs pass an explicit ``--memory-dir`` under tmp_path so the
suite never touches the real ``~/.claude`` tree, and an explicit
``--claude-mem-exe`` so resolution is deterministic on any machine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import harness.init as hi


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def make_project(tmp_path: Path) -> Tuple[Path, Path, Path]:
    """Create a fresh fake project; return (project, memory_dir, fake_exe)."""
    project = tmp_path / "proj"
    project.mkdir()
    memory_dir = tmp_path / "memhome" / "memory"
    fake_exe = tmp_path / "bin" / "claude-mem.exe"
    fake_exe.parent.mkdir(parents=True, exist_ok=True)
    fake_exe.write_bytes(b"")  # exists so no missing-exe warning fires
    return project, memory_dir, fake_exe


def run_init(
    project: Path, memory_dir: Path, fake_exe: Path, *extra: str
) -> int:
    """Invoke the real console entry point with deterministic flags."""
    argv = [
        "init",
        "--project-root", str(project),
        "--memory-dir", str(memory_dir),
        "--claude-mem-exe", str(fake_exe),
        *extra,
    ]
    return hi.main(argv)


def read_settings(project: Path, name: str = "settings.local.json") -> Dict[str, Any]:
    """Parse the project's settings JSON."""
    return json.loads(
        (project / ".claude" / name).read_text(encoding="utf-8")
    )


def all_hook_commands(settings: Dict[str, Any]) -> List[Tuple[str, str]]:
    """Flatten settings hooks to a list of (event, command) pairs."""
    out: List[Tuple[str, str]] = []
    for event, entries in settings.get("hooks", {}).items():
        for entry in entries:
            for hook in entry.get("hooks", []):
                out.append((event, hook.get("command", "")))
    return out


# ---------------------------------------------------------------------------
# Slug + memory dir derivation.
# ---------------------------------------------------------------------------


def test_slug_known_good_mapping() -> None:
    """The known-good mapping: d:\\myproject -> d--myproject."""
    assert hi.project_slug("d:\\myproject") == "d--myproject"


def test_slug_is_lowercased_and_separator_normalized() -> None:
    assert hi.project_slug("D:\\Myproject") == "d--myproject"
    assert hi.project_slug("D:\\My Projects\\Alpha") == "d--my projects-alpha"
    assert hi.project_slug("/home/someuser/proj") == "-home-someuser-proj"


def test_default_memory_dir_under_user_home() -> None:
    expected = (
        Path.home() / ".claude" / "projects" / "d--myproject" / "memory"
    )
    assert hi.default_memory_dir(Path("d:\\myproject")) == expected


# ---------------------------------------------------------------------------
# Skeleton creation.
# ---------------------------------------------------------------------------


def test_init_creates_skeleton(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe) == 0

    # Memory skeleton from template.
    memory_md = memory_dir / "MEMORY.md"
    assert memory_md.is_file()
    memory_text = memory_md.read_text(encoding="utf-8")
    assert "## INVARIANTS" in memory_text
    assert "LATEST" in memory_text  # the resume-anchor section

    # Append-only ledger from template.
    ledger = project / ".superpowers" / "sdd" / "progress.md"
    assert ledger.is_file()
    assert "APPEND-ONLY" in ledger.read_text(encoding="utf-8")

    # CLAUDE.md constitution from template.
    claude_md = project / "CLAUDE.md"
    assert claude_md.is_file()
    claude_text = claude_md.read_text(encoding="utf-8")
    assert "NO SHORTCUTS" in claude_text
    assert "GROUND TRUTH HIERARCHY" in claude_text

    # .claude-mem/ is claude-mem's own to create -- init must NOT make it.
    assert not (project / ".claude-mem").exists()


def test_init_wires_reference_hook_set(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe) == 0
    settings = read_settings(project)

    commands = all_hook_commands(settings)
    joined = {event: [c for e, c in commands if e == event] for event, _ in commands}

    # SessionStart: decay + health + claude-mem session-start (in order).
    # The kit hook scripts share a uniform REQUIRED flag pair
    # (--project-root AND --memory-dir); both must be baked into each.
    session_start = joined["SessionStart"]
    assert "memory_decay.py" in session_start[0]
    assert "--print" in session_start[0]
    assert "memory_health.py" in session_start[1]
    for cmd in session_start[:2]:
        assert "--project-root" in cmd
        assert "--memory-dir" in cmd
    assert "session-start --stdin" in session_start[2]
    # gen_decisions_state is opt-in and must NOT be wired by default.
    assert not any("gen_decisions_state" in c for c in session_start)

    assert any("prompt-submit --stdin" in c for c in joined["UserPromptSubmit"])
    assert any("tool-use --stdin" in c for c in joined["PreToolUse"])
    assert any("tool-use-post --stdin" in c for c in joined["PostToolUse"])
    session_end = joined["SessionEnd"]
    assert any("session-end --stdin" in c for c in session_end)
    assert any("capture-extract --stdin" in c for c in session_end)
    assert any("capture-synthesize --project-root" in c for c in session_end)

    # Matchers mirror the origin project's production wiring.
    pre = settings["hooks"]["PreToolUse"][0]["matcher"]
    post = settings["hooks"]["PostToolUse"][0]["matcher"]
    assert pre == "Edit|Write|Bash|Agent|NotebookEdit"
    assert post == "Edit|Write|Bash|Agent"

    # The explicit exe was baked in, absolute, forward-slash style.
    assert any(fake_exe.as_posix() in c for _, c in commands)


def test_with_decisions_flag_adds_optional_hook(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe, "--with-decisions") == 0
    settings = read_settings(project)
    session_start = [c for e, c in all_hook_commands(settings) if e == "SessionStart"]
    decisions = [c for c in session_start if "gen_decisions_state.py" in c]
    assert len(decisions) == 1
    # Uniform kit-hook interface: both path flags baked, plus --print.
    assert "--project-root" in decisions[0]
    assert "--memory-dir" in decisions[0]
    assert "--print" in decisions[0]


# ---------------------------------------------------------------------------
# Idempotency + merge preservation.
# ---------------------------------------------------------------------------


def test_rerun_is_idempotent(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe) == 0
    settings_path = project / ".claude" / "settings.local.json"
    first = settings_path.read_text(encoding="utf-8")

    assert run_init(project, memory_dir, fake_exe) == 0
    second = settings_path.read_text(encoding="utf-8")
    assert first == second  # byte-identical: nothing re-added

    # And no identity appears twice anywhere.
    settings = read_settings(project)
    identities = [
        (event, hi.command_identity(cmd))
        for event, cmd in all_hook_commands(settings)
    ]
    assert len(identities) == len(set(identities))


def test_existing_settings_content_preserved(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    pre_existing = {
        "permissions": {"allow": ["Bash(ls:*)"], "deny": []},
        "env": {"MY_FLAG": "1"},
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": "python x:/own/state_render.py --print",
                            "timeout": 15000,
                        }
                    ],
                }
            ],
            "PreToolUse": [
                {
                    "matcher": "Bash",
                    "hooks": [
                        {"type": "command", "command": "python x:/own/audit.py"}
                    ],
                }
            ],
        },
    }
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.local.json").write_text(
        json.dumps(pre_existing, indent=2), encoding="utf-8"
    )

    assert run_init(project, memory_dir, fake_exe) == 0
    settings = read_settings(project)

    # Non-hook keys survive verbatim.
    assert settings["permissions"] == pre_existing["permissions"]
    assert settings["env"] == pre_existing["env"]

    # The custom hooks survive, in place.
    session_start_group = settings["hooks"]["SessionStart"][0]
    assert session_start_group["hooks"][0]["command"] == (
        "python x:/own/state_render.py --print"
    )
    custom_pre = [
        g for g in settings["hooks"]["PreToolUse"] if g.get("matcher") == "Bash"
    ]
    assert len(custom_pre) == 1
    assert custom_pre[0]["hooks"][0]["command"] == "python x:/own/audit.py"

    # Kit hooks were still added: appended into the matching-"" group and
    # into a NEW PreToolUse group with the kit matcher.
    ss_cmds = [h["command"] for h in session_start_group["hooks"]]
    assert any("memory_decay.py" in c for c in ss_cmds)
    kit_pre = [
        g
        for g in settings["hooks"]["PreToolUse"]
        if g.get("matcher") == "Edit|Write|Bash|Agent|NotebookEdit"
    ]
    assert len(kit_pre) == 1
    assert any("tool-use --stdin" in h["command"] for h in kit_pre[0]["hooks"])


def test_preexisting_claude_mem_wiring_is_skipped(tmp_path: Path) -> None:
    """A project already wired with its own claude-mem install (different
    exe path, production style) must not get a second copy of any hook."""
    project, memory_dir, fake_exe = make_project(tmp_path)
    live_style = {
        "hooks": {
            "SessionStart": [
                {
                    "matcher": "",
                    "hooks": [
                        {
                            "type": "command",
                            "command": (
                                "C:/Users/Someone/Scripts/claude-mem.exe "
                                "session-start --stdin"
                            ),
                            "timeout": 10000,
                        }
                    ],
                }
            ]
        }
    }
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    (claude_dir / "settings.local.json").write_text(
        json.dumps(live_style, indent=2), encoding="utf-8"
    )

    assert run_init(project, memory_dir, fake_exe) == 0
    settings = read_settings(project)
    session_start_cmds = [
        c for e, c in all_hook_commands(settings) if e == "SessionStart"
    ]
    # Exactly one session-start (the pre-existing one), not two.
    assert (
        len([c for c in session_start_cmds if "session-start" in c]) == 1
    )
    assert any("C:/Users/Someone/Scripts/claude-mem.exe" in c for c in session_start_cmds)
    # The kit's decay/health hooks were still added alongside.
    assert any("memory_decay.py" in c for c in session_start_cmds)
    assert any("memory_health.py" in c for c in session_start_cmds)


# ---------------------------------------------------------------------------
# Never-overwrite + dry-run + failure modes.
# ---------------------------------------------------------------------------


def test_existing_claude_md_never_overwritten(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    sentinel = "# My constitution\n\nDo not touch.\n"
    (project / "CLAUDE.md").write_text(sentinel, encoding="utf-8")

    assert run_init(project, memory_dir, fake_exe) == 0
    assert (project / "CLAUDE.md").read_text(encoding="utf-8") == sentinel

    # Idempotent for MEMORY.md / ledger too: seed once, never replace.
    (memory_dir / "MEMORY.md").write_text("custom memory", encoding="utf-8")
    ledger = project / ".superpowers" / "sdd" / "progress.md"
    ledger.write_text("custom ledger", encoding="utf-8")
    assert run_init(project, memory_dir, fake_exe) == 0
    assert (memory_dir / "MEMORY.md").read_text(encoding="utf-8") == "custom memory"
    assert ledger.read_text(encoding="utf-8") == "custom ledger"


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe, "--dry-run") == 0

    assert not memory_dir.exists()
    assert not (project / ".claude").exists()
    assert not (project / ".superpowers").exists()
    assert not (project / "CLAUDE.md").exists()
    assert list(project.iterdir()) == []  # truly nothing inside the project


def test_dry_run_leaves_existing_settings_untouched(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    original = json.dumps({"permissions": {"allow": []}}, indent=2)
    settings_path = claude_dir / "settings.local.json"
    settings_path.write_text(original, encoding="utf-8")

    assert run_init(project, memory_dir, fake_exe, "--dry-run") == 0
    assert settings_path.read_text(encoding="utf-8") == original


def test_shared_settings_flag_targets_settings_json(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    assert run_init(project, memory_dir, fake_exe, "--settings", "shared") == 0
    assert (project / ".claude" / "settings.json").is_file()
    assert not (project / ".claude" / "settings.local.json").exists()
    settings = read_settings(project, "settings.json")
    assert "SessionStart" in settings["hooks"]


def test_unparseable_settings_aborts_before_any_write(tmp_path: Path) -> None:
    project, memory_dir, fake_exe = make_project(tmp_path)
    claude_dir = project / ".claude"
    claude_dir.mkdir(parents=True)
    settings_path = claude_dir / "settings.local.json"
    settings_path.write_text("{ this is not json", encoding="utf-8")

    assert run_init(project, memory_dir, fake_exe) == 1
    # The broken file is untouched and NOTHING else was created.
    assert settings_path.read_text(encoding="utf-8") == "{ this is not json"
    assert not (project / "CLAUDE.md").exists()
    assert not (project / ".superpowers").exists()
    assert not memory_dir.exists()


def test_missing_project_root_errors(tmp_path: Path) -> None:
    _, memory_dir, fake_exe = make_project(tmp_path)
    missing = tmp_path / "does-not-exist"
    assert run_init(missing, memory_dir, fake_exe) == 1


# ---------------------------------------------------------------------------
# Identity matching unit coverage.
# ---------------------------------------------------------------------------


def test_command_identity_distinguishes_tool_use_variants() -> None:
    assert hi.command_identity("x/claude-mem.exe tool-use --stdin") == (
        "claude-mem",
        "tool-use",
    )
    assert hi.command_identity("x/claude-mem.exe tool-use-post --stdin") == (
        "claude-mem",
        "tool-use-post",
    )


def test_command_identity_matches_module_form() -> None:
    ident = hi.command_identity(
        "C:/py/python.exe -X utf8 -m claude_mem.cli session-end --stdin"
    )
    assert ident == ("claude-mem", "session-end")


def test_command_identity_ignores_unrelated_commands() -> None:
    assert hi.command_identity("python x:/own/state_render.py --print") is None
