import json
from claude_mem.cli import _patch_settings


def test_patch_settings_registers_posttooluse(tmp_path):
    settings = tmp_path / "settings.json"
    _patch_settings(settings, "claude-mem")
    data = json.loads(settings.read_text(encoding="utf-8"))
    post = data["hooks"].get("PostToolUse", [])
    cmds = [h["command"] for e in post for h in e["hooks"]]
    assert any("tool-use-post --stdin" in c for c in cmds)
    matchers = [e.get("matcher") for e in post]
    assert "Edit|Write|Bash|Agent" in matchers
