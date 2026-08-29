from pathlib import Path
from claude_mem.bulk import collect_module_names, detect_module


def _make_state(root: Path, name: str) -> None:
    d = root / "docs" / "marathon" / "module_states"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.state.yaml").write_text(
        f"auto_derived:\n  config_flag: {name}\n", encoding="utf-8"
    )


def test_collect_module_names_reads_state_yaml(tmp_path):
    _make_state(tmp_path, "use_apollo")
    _make_state(tmp_path, "use_consequential_action")
    names = collect_module_names(tmp_path)
    assert "use_apollo" in names
    assert "use_consequential_action" in names
    assert "module_states" not in names


def test_detect_module_matches_flag_token(tmp_path):
    mods = ["use_apollo"]
    assert detect_module("the use_apollo loop is dormant", mods) == "use_apollo"
    assert detect_module("apollomania is unrelated", mods) is None


def test_detect_module_returns_earliest_position():
    # `use_l6_innovation` leads the chunk but sorts after `use_acc_meta_override`
    # alphabetically; earliest-position tagging must pick the flag it leads with.
    mods = ["use_acc_meta_override", "use_l6_innovation"]
    content = "L6 innovation: use_l6_innovation upstream of use_acc_meta_override clamp"
    assert detect_module(content, mods) == "use_l6_innovation"
