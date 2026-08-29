from claude_mem.bulk import collect_do_not_rebuild_modules, is_do_not_rebuild


def _state(root, name, dnr):
    d = root / "docs" / "marathon" / "module_states"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.state.yaml").write_text(
        f"auto_derived:\n  config_flag: {name}\ndo_not_rebuild: {str(dnr).lower()}\n",
        encoding="utf-8")


def test_collect_dnr_modules(tmp_path):
    _state(tmp_path, "use_apollo", True)
    _state(tmp_path, "use_experimental", False)
    s = collect_do_not_rebuild_modules(tmp_path)
    assert "use_apollo" in s
    assert "use_experimental" not in s


def test_dnr_phrases_cover_dormant_inventory_entries():
    # Inventory DORMANT/built-but-never-enabled wording must mark do_not_rebuild.
    assert is_do_not_rebuild(
        "use_shared_prototype_library: DORMANT. Built but never enabled in production."
    ) is True


def test_dnr_plain_sentence_stays_false():
    assert is_do_not_rebuild("We should investigate the felt-state projection later.") is False
