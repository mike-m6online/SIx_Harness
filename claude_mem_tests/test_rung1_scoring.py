from claude_mem.bulk import doc_signal_weight, signal_weight

SUMMARY = (
    "This session is being continued from a previous conversation that ran "
    "out of context. We decided no shortcuts and the root cause was X."
)

def test_compaction_summary_is_floored():
    assert signal_weight(SUMMARY, role="user") <= 5

def test_real_user_correction_still_high():
    assert signal_weight("no shortcuts, you already built that", role="user") == 100

def test_doc_signal_weight_canonical_inventory_outranks_memory():
    # Canonical Tier-2 inventory must outrank Tier-3 memory (40).
    assert doc_signal_weight("docs/marathon/substrate_capability_inventory.md") == 50

def test_doc_signal_weight_other_doc_defaults_to_20():
    assert doc_signal_weight("docs/marathon/some_other_atlas.md") == 20
