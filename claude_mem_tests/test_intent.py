from claude_mem.intent import (
    has_build_intent, has_existing_subsystem_intent,
    has_investigation_intent, has_decision_intent,
)


def test_build_intent_fires_on_build_verbs():
    assert has_build_intent("let's build a new differential dispatcher")
    assert has_build_intent("implement the gate")
    assert has_build_intent("we should add a knob for this")
    assert not has_build_intent("status?")
    assert not has_build_intent("how is the run going")


def test_investigation_intent_fires_on_diagnose_verbs():
    assert has_investigation_intent(
        "let's investigate why the verifier stops firing"
    )
    assert has_investigation_intent("trace why postIE_v1 went dead")
    assert has_investigation_intent("can we dig into the bistability")
    assert has_investigation_intent("deep dive on the attention path")
    assert not has_investigation_intent("the trace is at line 42")
    assert not has_investigation_intent("status")


def test_combined_existing_subsystem_intent():
    assert has_existing_subsystem_intent("build the X")
    assert has_existing_subsystem_intent("let's diagnose Y")
    assert has_existing_subsystem_intent("trace why we're failing")
    assert not has_existing_subsystem_intent("status?")
    assert not has_existing_subsystem_intent("done")


def test_build_intent_does_not_fire_on_past_tense_descriptions():
    """The verb list intentionally matches present-tense construction
    proposals, not past-tense descriptions of work already done."""
    assert not has_build_intent("the engine was already built")
    assert not has_build_intent("we already implemented that")


def test_decision_intent_matches_step_back_and_double_down():
    assert has_decision_intent("should we double-down on the FM-head rewire?")
    assert has_decision_intent("let's step back and look at the lineage")
    assert has_decision_intent("do we pivot or keep going?")
    assert has_decision_intent("should we keep this approach or change it?")


def test_decision_intent_does_not_fire_on_common_prose():
    assert not has_decision_intent("read the config file path")
    assert not has_decision_intent("run the tests and show me the output")
    assert not has_decision_intent("fix the import in module foo")


def test_existing_subsystem_intent_includes_decision():
    assert has_existing_subsystem_intent("should we double-down or pivot here?")


# Fix #6 tests: verb noise removal + gerund false-negative
def test_decision_intent_bare_lineage_no_longer_fires():
    """'lineage' alone used to fire (was bare in list); only multi-word anchors now."""
    assert not has_decision_intent("git lineage shows the commit flow")


def test_decision_intent_or_change_it_no_longer_fires():
    """'or change it' was too broad; routine prose must not trigger."""
    assert not has_decision_intent("the fn should return X or change it to None")


def test_decision_intent_revisiting_gerund_fires():
    """'revisiting the decision' (gerund) was a false-negative; must now match."""
    assert has_decision_intent("are we revisiting the decision on the FM-head?")


def test_decision_intent_decision_lineage_multiword_fires():
    """Multi-word anchor 'decision lineage' must still fire."""
    assert has_decision_intent("let's read the decision lineage")


def test_decision_intent_replay_prompt_still_fires():
    """Regression: the acceptance-test replay prompt must still match."""
    assert has_decision_intent(
        "should we double-down on the FM-head rewire or step back?"
    )
