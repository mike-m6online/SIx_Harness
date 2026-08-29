import claude_mem


def test_version_is_string():
    assert isinstance(claude_mem.__version__, str)
    assert len(claude_mem.__version__) > 0
