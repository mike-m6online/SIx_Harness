from claude_mem.hooks.tool_use_post import _extract_text_post


def test_extract_text_post_edit_includes_path_and_result():
    text = _extract_text_post(
        "Edit",
        {"file_path": "src/foo.cu", "old_string": "a", "new_string": "b"},
        {"structuredPatch": [], "userModified": False},
    )
    assert "src/foo.cu" in text


def test_extract_text_post_bash_includes_command_and_output():
    text = _extract_text_post(
        "Bash",
        {"command": "grep felt_zscore_clamp", "description": "search clamp"},
        {"stdout": "config.h: felt_zscore_clamp = 0.0f;", "stderr": ""},
    )
    assert "felt_zscore_clamp" in text
    assert "config.h" in text


def test_extract_text_post_grep_includes_pattern_and_hits():
    text = _extract_text_post(
        "Grep",
        {"pattern": "discriminative embedding", "output_mode": "content"},
        "mlm_discriminative_head_forward_kernel.cu: emb = ...",
    )
    assert "discriminative embedding" in text
    assert "mlm_discriminative_head" in text


from claude_mem.hooks import tool_use_post


class _InjectedSearcher:
    def __init__(self, rows):
        self._rows = rows
    def search(self, *a, **k):
        return self._rows
    def close(self):
        pass


def _row(cid, score):
    return {"id": cid, "content": "existing felt welford normalizer subsystem",
            "file_path": "tools/x.py", "module": "felt", "source": "doc",
            "final_score": score}


def test_run_emits_then_suppresses_on_second_delivery(tmp_path, monkeypatch):
    from claude_mem.schema import init_db
    db = tmp_path / "index.db"
    init_db(db)
    from claude_mem.config import ProjectConfig
    real_init = ProjectConfig.__init__
    def patched(self, project_root):
        real_init(self, project_root=project_root)
        self.db_path = db
    monkeypatch.setattr(ProjectConfig, "__init__", patched)
    monkeypatch.setattr(tool_use_post, "_build_searcher",
                        lambda cfg: _InjectedSearcher([_row("c1", 0.05)]))
    args = ("Write",
            {"file_path": "felt_welford.py",
             "content": "implement welford normalizer"},
            {"ok": True}, "sessX")
    first = tool_use_post.run(*args, project_root=tmp_path)
    assert "existing felt welford" in first
    second = tool_use_post.run(*args, project_root=tmp_path)
    assert second == ""


def test_run_suppresses_below_threshold(tmp_path, monkeypatch):
    from claude_mem.schema import init_db
    from claude_mem.config import ProjectConfig
    db = tmp_path / "index.db"; init_db(db)
    real_init = ProjectConfig.__init__
    def patched(self, project_root):
        real_init(self, project_root=project_root)
        self.db_path = db
    monkeypatch.setattr(ProjectConfig, "__init__", patched)
    monkeypatch.setattr(tool_use_post, "_build_searcher",
                        lambda cfg: _InjectedSearcher([_row("c1", 0.001)]))
    out = tool_use_post.run("Write",
                            {"file_path": "f.py",
                             "content": "implement felt"},
                            {"ok": True}, "sessY", project_root=tmp_path)
    assert out == ""


def test_extract_text_post_tolerates_non_dict_input():
    # A non-dict tool_input must not raise (always-exit-0 safeguard).
    assert _extract_text_post("Write", "not-a-dict", {"ok": True}) is not None
    assert _extract_text_post("Edit", ["a", "list"], "result text") is not None


def test_extract_text_tolerates_non_dict_input():
    from claude_mem.hooks.tool_use import _extract_text
    assert _extract_text("Write", "not-a-dict") == ""
    assert _extract_text("Edit", ["a", "list"]) == ""


import json, subprocess, sys


def test_cli_tool_use_post_exits_zero_on_empty(tmp_path):
    payload = json.dumps({"tool_name": "Write",
                          "tool_input": {"file_path": "f.py", "content": "x"},
                          "tool_response": {"ok": True},
                          "session_id": "s", "cwd": str(tmp_path)})
    proc = subprocess.run(
        [sys.executable, "-c",
         "from claude_mem.cli import cli; cli()", "tool-use-post", "--stdin"],
        input=payload, capture_output=True, text=True,
    )
    assert proc.returncode == 0


def test_extract_text_post_bounds_huge_response():
    """Regression for the 2026-07-06 hook outage: the full stringified
    tool_response (multi-MB Bash/Read output) flowed into search unbounded.
    The extract must cap at MAX_QUERY_TEXT_CHARS while keeping the head
    (where the identifiers live)."""
    from claude_mem.hooks.tool_use import MAX_QUERY_TEXT_CHARS
    huge = "payload_head_symbol " + ("filler " * 2_000_000)
    text = _extract_text_post("Bash", {"command": "run thing"}, huge)
    assert len(text) <= MAX_QUERY_TEXT_CHARS
    assert "payload_head_symbol" in text
    assert "run thing" in text


def test_extract_text_bounds_huge_write_content():
    """Same bound on the PreToolUse side (a giant Write.content)."""
    from claude_mem.hooks.tool_use import MAX_QUERY_TEXT_CHARS, _extract_text
    text = _extract_text(
        "Write",
        {"file_path": "src/big.py", "content": "word " * 2_000_000},
    )
    assert len(text) <= MAX_QUERY_TEXT_CHARS
    assert "src/big.py" in text
