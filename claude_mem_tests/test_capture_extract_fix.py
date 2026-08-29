import json
import shutil
import tempfile
from pathlib import Path

from claude_mem.capture import CaptureStore
from claude_mem.hooks.session_end import capture_from_jsonl, run_candidates
from claude_mem.schema import init_db


def _write_session_jsonl(path: Path) -> None:
    rows = [
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "Decision: we will adopt the live-Welford "
             "normalizer instead of the frozen z-score for the felt path."}]}},
        {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "We rejected the variance-floor approach; "
             "it was shelved because Mike ruled an estimator is not a floor."}]}},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_capture_from_jsonl_lands_watermark_and_rows(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    jsonl = tmp_path / "sess-fixture.jsonl"
    _write_session_jsonl(jsonl)
    store = CaptureStore(db)
    try:
        n_dec, n_de, skips = capture_from_jsonl(store, jsonl)
        # first sight anchors, captures nothing
        assert (n_dec, n_de, skips.total) == (0, 0, 0)
        assert store.get_meta(f"extract_offset:{jsonl.name}") is not None
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Decision: gate the clamp behind "
                 "felt_zscore_clamp, default off."}]}}) + "\n")
        n_dec, _n_de, _skips = capture_from_jsonl(store, jsonl)
        assert n_dec == 1
    finally:
        store.close()


def test_run_candidates_recovers_when_session_id_missing(monkeypatch):
    # Use a short base path so the proj slug remains within Windows MAX_PATH
    # when nested under home/.claude/projects/<slug>/ -- the slug encodes the
    # full project_root string, so a deep pytest tmp_path doubles the nesting.
    base = Path(tempfile.gettempdir()) / "cm_cef"
    if base.exists():
        shutil.rmtree(base)
    base.mkdir(parents=True)
    try:
        db = base / "index.db"
        init_db(db)
        proj = base / "p"
        proj.mkdir()
        home = base / "h"
        slug = str(proj.resolve()).replace(":", "-").replace("\\", "-").replace("/", "-")
        pdir = home / ".claude" / "projects" / slug
        pdir.mkdir(parents=True)
        jsonl = pdir / "11111111-2222-3333-4444-555555555555.jsonl"
        _write_session_jsonl(jsonl)
        monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
        from claude_mem.config import ProjectConfig
        real_init = ProjectConfig.__init__
        def patched_init(self, project_root):
            real_init(self, project_root=project_root)
            self.db_path = db
        monkeypatch.setattr(ProjectConfig, "__init__", patched_init)
        run_candidates("", proj.resolve())
        store = CaptureStore(db)
        try:
            assert store.get_meta(f"extract_offset:{jsonl.name}") is not None
        finally:
            store.close()
        # Append a decision line and call again; assert real capture occurred.
        with jsonl.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"type": "assistant", "message": {"content": [
                {"type": "text", "text": "Decision: we will adopt the live-Welford "
                 "normalizer for all felt-path signals."}]}}) + "\n")
        run_candidates("", proj.resolve())
        store2 = CaptureStore(db)
        try:
            assert len(store2.list_decisions()) > 0
        finally:
            store2.close()
    finally:
        shutil.rmtree(base, ignore_errors=True)
