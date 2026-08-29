import json
import sqlite3
import tempfile
from pathlib import Path

from claude_mem.schema import init_db
from claude_mem.capture import CaptureStore
from claude_mem.hooks import session_end


def test_run_candidates_first_run_anchors_no_capture(monkeypatch):
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        (root / ".claude-mem").mkdir(parents=True)
        init_db(root / ".claude-mem" / "index.db")

        # Stand up a home projects dir + a matching session JSONL. The file
        # ends with a trailing newline so EOF is a clean line boundary.
        home = Path(tmp) / "home"
        home_proj = home / ".claude" / "projects" / "slugdir"
        home_proj.mkdir(parents=True)
        sid = "abc123"
        jsonl = home_proj / f"{sid}.jsonl"
        jsonl.write_text("\n".join([
            json.dumps({"message": {"role": "user",
                                    "content": "We decided to keep the MLM "
                                               "head wired for the F8 test."}}),
            json.dumps({"message": {"role": "assistant",
                                    "content": "Option X is rejected; it hits "
                                               "the predict-not-cause wall."}}),
        ]) + "\n", encoding="utf-8")

        # Redirect Path.home() to the temp home so run_candidates finds the JSONL.
        monkeypatch.setattr(session_end.Path, "home", lambda: home)

        # First run anchors the watermark at EOF and captures nothing.
        msg = session_end.run_candidates(sid, root)
        assert msg == ""
        conn = sqlite3.connect(root / ".claude-mem" / "index.db")
        try:
            assert conn.execute(
                "SELECT COUNT(*) FROM decisions"
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT COUNT(*) FROM dead_ends"
            ).fetchone()[0] == 0
        finally:
            conn.close()

        # Append newly-written session content; the second run mines only it.
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "We decided to go with the GW causal-write "
                                "path for the dialogue substrate."}}) + "\n")
            fh.write(json.dumps({"message": {"role": "assistant",
                     "content": "The FM-head rewire is rejected; it hits "
                                "the predict-not-cause wall."}}) + "\n")

        msg2 = session_end.run_candidates(sid, root)
        assert "pending" in msg2
        conn = sqlite3.connect(root / ".claude-mem" / "index.db")
        try:
            n_dec = conn.execute(
                "SELECT COUNT(*) FROM decisions WHERE state='pending'"
            ).fetchone()[0]
            n_de = conn.execute(
                "SELECT COUNT(*) FROM dead_ends WHERE state='pending'"
            ).fetchone()[0]
        finally:
            conn.close()
        assert n_dec >= 1
        assert n_de >= 1


def test_capture_from_jsonl_resume_and_truncate(tmp_path):
    db = tmp_path / "index.db"
    init_db(db)
    store = CaptureStore(db)
    try:
        jsonl = tmp_path / "session.jsonl"
        jsonl.write_text(json.dumps(
            {"message": {"role": "user", "content":
                "We decided to keep the MLM head wired for the F8 test."}}
        ) + "\n", encoding="utf-8")

        # First observation anchors the watermark at the current size.
        n_dec, n_de, _skips = session_end.capture_from_jsonl(store, jsonl)
        assert (n_dec, n_de) == (0, 0)
        assert store.get_meta(f"extract_offset:{jsonl.name}") == str(
            jsonl.stat().st_size
        )

        # Append a new decision-cue record; the next run mines only it.
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "We decided to go with the GW causal-write "
                                "path for the dialogue substrate."}}) + "\n")
        n_dec, _n_de, _skips = session_end.capture_from_jsonl(store, jsonl)
        assert n_dec >= 1

        # Rewrite the file SHORTER than the stored offset; truncation path must
        # clamp start to size and not raise.
        jsonl.write_text(json.dumps(
            {"message": {"role": "user", "content": "short."}}
        ) + "\n", encoding="utf-8")
        session_end.capture_from_jsonl(store, jsonl)
    finally:
        store.close()


def test_run_candidates_no_session_returns_empty():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        (root / ".claude-mem").mkdir(parents=True)
        init_db(root / ".claude-mem" / "index.db")
        assert session_end.run_candidates("", root) == ""


def test_run_candidates_summary_exposes_skip_counts(monkeypatch):
    """A6: the capture-extract summary line reports what the precision
    deny-list dropped, per category, instead of dropping silently."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        (root / ".claude-mem").mkdir(parents=True)
        init_db(root / ".claude-mem" / "index.db")

        home = Path(tmp) / "home"
        home_proj = home / ".claude" / "projects" / "slugdir"
        home_proj.mkdir(parents=True)
        sid = "skipcount"
        jsonl = home_proj / f"{sid}.jsonl"
        jsonl.write_text(json.dumps({"message": {"role": "user",
                         "content": "warm-up line, no cue at all."}}) + "\n",
                         encoding="utf-8")
        monkeypatch.setattr(session_end.Path, "home", lambda: home)

        # First run anchors the watermark.
        assert session_end.run_candidates(sid, root) == ""

        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "We decided to go with the GW causal-write "
                                "path for the dialogue substrate."}}) + "\n")
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "All four locked in."}}) + "\n")
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "Trimming superseded anchors from the "
                                "ledger per compaction discipline."}}) + "\n")

        msg = session_end.run_candidates(sid, root)
        assert "1 pending decision(s)" in msg
        assert "2 low-signal candidate(s) skipped" in msg
        assert "status-echo=1" in msg
        assert "memory-maintenance-records=1" in msg


def test_run_candidates_summary_silent_without_skips(monkeypatch):
    """No skips and no rows -> the summary stays empty (unchanged
    behavior); the skip clause never renders at zero."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "proj"
        (root / ".claude-mem").mkdir(parents=True)
        init_db(root / ".claude-mem" / "index.db")
        home = Path(tmp) / "home"
        home_proj = home / ".claude" / "projects" / "slugdir"
        home_proj.mkdir(parents=True)
        sid = "noskips"
        jsonl = home_proj / f"{sid}.jsonl"
        jsonl.write_text(json.dumps({"message": {"role": "user",
                         "content": "warm-up line, no cue at all."}}) + "\n",
                         encoding="utf-8")
        monkeypatch.setattr(session_end.Path, "home", lambda: home)
        assert session_end.run_candidates(sid, root) == ""
        with open(jsonl, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"message": {"role": "user",
                     "content": "still nothing resembling a cue here."}}) + "\n")
        assert session_end.run_candidates(sid, root) == ""
