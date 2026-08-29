import json
import sqlite3
import tempfile
from pathlib import Path

from click.testing import CliRunner

from claude_mem.cli import cli
from claude_mem.schema import init_db


def _init_project(tmp: str) -> Path:
    root = Path(tmp)
    (root / ".claude-mem").mkdir(parents=True)
    init_db(root / ".claude-mem" / "index.db")
    return root


def test_capture_extract_runs_without_session(tmp_path):
    root = _init_project(str(tmp_path))
    runner = CliRunner()
    res = runner.invoke(
        cli,
        ["capture-extract", "--project-root", str(root), "--session-id", ""],
    )
    assert res.exit_code == 0


def test_formalize_flow(tmp_path):
    root = _init_project(str(tmp_path))
    runner = CliRunner()

    r1 = runner.invoke(cli, [
        "thread-add", "--project-root", str(root),
        "--name", "Inner Dialogue Drives Action",
        "--summary", "two grounding problems",
    ])
    assert r1.exit_code == 0

    r2 = runner.invoke(cli, [
        "decision-add", "--project-root", str(root),
        "--title", "keep MLM for F8", "--date", "2026-06-04",
        "--rationale", "R3 does not touch F8 value",
        "--thread", "inner-dialogue-drives-action",
        "--rejected", "double-down FM-head", "--rejected", "resolution-PE",
        "--confirmed", "--mike-approved",
    ])
    assert r2.exit_code == 0

    r3 = runner.invoke(cli, [
        "dead-end-add", "--project-root", str(root),
        "--approach", "GW path never built", "--date", "2026-04-09",
        "--why", "deferred into F8 work",
        "--thread", "inner-dialogue-drives-action",
    ])
    assert r3.exit_code == 0

    r4 = runner.invoke(cli, ["capture-list", "--project-root", str(root)])
    assert r4.exit_code == 0
    assert "Inner Dialogue Drives Action" in r4.output
    assert "keep MLM for F8" in r4.output
    assert "GW path never built" in r4.output


def test_decision_confirm(tmp_path):
    root = _init_project(str(tmp_path))
    runner = CliRunner()
    runner.invoke(cli, [
        "decision-add", "--project-root", str(root),
        "--title", "pending one", "--date", "2026-06-04",
    ])
    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    did = conn.execute("SELECT id FROM decisions").fetchone()[0]
    conn.close()
    r = runner.invoke(cli, [
        "decision-confirm", "--project-root", str(root),
        "--id", did, "--mike-approved",
    ])
    assert r.exit_code == 0
    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    state, appr = conn.execute(
        "SELECT state, mike_approved FROM decisions WHERE id=?", (did,)
    ).fetchone()
    conn.close()
    assert state == "confirmed"
    assert appr == 1


from claude_mem.ingest import Ingester, Chunk


class _Emb:
    def embed(self, text):
        return [0.1] * 1024


def test_backfill_from_is_decision_chunks(tmp_path):
    root = _init_project(str(tmp_path))
    ing = Ingester(db_path=root / ".claude-mem" / "index.db", embedder=_Emb())
    ing.add(Chunk(content="We locked the Path A felt-state buildout decision.",
                  source="doc", role="doc", is_decision=True, signal_weight=80))
    ing.add(Chunk(content="Ordinary note, not a decision.",
                  source="doc", role="doc", is_decision=False))
    ing.add(Chunk(content="Conversational chatter tagged as a decision somehow.",
                  source="claude_code", role="user", is_decision=True,
                  signal_weight=100))
    ing.close()

    runner = CliRunner()
    res = runner.invoke(cli, ["capture-backfill-chunks", "--project-root", str(root)])
    assert res.exit_code == 0

    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    titles = [r[0] for r in conn.execute(
        "SELECT title FROM decisions WHERE state='pending'").fetchall()]
    conn.close()
    assert any("Path A felt-state" in t for t in titles)
    assert not any("Ordinary note" in t for t in titles)        # is_decision=False excluded
    assert not any("Conversational chatter" in t for t in titles)  # claude_code excluded


def test_backfill_title_word_boundary_clip_and_first_sentence_rationale(tmp_path):
    """A2 at cli capture-backfill-chunks: title is a 200-char whole-word
    clip (was a bare [:120] mid-word slice) and rationale carries the
    chunk's FULL first sentence (collapsed, capped 600)."""
    from claude_mem.textutil import ELLIPSIS

    first_sentence_text = (
        "We locked the Path A felt-state buildout decision because the "
        "live-Welford normalizer beat the frozen z-score on every felt-path "
        "signal in the matched control and Mike ratified the ordering."
    )
    content = (
        first_sentence_text
        + " A second sentence with follow-up detail that the title need "
          "not carry but the miner used to clip away mid-word."
    )
    assert len(content) > 200
    root = _init_project(str(tmp_path))
    ing = Ingester(db_path=root / ".claude-mem" / "index.db", embedder=_Emb())
    ing.add(Chunk(content=content, source="doc", role="doc",
                  is_decision=True, signal_weight=80))
    ing.close()

    runner = CliRunner()
    res = runner.invoke(cli, ["capture-backfill-chunks", "--project-root", str(root)])
    assert res.exit_code == 0, res.output

    conn = sqlite3.connect(root / ".claude-mem" / "index.db")
    title, rationale = conn.execute(
        "SELECT title, rationale FROM decisions WHERE state='pending'"
    ).fetchone()
    conn.close()
    assert len(title) <= 200
    assert title.endswith(ELLIPSIS)
    # Whole-word clip: every head token is a whole source token.
    assert all(tok in content.split() for tok in title[:-1].split())
    # Rationale = the FULL first sentence, untruncated (< 600 chars).
    assert rationale == first_sentence_text


def test_capture_synthesize_handles_ollama_down(tmp_path):
    root = _init_project(str(tmp_path))
    from claude_mem.capture import CaptureStore, Thread, Decision
    store = CaptureStore(root / ".claude-mem" / "index.db")
    store.add_thread(Thread(name="T", id="t", summary="s"))
    store.add_decision(Decision(title="d1", date="2026-06-04", state="confirmed",
                                thread_id="t"))
    store.close()
    runner = CliRunner()
    # Point at an unused endpoint so generation fails; command must still exit 0.
    res = runner.invoke(cli, [
        "capture-synthesize", "--project-root", str(root),
        "--endpoint", "http://127.0.0.1:1",
    ])
    assert res.exit_code == 0
    assert "thread" in res.output.lower()
