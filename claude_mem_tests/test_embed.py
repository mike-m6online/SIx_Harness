import pytest

from claude_mem.embed import EmbeddingClient, EmbeddingError


class _StaticResponse:
    """Hand-rolled response substitute used in place of patched HTTP."""
    def __init__(self, status_code: int, payload: dict | None = None,
                 text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_embed_text_returns_vector(monkeypatch):
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model  # skip the probe
    monkeypatch.setattr(
        client._http, "post",
        lambda *a, **k: _StaticResponse(200, {"embedding": [0.01] * 1024}),
    )
    vec = client.embed("hello world")
    assert len(vec) == 1024
    assert isinstance(vec[0], float)


def test_embed_text_raises_on_http_error(monkeypatch):
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    # Pre-resolve to skip the probe loop and exercise the request-error path.
    client._resolved_model = client.model
    monkeypatch.setattr(
        client._http, "post",
        lambda *a, **k: _StaticResponse(500, text="ollama down"),
    )
    with pytest.raises(EmbeddingError, match="ollama"):
        client.embed("hello world")


def test_embed_falls_back_to_secondary_model(monkeypatch):
    """If the primary model returns 404, the client probes the fallback."""
    client = EmbeddingClient(
        model="qwen3-embedding:0.6b",
        fallback_model="nomic-embed-text:latest",
    )
    call_log: list[str] = []

    def _post(url, json=None, **kw):
        model = (json or {}).get("model", "")
        call_log.append(model)
        if model == "qwen3-embedding:0.6b":
            return _StaticResponse(404, text="model not found")
        return _StaticResponse(200, {"embedding": [0.02] * 1024})

    monkeypatch.setattr(client._http, "post", _post)
    vec = client.embed("hi")
    assert len(vec) == 1024
    assert "nomic-embed-text:latest" in call_log


# ---------------------------------------------------------------------------
# Dimension-mismatch guard (live-ingestion defect: qwen3-embedding:0.6b
# (1024-dim) falling back to nomic-embed-text (768-dim) against a vec table
# fixed at 1024 dims -- every embed() "succeeded" and then failed the
# sqlite-vec INSERT with "Dimension mismatch ... Expected 1024" (85 live
# occurrences, stalled the backfill at 8,602/15,157 while looking like it
# was working). A dimensionally-incompatible fallback is worse than a
# failure: it burns GPU time producing an unusable vector.
# ---------------------------------------------------------------------------

def test_embed_hard_errors_when_primary_returns_wrong_dim(monkeypatch):
    """The PRIMARY model returning the wrong dimension is a config/model
    mismatch, not a fallback-worthy condition -- it must raise loudly."""
    client = EmbeddingClient(
        model="qwen3-embedding:0.6b",
        fallback_model="nomic-embed-text:latest",
        embedding_dim=1024,
    )
    client._resolved_model = client.model  # skip the probe
    monkeypatch.setattr(
        client._http, "post",
        lambda *a, **k: _StaticResponse(200, {"embedding": [0.01] * 768}),
    )
    with pytest.raises(EmbeddingError, match="768"):
        client.embed("hello world")


def test_embed_refuses_wrong_dim_fallback_vector(monkeypatch):
    """When the primary is unreachable and the resolved fallback model
    produces a vector of the wrong dimensionality, embed() must refuse
    to return it (raise EmbeddingError) rather than silently returning
    a 768-dim vector into a 1024-dim caller."""
    client = EmbeddingClient(
        model="qwen3-embedding:0.6b",
        fallback_model="nomic-embed-text:latest",
        embedding_dim=1024,
    )

    def _post(url, json=None, **kw):
        model = (json or {}).get("model", "")
        if model == "qwen3-embedding:0.6b":
            return _StaticResponse(404, text="model not found")
        # fallback model resolved -- but it is a 768-dim model
        return _StaticResponse(200, {"embedding": [0.02] * 768})

    monkeypatch.setattr(client._http, "post", _post)
    with pytest.raises(EmbeddingError, match="768"):
        client.embed("hi")
    # The failure message must name the incompatibility explicitly so it
    # is actionable from ingestion_log.detail without re-deriving it.
    with pytest.raises(EmbeddingError, match="1024"):
        client.embed("hi")


def test_embed_dimension_check_applies_after_fallback_resolved(monkeypatch):
    """Same-dimension fallback models remain fully usable -- only a
    dimension MISMATCH is refused, not the fallback mechanism itself."""
    client = EmbeddingClient(
        model="qwen3-embedding:0.6b",
        fallback_model="some-other-1024-model",
        embedding_dim=1024,
    )

    def _post(url, json=None, **kw):
        model = (json or {}).get("model", "")
        if model == "qwen3-embedding:0.6b":
            return _StaticResponse(404, text="model not found")
        return _StaticResponse(200, {"embedding": [0.02] * 1024})

    monkeypatch.setattr(client._http, "post", _post)
    vec = client.embed("hi")
    assert len(vec) == 1024


def test_config_default_fallback_is_none_not_dimension_incompatible():
    """DEFAULT_CONFIG['embedding_fallback'] used to name nomic-embed-text
    (768-dim) as the default fallback for the 1024-dim primary -- broken
    by construction, since embed() now refuses any dim-mismatched
    fallback vector. The default must be None (no fallback) unless an
    operator explicitly configures a dimension-compatible one."""
    from claude_mem.config import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["embedding_fallback"] is None


def test_no_fallback_configured_hard_errors_when_primary_unreachable(monkeypatch):
    """fallback_model=None (the new default) must not be probed as a
    literal model name -- if the primary is unreachable and there is no
    fallback configured, embed() raises EmbeddingError cleanly."""
    client = EmbeddingClient(model="qwen3-embedding:0.6b", fallback_model=None)
    monkeypatch.setattr(
        client._http, "post",
        lambda *a, **k: _StaticResponse(404, text="model not found"),
    )
    with pytest.raises(EmbeddingError):
        client.embed("hi")


# ---------------------------------------------------------------------------
# Context-overflow head-truncation (live attrition: ~0.7% of chunks exceed
# the 8192-token embed window; Ollama returns 500 "the input length exceeds
# the context length" and the chunk stays vectorless forever). On overflow
# the client retries with the HEAD of the text, halving until it fits: a
# head-vector strictly dominates no vector, and chunks_fts still covers the
# full text for the BM25 half of hybrid search.
# ---------------------------------------------------------------------------

_OVERFLOW_500 = '{"error":"the input length exceeds the context length"}'


def _overflow_post(accept_below: int, prompts: list):
    def _post(url, json=None, **kw):
        prompt = (json or {}).get("prompt", "")
        prompts.append(prompt)
        if len(prompt) >= accept_below:
            return _StaticResponse(500, text=_OVERFLOW_500)
        return _StaticResponse(200, {"embedding": [0.03] * 1024})
    return _post


def test_embed_truncates_head_on_context_overflow(monkeypatch):
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model
    prompts: list = []
    monkeypatch.setattr(
        client._http, "post", _overflow_post(30000, prompts))
    text = "x" * 50000 + "TAIL-NEVER-EMBEDDED"
    vec = client.embed(text)
    assert len(vec) == 1024
    assert len(prompts) >= 2  # full attempt, then truncated retr(ies)
    final = prompts[-1]
    assert len(final) < 30000
    assert final == text[: len(final)]  # a HEAD slice, never a resample
    assert "TAIL-NEVER-EMBEDDED" not in final


def test_embed_overflow_keeps_halving_until_fit(monkeypatch):
    # Dense-token content: even the first truncated length overflows;
    # the client halves again rather than giving up.
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model
    prompts: list = []
    monkeypatch.setattr(
        client._http, "post", _overflow_post(10000, prompts))
    vec = client.embed("y" * 60000)
    assert len(vec) == 1024
    assert len(prompts[-1]) < 10000


def test_embed_overflow_raises_after_floor(monkeypatch):
    # Pathological server that rejects everything: the retry loop must
    # terminate and surface the original failure class for
    # ingestion_log(action='embed_fail'), not spin forever.
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model
    prompts: list = []
    monkeypatch.setattr(client._http, "post", _overflow_post(0, prompts))
    with pytest.raises(EmbeddingError, match="context length"):
        client.embed("z" * 60000)
    assert len(prompts) < 20  # bounded, no infinite halving


def test_embed_non_overflow_500_does_not_truncate(monkeypatch):
    # Only the context-overflow failure triggers the retry; any other
    # 500 propagates immediately (no masking of real server errors).
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model
    prompts: list = []

    def _post(url, json=None, **kw):
        prompts.append((json or {}).get("prompt", ""))
        return _StaticResponse(500, text='{"error":"cuda OOM"}')

    monkeypatch.setattr(client._http, "post", _post)
    with pytest.raises(EmbeddingError, match="cuda OOM"):
        client.embed("w" * 60000)
    assert len(prompts) == 1


# ---------------------------------------------------------------------------
# keep_alive (2026-08-19 embed-resilience fix): Ollama evicts an idle model
# after its server-default 5m window, so under marathon usage every
# post-idle hook-path embed hit a COLD ~2s reload right at the 2.0s read
# timeout and the vector leg silently died to BM25-only. Every request must
# carry keep_alive (a request that omits it RESETS the residency timer to
# the server default) -- on the resolve probe AND on every embed call.
# ---------------------------------------------------------------------------

def test_embed_request_carries_default_keep_alive(monkeypatch):
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    client._resolved_model = client.model  # skip the probe
    bodies: list[dict] = []

    def _post(url, json=None, **kw):
        bodies.append(json or {})
        return _StaticResponse(200, {"embedding": [0.01] * 1024})

    monkeypatch.setattr(client._http, "post", _post)
    client.embed("hello world")
    assert bodies and bodies[0]["keep_alive"] == "4h"


def test_embed_request_carries_configured_keep_alive(monkeypatch):
    """The config.py DEFAULT_CONFIG override pattern: a per-project
    embedding_keep_alive value flows into every request body."""
    client = EmbeddingClient(model="qwen3-embedding:0.6b", keep_alive="30m")
    client._resolved_model = client.model
    bodies: list[dict] = []

    def _post(url, json=None, **kw):
        bodies.append(json or {})
        return _StaticResponse(200, {"embedding": [0.01] * 1024})

    monkeypatch.setattr(client._http, "post", _post)
    client.embed("hello world")
    assert bodies and bodies[0]["keep_alive"] == "30m"


def test_resolve_probe_carries_keep_alive(monkeypatch):
    """The model-resolve probe is a real Ollama request too: omitting
    keep_alive there would reset the residency timer at every hook
    process start."""
    client = EmbeddingClient(model="qwen3-embedding:0.6b")
    bodies: list[dict] = []

    def _post(url, json=None, **kw):
        bodies.append(json or {})
        return _StaticResponse(200, {"embedding": [0.01] * 1024})

    monkeypatch.setattr(client._http, "post", _post)
    client.embed("hi")
    # First request is the resolve probe, second the real embed.
    assert len(bodies) == 2
    assert all(b["keep_alive"] == "4h" for b in bodies)


def test_default_config_declares_keep_alive():
    from claude_mem.config import DEFAULT_CONFIG
    assert DEFAULT_CONFIG["embedding_keep_alive"] == "4h"
