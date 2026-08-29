"""Ollama embedding client. Default model: Qwen3-Embedding-0.6B.

Fallback: an operator-configured model if the primary is unavailable at
the Ollama endpoint (resolved at first call; cached thereafter). No
fallback is configured by default (see config.py) -- see the dimension-
mismatch defect account below for why.

2026-07 dimension-mismatch defect (controller-diagnosed from a live
ingestion_log during the backfill): the old default fallback,
nomic-embed-text:latest, produces 768-dim vectors while the vec table is
fixed at embedding_dim (1024, matching the primary qwen3-embedding:0.6b).
When the primary went unreachable/evicted mid-backfill, _resolve_model()
silently pinned the 768-dim fallback; every embed() call then "succeeded"
and only failed downstream at the sqlite-vec INSERT with an opaque
"Dimension mismatch for inserted vector ... Expected 1024" -- 85 live
occurrences, and the backfill appeared to be running (embed() raised
nothing) while actually stalled at a fixed offset (8,602/15,157) burning
GPU time on unusable vectors. embed() now verifies the returned vector's
length against embedding_dim itself: a wrong-dim PRIMARY response is a
hard config/model-mismatch error (never silently substitute); a wrong-dim
FALLBACK response is refused with the same EmbeddingError failure class
callers already log via ingestion_log(action='embed_fail') -- the
fallback MECHANISM is unchanged and still used for genuinely
dimension-compatible fallback models, only an incompatible vector is
refused.

Root cause of the 2026-07 zero-vector incident (spec R2b): two prior
fixes each solved half the problem and left chunks_vec at 0 rows.
  1. 2026-05-24 (b46c40cb) cut read_timeout_s from 30.0 -> 2.0 to stop a
     DEAD Ollama daemon from stalling interactive Claude Code hooks for
     minutes. Correct for the hook path, sized for a short warm probe.
  2. 2026-06-04 (634372ee) added embed_num_ctx=8192 so large chunk bodies
     stop being silently truncated before encoding -- but did not widen
     the read timeout to match the larger context window.
The result: bulk/backfill embedding of this project's chunks (avg ~1.2KB,
max ~42KB content) against a LIVE-but-slow Ollama (often sharing the
daemon with a concurrent qwen2.5:7b generation call) frequently exceeds
2.0s and raises httpx.ReadTimeout -> EmbeddingError. ingest.py's
`except Exception: pass` around the embed call (no logging) swallowed
every one of those failures with zero trace, so chunks_vec stayed empty
through both "fixes" and no operator ever saw a failure count.
The fix (this module): keep read_timeout_s=2.0 as the DEFAULT (hook path
correctness is unchanged), and require bulk/offline callers to pass an
explicit generous read_timeout_s (see BULK_READ_TIMEOUT_S) instead of
inheriting the hook-tuned default. ingest.py now logs every embed
failure to ingestion_log instead of swallowing it (see embed_backfill.py
for the resumable repair pass).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import httpx

# Read timeout for bulk/offline embedding callers (bulk ingestion,
# embed-backfill, extract-corrections) -- NOT the interactive hook
# default. Generous enough to tolerate a contended-but-alive Ollama
# daemon (e.g. a concurrent qwen2.5:7b generation load) without ever
# hanging indefinitely.
BULK_READ_TIMEOUT_S = 60.0


class EmbeddingError(RuntimeError):
    """Raised when Ollama returns a non-200 or malformed response."""


class ContextLengthExceeded(EmbeddingError):
    """Ollama 500 'the input length exceeds the context length': the text
    does not fit the embed model's num_ctx window. Retryable by
    head-truncation, unlike every other EmbeddingError."""


# Head-truncation retry bounds for ContextLengthExceeded. START is a
# first-guess character budget for an 8192-token window (~3 chars/token
# holds for English prose; dense code/JSON runs lower, which the halving
# loop absorbs). FLOOR guards termination: if ~1000 chars still overflow
# 8192 tokens, the failure is not about length -- surface it.
_CTX_OVERFLOW_START_CHARS = 24000
_CTX_OVERFLOW_FLOOR_CHARS = 1000


@dataclass
class EmbeddingClient:
    model: str = "qwen3-embedding:0.6b"
    # No fallback by default (see config.py DEFAULT_CONFIG comment): a
    # fallback model must produce vectors of the same dimensionality as
    # `embedding_dim`, and that is a property of the specific model an
    # operator chooses to configure, not something this client can
    # assume. Explicitly pass a dimension-compatible fallback_model to
    # opt in.
    fallback_model: Optional[str] = None
    # 127.0.0.1 literal, NOT "localhost" -- see config.py DEFAULT_CONFIG
    # ("ollama_endpoint"): IPv6-first resolution vs Ollama's IPv4-only
    # bind wastes the connect budget on a doomed ::1 attempt.
    endpoint: str = "http://127.0.0.1:11434"
    # The vec table (schema.py DEFAULT_EMBEDDING_DIM) is fixed at this
    # width. embed() verifies every returned vector against it -- see
    # the module docstring's 2026-07 dimension-mismatch defect account.
    embedding_dim: int = 1024
    # 0.5s connect so a dead Ollama daemon does not block interactive Claude
    # Code hooks for minutes -- an unreachable port is caught immediately.
    connect_timeout_s: float = 0.5
    # Root cause (2026-07 recon, see docstring below): read_timeout_s=2.0
    # is correct for the interactive hook path (prompt_submit/tool_use/
    # tool_use_post/session_end all construct EmbeddingClient() with
    # defaults on a 3-5s hook timeout budget -- a dead/contended Ollama
    # must fail fast so Claude Code is never blocked). Bulk/offline callers
    # (bulk, embed-backfill, extract-corrections) that embed many chunks
    # against a possibly-contended-but-alive daemon must NOT use this
    # default -- they pass an explicit, much larger read_timeout_s (see
    # BULK_READ_TIMEOUT_S below) so legitimate slow responses are awaited
    # instead of raising EmbeddingError.
    read_timeout_s: float = 2.0
    # Ollama's default context window for embedding models is typically a few
    # hundred tokens, which silently truncates large chunk bodies before they
    # are embedded. Setting num_ctx to 8192 ensures full chunks are encoded.
    embed_num_ctx: int = 8192
    # Ollama model residency (2026-08-19 embed-resilience fix): Ollama
    # evicts an idle model after its keep_alive window (server default
    # 5m). Under marathon usage the embed model was evicted between
    # turns, so the next hook-path embed hit a COLD reload (~2s) right
    # at the 2.0s read_timeout_s edge and the vector leg silently died
    # to BM25-only. Every request re-arms the residency timer with this
    # value, so a 4h window keeps the embedder resident across normal
    # idle gaps without pinning it forever. Configurable per project via
    # config.py DEFAULT_CONFIG["embedding_keep_alive"]; passed on BOTH
    # the resolve probe and every embed request (an Ollama request that
    # omits keep_alive RESETS the timer to the server default, which
    # would silently undo the residency).
    keep_alive: str = "4h"
    _http: httpx.Client = field(init=False)
    _resolved_model: str = field(init=False, default="")

    def __post_init__(self) -> None:
        self._http = httpx.Client(
            timeout=httpx.Timeout(
                self.read_timeout_s, connect=self.connect_timeout_s,
            ),
        )

    def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model
        # Try primary first by issuing a 1-token probe. fallback_model may
        # be None (no fallback configured) -- skip probing it as a literal
        # candidate rather than sending a null model name to Ollama.
        candidates = [self.model]
        if self.fallback_model:
            candidates.append(self.fallback_model)
        for candidate in candidates:
            try:
                # The probe MUST carry the same options as _embed_once:
                # Ollama reloads a model whenever a request's effective
                # num_ctx differs from the loaded instance, so a probe
                # without options (server-default ctx) followed by an
                # embed at embed_num_ctx forced TWO full model reloads
                # per client -- each ~2-4s, blowing read_timeout_s and
                # silently killing the hook-path vector leg (traced
                # 2026-08-19 via the embed_degradation telemetry).
                resp = self._http.post(
                    f"{self.endpoint}/api/embeddings",
                    json={"model": candidate, "prompt": "probe",
                          "keep_alive": self.keep_alive,
                          "options": {"num_ctx": self.embed_num_ctx}},
                )
                if resp.status_code == 200:
                    self._resolved_model = candidate
                    return candidate
            except httpx.HTTPError:
                continue
        raise EmbeddingError(
            f"Neither {self.model} nor {self.fallback_model} reachable at "
            f"{self.endpoint}"
        )

    def embed(self, text: str) -> List[float]:
        """Embed `text`, retrying with the HEAD of the text when it
        overflows the embed model's context window. The head-vector
        indexes the chunk's opening for the vector half of hybrid
        search; chunks_fts still covers the FULL text for BM25 -- a
        head-vector strictly dominates the alternative (the chunk
        staying vectorless forever, the pre-fix live attrition on
        ~0.7% of the corpus). Every other failure propagates on the
        first attempt."""
        model = self._resolve_model()
        try:
            return self._embed_once(model, text)
        except ContextLengthExceeded:
            limit = min(len(text), _CTX_OVERFLOW_START_CHARS)
            while True:
                try:
                    return self._embed_once(model, text[:limit])
                except ContextLengthExceeded:
                    if limit <= _CTX_OVERFLOW_FLOOR_CHARS:
                        raise
                    limit //= 2

    def _embed_once(self, model: str, text: str) -> List[float]:
        try:
            resp = self._http.post(
                f"{self.endpoint}/api/embeddings",
                json={"model": model, "prompt": text,
                      "keep_alive": self.keep_alive,
                      "options": {"num_ctx": self.embed_num_ctx}},
            )
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"ollama request failed: {exc}") from exc
        if resp.status_code != 200:
            if "exceeds the context length" in resp.text:
                raise ContextLengthExceeded(
                    f"ollama returned {resp.status_code}: {resp.text[:200]}"
                )
            raise EmbeddingError(
                f"ollama returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        vec = data.get("embedding")
        if not isinstance(vec, list) or not vec:
            raise EmbeddingError(f"ollama response missing 'embedding': {data}")
        vec = [float(x) for x in vec]
        if len(vec) != self.embedding_dim:
            if model == self.model:
                # Primary model returned the wrong width: a config/model
                # mismatch (e.g. embedding_dim misconfigured, or the
                # primary model at the endpoint was silently swapped) --
                # never silently reshape or truncate; surface it loudly.
                raise EmbeddingError(
                    f"primary model {model!r} returned a {len(vec)}-dim "
                    f"vector, expected embedding_dim={self.embedding_dim} "
                    f"-- config/model mismatch, refusing to use it"
                )
            # A resolved FALLBACK model producing the wrong width is the
            # 2026-07 live defect: it silently "succeeds" here and then
            # fails far away at the sqlite-vec INSERT with an opaque
            # "Dimension mismatch" error, after burning GPU time. Refuse
            # it with the same EmbeddingError failure class callers
            # already log to ingestion_log(action='embed_fail') -- the
            # fallback mechanism itself is fine; only this vector is not.
            raise EmbeddingError(
                f"fallback model dimension {len(vec)} != "
                f"{self.embedding_dim} -- refusing incompatible vector"
            )
        return vec

    def close(self) -> None:
        self._http.close()
