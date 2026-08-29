"""Local Ollama text-generation client (anti-recurrence Rung 3 synthesis).

Mirrors embed.py's httpx pattern, but calls /api/generate instead of
/api/embeddings. Used off the prompt hot path (SessionEnd + capture-synthesize)
to turn a thread's decisions/dead-ends into a dated lineage paragraph. All
failures raise GenerationError so callers can fall back without blocking."""
from __future__ import annotations

import httpx


class GenerationError(RuntimeError):
    pass


class GenerationClient:
    def __init__(
        self,
        *,
        model: str,
        # 127.0.0.1 literal, NOT "localhost" -- see config.py
        # DEFAULT_CONFIG ("ollama_endpoint").
        endpoint: str = "http://127.0.0.1:11434",
        connect_timeout_s: float = 0.5,
        read_timeout_s: float = 120.0,
    ) -> None:
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self._http = httpx.Client(
            timeout=httpx.Timeout(read_timeout_s, connect=connect_timeout_s),
        )

    def close(self) -> None:
        self._http.close()

    def generate(self, prompt: str) -> str:
        try:
            resp = self._http.post(
                f"{self.endpoint}/api/generate",
                json={"model": self.model, "prompt": prompt, "stream": False},
            )
        except httpx.HTTPError as exc:
            raise GenerationError(f"ollama transport error: {exc}") from exc
        if resp.status_code != 200:
            raise GenerationError(
                f"ollama returned {resp.status_code}: {resp.text[:200]}"
            )
        data = resp.json()
        text = data.get("response")
        if not isinstance(text, str) or not text.strip():
            raise GenerationError(f"ollama response missing 'response': {data}")
        return text.strip()
