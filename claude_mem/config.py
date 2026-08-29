"""Per-project + global config loader."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "embedding_model": "qwen3-embedding:0.6b",
    # No fallback by default. nomic-embed-text:latest (the prior default)
    # produces 768-dim vectors while the vec table (embedding_dim below)
    # is fixed at 1024 -- a dimensionally-incompatible fallback is worse
    # than no fallback: embed.py's EmbeddingClient now hard-refuses any
    # returned vector whose width != embedding_dim, so a 768-dim default
    # fallback would just turn every primary-unreachable embed() call
    # into a logged embed_fail instead of a usable vector anyway (see the
    # 2026-07 dimension-mismatch defect account in embed.py's module
    # docstring). Set this explicitly only to a model that is confirmed
    # to emit `embedding_dim`-wide vectors.
    "embedding_fallback": None,
    "generation_model": "qwen2.5:7b",
    # 127.0.0.1 literal, NOT "localhost": Windows getaddrinfo orders ::1
    # before 127.0.0.1, Ollama binds 127.0.0.1 only, and a blackholed
    # IPv6 connect attempt eats the whole 0.5s connect budget (embed.py)
    # -- the client then reports "unreachable" without ever trying IPv4
    # (measured 2026-08-19: +2s/call via urllib, hard-fail via httpx).
    "ollama_endpoint": "http://127.0.0.1:11434",
    "embedding_dim": 1024,
    # Ollama residency window for the embed model, passed as keep_alive
    # on every embedding request (probe + embed). Ollama evicts an idle
    # model after its server-default 5m; under marathon usage that made
    # every post-idle hook-path embed a COLD ~2s reload right at the
    # 2.0s hook read timeout, silently degrading search to BM25-only.
    # 4h keeps the embedder resident across normal idle gaps. Accepts
    # any Ollama duration string ("30m", "4h") or -1 (pin forever).
    "embedding_keep_alive": "4h",
    "isolate_from_cross_project": False,
    "top_k_default": 10,
    "verbose": False,
}


@dataclass
class ProjectConfig:
    project_root: Path
    state_dir: Path = field(init=False)
    config_path: Path = field(init=False)
    db_path: Path = field(init=False)
    telemetry_path: Path = field(init=False)
    values: Dict[str, Any] = field(init=False)

    def __post_init__(self) -> None:
        self.state_dir = self.project_root / ".claude-mem"
        self.config_path = self.state_dir / "config.yaml"
        self.db_path = self.state_dir / "index.db"
        self.telemetry_path = self.state_dir / "telemetry.db"
        self.values = dict(DEFAULT_CONFIG)
        if self.config_path.is_file():
            loaded = yaml.safe_load(self.config_path.read_text()) or {}
            self.values.update(loaded)

    def write(self) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(yaml.safe_dump(self.values, sort_keys=True))
