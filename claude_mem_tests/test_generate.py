import httpx
import pytest

from claude_mem.generate import GenerationClient, GenerationError


class _Resp:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


def test_generate_returns_response_text(monkeypatch):
    client = GenerationClient(model="qwen2.5:7b", endpoint="http://localhost:11434")

    def _post(url, json=None):
        assert url.endswith("/api/generate")
        assert json["model"] == "qwen2.5:7b"
        assert json["stream"] is False
        assert "prompt" in json
        return _Resp(200, {"response": "A -> B -> C lineage."})

    monkeypatch.setattr(client._http, "post", _post)
    assert client.generate("summarize this thread") == "A -> B -> C lineage."


def test_generate_raises_on_non_200(monkeypatch):
    client = GenerationClient(model="qwen2.5:7b", endpoint="http://localhost:11434")
    monkeypatch.setattr(client._http, "post",
                        lambda url, json=None: _Resp(500, text="boom"))
    with pytest.raises(GenerationError):
        client.generate("x")


def test_generate_raises_on_missing_field(monkeypatch):
    client = GenerationClient(model="qwen2.5:7b", endpoint="http://localhost:11434")
    monkeypatch.setattr(client._http, "post",
                        lambda url, json=None: _Resp(200, {}))
    with pytest.raises(GenerationError):
        client.generate("x")


def test_generate_wraps_transport_error(monkeypatch):
    client = GenerationClient(model="qwen2.5:7b", endpoint="http://localhost:11434")

    def _boom(url, json=None):
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(client._http, "post", _boom)
    with pytest.raises(GenerationError):
        client.generate("x")


def test_default_read_timeout_is_generous():
    client = GenerationClient(model="qwen2.5:7b")
    # Off-hot-path generation needs headroom for a multi-sentence 7B response.
    assert client._http.timeout.read == 120.0
    assert client._http.timeout.connect == 0.5
