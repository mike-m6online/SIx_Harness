from claude_mem.embed import EmbeddingClient


def test_embed_payload_sets_num_ctx(monkeypatch):
    captured = {}

    class FakeResp:
        status_code = 200

        def json(self):
            return {"embedding": [0.1] * 1024}

    client = EmbeddingClient()
    client._resolved_model = "qwen3-embedding:0.6b"

    def fake_post(url, json):
        captured.update(json)
        return FakeResp()

    monkeypatch.setattr(client._http, "post", fake_post)
    client.embed("hello")
    assert captured.get("options", {}).get("num_ctx", 0) >= 8192
