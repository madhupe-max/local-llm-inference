from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.inference_compare import call_ollama, call_openai, compare_inference


class DummyResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_call_ollama_success(monkeypatch):
    def fake_post(url, json, timeout):
        assert url.endswith("/api/generate")
        assert json["prompt"] == "hello"
        return DummyResponse({"response": "local answer"})

    monkeypatch.setattr("src.inference_compare.requests.post", fake_post)
    assert call_ollama("hello") == "local answer"


def test_call_openai_requires_api_key():
    try:
        call_openai("hello", api_key=None)
    except ValueError as exc:
        assert "OPENAI_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError when api key is missing")


def test_call_openai_accepts_non_string_api_key(monkeypatch):
    def fake_post(url, headers, json, timeout):
        assert url.endswith("/chat/completions")
        assert headers["Authorization"] == "Bearer 12345"
        assert json["messages"][0]["content"] == "hello"
        return DummyResponse({"choices": [{"message": {"content": "cloud answer"}}]})

    monkeypatch.setattr("src.inference_compare.requests.post", fake_post)
    assert call_openai("hello", api_key=12345) == "cloud answer"


def test_compare_inference_uses_both_clients():
    results = compare_inference(
        "compare this",
        local_client=lambda prompt: f"local:{prompt}",
        cloud_client=lambda prompt: f"cloud:{prompt}",
    )

    assert results["local_response"] == "local:compare this"
    assert results["cloud_response"] == "cloud:compare this"
    assert results["local_latency_ms"] >= 0
    assert results["cloud_latency_ms"] >= 0
