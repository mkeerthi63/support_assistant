from types import SimpleNamespace

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


def _fake_completion(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeGroq:
    """Returns queued responses in order and records every request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _fake_completion(self.responses.pop(0))


def test_policy_question_returns_sources():
    resp = client.post("/ask", json={"query": "What is the return policy for damaged items?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["sources"]
    assert 0.0 <= body["confidence"] <= 1.0


def test_general_question_is_not_answered_from_policies():
    body = client.post("/ask", json={"query": "What is the capital of France?"}).json()
    assert body["sources"] == []
    assert "only answer questions about Zepto policies" in body["answer"]


def test_llm_answer_parses_valid_json(monkeypatch):
    fake = FakeGroq(['{"answer": "Within 7 days.", "sources": ["doc_02"], "confidence": 0.9}'])
    monkeypatch.setattr(main, "_get_groq_client", lambda: fake)
    docs = [{"id": "doc_02", "text": "...", "distance": 0.3}]
    resp = main.llm_answer("return window?", docs)
    assert resp.answer == "Within 7 days."
    assert resp.sources == ["doc_02"]
    assert len(fake.calls) == 1


def test_llm_answer_retries_after_invalid_json(monkeypatch):
    fake = FakeGroq([
        '{"answer": "x", "sources": [], "confidence": 7}',
        '{"answer": "Fixed.", "sources": ["doc_02"], "confidence": 0.8}',
    ])
    monkeypatch.setattr(main, "_get_groq_client", lambda: fake)
    resp = main.llm_answer("q", [{"id": "doc_02", "text": "...", "distance": 0.3}])
    assert resp.answer == "Fixed."
    assert len(fake.calls) == 2
    assert "failed validation" in fake.calls[1]["messages"][-1]["content"]


def test_llm_answer_falls_back_after_max_retries(monkeypatch):
    fake = FakeGroq(["not json"] * (main.MAX_RETRIES + 1))
    monkeypatch.setattr(main, "_get_groq_client", lambda: fake)
    docs = [{"id": "doc_08", "text": "Support is 24/7 via chat.", "distance": 0.4}]
    resp = main.llm_answer("q", docs)
    assert resp.answer.startswith("Based on the retrieved context")
    assert len(fake.calls) == main.MAX_RETRIES + 1


def test_llm_answer_drops_hallucinated_sources(monkeypatch):
    fake = FakeGroq(['{"answer": "a", "sources": ["doc_02", "doc_99"], "confidence": 0.5}'])
    monkeypatch.setattr(main, "_get_groq_client", lambda: fake)
    resp = main.llm_answer("q", [{"id": "doc_02", "text": "...", "distance": 0.3}])
    assert resp.sources == ["doc_02"]
