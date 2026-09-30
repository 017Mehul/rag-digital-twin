from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.index as api


class _Store:
    def __init__(self, size=1):
        self.size = size

    def __len__(self):
        return self.size


class _Response:
    response_text = "Grounded answer"
    sources = ["guide.txt"]
    confidence_score = 0.8
    context_used = True
    model_used = "test-model"
    generation_time = 0.01


class _Pipeline:
    vector_store = _Store()

    def query(self, query, k=None, threshold=None):
        return _Response()


def test_query_api_returns_grounded_payload(monkeypatch):
    monkeypatch.setattr(api, "get_session_pipeline", lambda session_id: _Pipeline())
    client = TestClient(api.app)
    client.cookies.set("rag_session", "test-session")
    response = client.post("/api/query", json={"query": "What is this?"})
    assert response.status_code == 200
    body = response.json()
    assert body["response_text"] == "Grounded answer"
    assert body["sources"] == ["guide.txt"]


def test_query_api_rejects_empty_knowledge_base(monkeypatch):
    class EmptyPipeline:
        vector_store = _Store(0)

    monkeypatch.setattr(api, "get_session_pipeline", lambda session_id: EmptyPipeline())
    client = TestClient(api.app)
    client.cookies.set("rag_session", "test-session")
    response = client.post("/api/query", json={"query": "What is this?"})
    assert response.status_code == 400
    assert "No documents" in response.json()["detail"]


def test_query_api_requires_session(monkeypatch):
    monkeypatch.setattr(api, "get_session_pipeline", lambda session_id: None)
    client = TestClient(api.app)
    response = client.post("/api/query", json={"query": "What is this?"})
    assert response.status_code == 400
    assert "session has expired" in response.json()["detail"]


def test_query_api_validates_query_length():
    client = TestClient(api.app)
    response = client.post("/api/query", json={"query": ""})
    assert response.status_code == 422


def test_query_api_rejects_overlong_query():
    client = TestClient(api.app)
    response = client.post("/api/query", json={"query": "x" * 4001})
    assert response.status_code == 422


def test_query_api_rate_limit(monkeypatch):
    api._requests_by_client.clear()
    monkeypatch.setenv("RAG_QUERY_RATE_LIMIT", "1")
    monkeypatch.setattr(api, "get_session_pipeline", lambda session_id: _Pipeline())
    client = TestClient(api.app)
    client.cookies.set("rag_session", "test-session")
    assert client.post("/api/query", json={"query": "one"}).status_code == 200
    assert client.post("/api/query", json={"query": "two"}).status_code == 429


def test_api_responses_are_not_cacheable(monkeypatch):
    monkeypatch.setattr(api, "get_session_pipeline", lambda session_id: None)
    response = TestClient(api.app).get("/api/health")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["pragma"] == "no-cache"


def test_ingest_rate_limit_is_separate_from_query_limit(monkeypatch):
    api._requests_by_client.clear()
    monkeypatch.setenv("RAG_INGEST_RATE_LIMIT", "1")
    request = SimpleNamespace(client=SimpleNamespace(host="test-client"))
    api._rate_limit(request, "ingest", 1)
    try:
        api._rate_limit(request, "ingest", 1)
        assert False, "expected rate limit"
    except Exception as exc:
        assert getattr(exc, "status_code", None) == 429
