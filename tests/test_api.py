"""Tests for FastAPI HTTP endpoints."""

from fastapi.testclient import TestClient
from app.api.server import create_app


def test_api_health_endpoint():
    app = create_app()
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["chunks_indexed"] > 0


def test_api_query_endpoint():
    app = create_app()
    with TestClient(app) as client:
        payload = {
            "question": "What is the election timeout duration in Raft consensus?",
            "strategy": "hybrid_rerank_verify",
        }
        response = client.post("/api/v1/query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "citations" in data
        assert "verifications" in data
        assert "verification" in data
        assert "timing_ms" in data


def test_api_documents_ingest_endpoint():
    app = create_app()
    with TestClient(app) as client:
        payload = {
            "content": "Vector databases index high-dimensional embeddings using HNSW graphs for approximate nearest neighbor search.",
            "title": "Vector Databases",
            "source": "api_test.txt",
        }
        response = client.post("/api/v1/documents/ingest", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["chunks_created"] >= 1
        assert "Vector Databases" in data["title"]
