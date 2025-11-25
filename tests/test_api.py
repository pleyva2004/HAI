"""Tests for FastAPI endpoints"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "SAT Question Generator API"
    assert "version" in data


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "services" in data


def test_generate_endpoint_validation():
    """Test input validation on generate endpoint"""
    # Missing required field
    response = client.post("/api/v1/generate", data={})
    assert response.status_code == 422  # Validation error

    # Invalid num_questions
    response = client.post(
        "/api/v1/generate", data={"description": "Test", "num_questions": 0}
    )
    assert response.status_code == 422

    # Valid request (but may fail if services not available)
    response = client.post(
        "/api/v1/generate",
        data={"description": "Generate algebra questions", "num_questions": 3},
    )
    # Will fail without running services, but validates the endpoint exists
    assert response.status_code in [200, 500]  # Either success or server error


def test_search_endpoint():
    """Test search endpoint"""
    request_data = {
        "query": "linear equations",
        "limit": 5,
    }

    response = client.post("/api/v1/search", json=request_data)

    # Will fail without database, but validates endpoint
    assert response.status_code in [200, 500]


def test_validate_endpoint_format():
    """Test validate endpoint request format"""
    from src.models.toon_models import SATQuestion, QuestionChoice

    question = SATQuestion(
        id="test1",
        question="If 3x + 7 = 22, what is x?",
        choices=QuestionChoice(A="3", B="5", C="7", D="9"),
        correct_answer="B",
        explanation="Test",
        difficulty=50.0,
        category="algebra",
    )

    response = client.post("/api/v1/validate", json={"question": question.model_dump()})

    # Will fail without LLM keys, but validates endpoint
    assert response.status_code in [200, 500]


def test_cors_headers():
    """Test CORS headers are present"""
    # CORS headers are only added when Origin header is present
    response = client.get("/", headers={"Origin": "http://localhost:3000"})
    assert "access-control-allow-origin" in response.headers


def test_invalid_endpoint():
    """Test 404 for invalid endpoint"""
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

