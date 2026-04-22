
import pytest
from starlette.testclient import TestClient
from agent import app

def test_health_check_endpoint():
    """Functional test: Ensures the /health endpoint returns status ok."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"