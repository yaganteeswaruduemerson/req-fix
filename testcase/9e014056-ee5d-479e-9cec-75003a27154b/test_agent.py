
import pytest
from starlette.testclient import TestClient
import agent

@pytest.fixture(scope="module")
def client():
    """Fixture to provide a FastAPI test client for the agent app."""
    with TestClient(agent.app) as c:
        yield c

def test_integration_submit_requirement_endpoint_with_malformed_json(client):
    """
    Ensures that submitting malformed JSON to /submit-requirement triggers the validation_exception_handler.
    """
    # Malformed JSON: missing closing brace
    malformed_json = '{"requirement_description": "Write a function", "output_format": "code"'
    headers = {"Content-Type": "application/json"}
    response = client.post("/submit-requirement", data=malformed_json, headers=headers)
    assert response.status_code == 422
    data = response.json()
    assert data["success"] is False
    assert "Malformed JSON" in data["error"]
    assert "details" in data
    assert "tips" in data