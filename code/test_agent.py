
import pytest
import asyncio
import types
import json
from unittest.mock import patch, MagicMock, AsyncMock

import agent

from agent import SubmitRequirementRequest, RequirementValidator, LLMService, CodeFormatter, CodeQualityChecker, ErrorHandler, AgentOrchestrator, PythonCodeGenerationAgent, sanitize_llm_output, app

# For FastAPI endpoint tests
from fastapi.testclient import TestClient

@pytest.fixture
def test_client():
    return TestClient(app)

def test_SubmitRequirementRequest_validation_valid_input():
    """Validates that SubmitRequirementRequest accepts a proper requirement_description and output_format."""
    req = SubmitRequirementRequest(
        requirement_description="Write a function to add two numbers.",
        output_format="code"
    )
    assert req.requirement_description == "Write a function to add two numbers."
    assert req.output_format == "code"

def test_SubmitRequirementRequest_validation_invalid_input():
    """Checks that SubmitRequirementRequest raises a ValidationError for short or empty requirement_description."""
    with pytest.raises(agent.ValidationError) as excinfo:
        SubmitRequirementRequest(
            requirement_description="short",
            output_format="code"
        )
    assert "at least 10 characters" in str(excinfo.value)

def test_RequirementValidator_validate_returns_true_for_valid_input():
    """Ensures RequirementValidator.validate returns True for a valid requirement_description."""
    validator = RequirementValidator()
    result = validator.validate("Write a Python script to reverse a string.")
    assert result is True

def test_RequirementValidator_validate_returns_false_for_invalid_input():
    """Ensures RequirementValidator.validate returns False for empty or too short requirement_description."""
    validator = RequirementValidator()
    result = validator.validate("short")
    assert result is False

@pytest.mark.asyncio
async def test_LLMService_generate_code_returns_code_for_valid_prompt():
    """Tests that LLMService.generate_code returns a non-empty string for a valid prompt."""
    llm_service = LLMService()
    fake_response = MagicMock()
    fake_response.choices = [MagicMock()]
    fake_response.choices[0].message.content = "def add(a, b):\n    return a + b"
    fake_response.usage = MagicMock()
    fake_response.usage.prompt_tokens = 10
    fake_response.usage.completion_tokens = 10

    with patch("agent.get_llm_client") as mock_client_factory:
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=fake_response)
        mock_client_factory.return_value = mock_client
        llm_service.client = None  # force re-init
        result = await llm_service.generate_code("Write a Python function to add two numbers.")
        assert isinstance(result, str)
        assert result.strip() != ""
        assert "def " in result

@pytest.mark.asyncio
async def test_AgentOrchestrator_process_request_valid_requirement():
    """Integration test for AgentOrchestrator.process_request with a valid requirement."""
    orchestrator = AgentOrchestrator()
    fake_code = "def add(a, b):\n    return a + b"
    with patch("agent.LLMService.generate_code", new=AsyncMock(return_value=fake_code)):
        with patch("agent.sanitize_llm_output", side_effect=lambda x, content_type="code": x):
            response = await orchestrator.process_request(
                requirement_description="Write a Python function to add two numbers.",
                output_format="code"
            )
    assert response["success"] is True
    assert "def " in response["code"]
    assert response.get("quality_report") is not None

@pytest.mark.asyncio
async def test_AgentOrchestrator_process_request_invalid_requirement():
    """Integration test for AgentOrchestrator.process_request with an invalid requirement."""
    orchestrator = AgentOrchestrator()
    response = await orchestrator.process_request(
        requirement_description="short",
        output_format="code"
    )
    assert response["success"] is False
    assert response.get("clarification") is not None

@pytest.mark.asyncio
async def test_PythonCodeGenerationAgent_submit_requirement_valid_flow():
    """Functional test for the main agent flow with a valid requirement."""
    agent_instance = PythonCodeGenerationAgent()
    fake_code = "def add(a, b):\n    return a + b"
    with patch("agent.LLMService.generate_code", new=AsyncMock(return_value=fake_code)):
        with patch("agent.sanitize_llm_output", side_effect=lambda x, content_type="code": x):
            response = await agent_instance.submit_requirement(
                requirement_description="Write a Python function to add two numbers.",
                output_format="code"
            )
    assert response["success"] is True
    assert "def " in response["code"]
    assert response.get("quality_report") is not None

@pytest.mark.asyncio
async def test_PythonCodeGenerationAgent_submit_requirement_unclear_requirement():
    """Functional test for the main agent flow with an unclear requirement."""
    agent_instance = PythonCodeGenerationAgent()
    response = await agent_instance.submit_requirement(
        requirement_description="short",
        output_format="code"
    )
    assert response["success"] is False
    assert response.get("clarification") is not None

def test_CodeFormatter_format_code_returns_pep8_code():
    """Unit test for CodeFormatter.format_code to ensure output is PEP8-compliant."""
    formatter = CodeFormatter()
    code = "def add(a,b):return a+b"
    formatted = formatter.format_code(code)
    assert isinstance(formatted, str)
    assert "def add(a, b):" in formatted
    # PEP8: should have newline after function def and proper indentation
    assert "\n" in formatted

def test_CodeQualityChecker_check_quality_returns_no_issues_for_clean_code():
    """Unit test for CodeQualityChecker.check_quality with PEP8-compliant code."""
    checker = CodeQualityChecker()
    code = "def add(a, b):\n    return a + b"
    # Patch subprocess.run to simulate flake8 returning no issues
    with patch("subprocess.run") as mock_run:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = ""
        mock_run.return_value = mock_result
        result = checker.check_quality(code)
    assert "No issues found" in result

def test_ErrorHandler_handle_error_returns_clarification_for_invalid_requirement():
    """Unit test for ErrorHandler.handle_error with INVALID_REQUIREMENT."""
    handler = ErrorHandler()
    response = handler.handle_error("INVALID_REQUIREMENT", {"requirement": "short"})
    assert response["success"] is False
    assert response.get("clarification") is not None
    assert "invalid or unclear" in response.get("error", "")

def test_ErrorHandler_handle_error_returns_fallback_for_code_generation_error():
    """Unit test for ErrorHandler.handle_error with CODE_GENERATION_ERROR."""
    handler = ErrorHandler()
    response = handler.handle_error("CODE_GENERATION_ERROR", {"requirement": "valid but fails"})
    assert response["success"] is False
    assert response.get("fallback") is not None
    assert "failed after multiple attempts" in response.get("error", "")

def test_sanitize_llm_output_removes_markdown_fences_and_signoffs():
    """Unit test for sanitize_llm_output to ensure it strips markdown fences and sign-off lines."""
    raw = "```python\ndef add(a, b):\n    return a + b\n```\nHappy coding!"
    cleaned = agent.sanitize_llm_output(raw, content_type="code")
    assert "```" not in cleaned
    assert "Happy coding!" not in cleaned
    assert "def add(a, b):" in cleaned

@pytest.mark.asyncio
async def test_api_submit_requirement_endpoint_valid_request(test_client):
    """Functional test for the /submit-requirement endpoint with a valid requirement."""
    fake_code = "def add(a, b):\n    return a + b"
    with patch("agent.LLMService.generate_code", new=AsyncMock(return_value=fake_code)):
        with patch("agent.sanitize_llm_output", side_effect=lambda x, content_type="code": x):
            payload = {
                "requirement_description": "Write a Python function to add two numbers.",
                "output_format": "code"
            }
            response = test_client.post("/submit-requirement", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "def " in data["code"]

@pytest.mark.asyncio
async def test_api_submit_requirement_endpoint_invalid_request(test_client):
    """Functional test for the /submit-requirement endpoint with an invalid requirement."""
    payload = {
        "requirement_description": "short",
        "output_format": "code"
    }
    response = test_client.post("/submit-requirement", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is False
    assert data.get("clarification") is not None

def test_api_health_endpoint_returns_ok(test_client):
    """Functional test for the /health endpoint."""
    response = test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"