import asyncio as _asyncio

import time as _time
from observability.observability_wrapper import (
    trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
)
from config import settings as _obs_settings

import logging as _obs_startup_log
from contextlib import asynccontextmanager
from observability.instrumentation import initialize_tracer

_obs_startup_logger = _obs_startup_log.getLogger(__name__)

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {
    'content_safety_enabled': True,
    'runtime_enabled': True,
    'content_safety_severity_threshold': 3,
    'check_toxicity': True,
    'check_jailbreak': True,
    'check_pii_input': False,
    'check_credentials_output': True,
    'check_output': True,
    'check_toxic_code_output': True,
    'sanitize_pii': False
}

import logging
import json
from typing import Optional, Any, Dict
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationError, field_validator
from pathlib import Path

import openai
from config import Config

import black
import subprocess

# ========== SYSTEM PROMPT AND CONSTANTS ==========
SYSTEM_PROMPT = (
    "You are a professional Python code generation assistant. Your role is to interpret user requirements expressed in natural language and generate high-quality, efficient, and readable Python code that fulfills the specified task. \n\n"
    "Instructions:\n\n"
    "- Carefully analyze the user's requirement and clarify ambiguities if necessary.\n"
    "- Generate Python code that is well-structured, follows PEP8 style guidelines, and includes appropriate comments.\n"
    "- If the requirement is unclear or incomplete, politely request additional information.\n"
    "- Do not generate code for malicious, unethical, or unsafe purposes.\n"
    "- If the requirement cannot be fulfilled, provide a clear explanation and suggest next steps.\n\n"
    "Output Format:\n\n"
    "- Return only the Python code in a properly formatted code block.\n"
    "- If clarification is needed, ask a concise, professional follow-up question.\n"
    "- If unable to generate code, return a polite fallback message.\n\n"
    "Fallback Behavior:\n\n"
    "- If the requirement is unclear or cannot be fulfilled, respond with a message explaining the issue and request clarification or provide a generic code template."
)
OUTPUT_FORMAT = "- Python code in a markdown code block (```python ... ```)\n- If clarification is needed, a single follow-up question\n- If unable to generate code, a polite fallback message"
FALLBACK_RESPONSE = "I'm unable to generate Python code for the given requirement. Please provide more details or clarify your request."
FEW_SHOT_EXAMPLES = [
    "Write a Python function to calculate the factorial of a number.",
    "Create a script that reads a CSV file and prints the sum of a column named 'amount'."
]

VALIDATION_CONFIG_PATH = Config.VALIDATION_CONFIG_PATH or str(Path(__file__).parent / "validation_config.json")

# ========== LLM CLIENT INITIALIZATION ==========
@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    api_key = Config.AZURE_OPENAI_API_KEY
    if not api_key:
        raise ValueError("AZURE_OPENAI_API_KEY not configured")
    return openai.AsyncAzureOpenAI(
        api_key=api_key,
        api_version="2024-02-01",
        azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
    )

# ========== SANITIZER UTILITY ==========
import re as _re

_FENCE_RE = _re.compile(r"```(?:\w+)?\s*\n(.*?)```", _re.DOTALL)
_LONE_FENCE_START_RE = _re.compile(r"^```\w*$")
_WRAPPER_RE = _re.compile(
    r"^(?:"
    r"Here(?:'s| is)(?: the)? (?:the |your |a )?(?:code|solution|implementation|result|explanation|answer)[^:]*:\s*"
    r"|Sure[!,.]?\s*"
    r"|Certainly[!,.]?\s*"
    r"|Below is [^:]*:\s*"
    r")",
    _re.IGNORECASE,
)
_SIGNOFF_RE = _re.compile(
    r"^(?:Let me know|Feel free|Hope this|This code|Note:|Happy coding|If you)",
    _re.IGNORECASE,
)
_BLANK_COLLAPSE_RE = _re.compile(r"\n{3,}")

def _strip_fences(text: str, content_type: str) -> str:
    """Extract content from Markdown code fences."""
    fence_matches = _FENCE_RE.findall(text)
    if fence_matches:
        if content_type == "code":
            return "\n\n".join(block.strip() for block in fence_matches)
        for match in fence_matches:
            fenced_block = _FENCE_RE.search(text)
            if fenced_block:
                text = text[:fenced_block.start()] + match.strip() + text[fenced_block.end():]
        return text
    lines = text.splitlines()
    if lines and _LONE_FENCE_START_RE.match(lines[0].strip()):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()

def _strip_trailing_signoffs(text: str) -> str:
    """Remove conversational sign-off lines from the end of code output."""
    lines = text.splitlines()
    while lines and _SIGNOFF_RE.match(lines[-1].strip()):
        lines.pop()
    return "\n".join(lines).rstrip()

@with_content_safety(config=GUARDRAILS_CONFIG)
def sanitize_llm_output(raw: str, content_type: str = "code") -> str:
    """
    Generic post-processor that cleans common LLM output artefacts.
    Args:
        raw: Raw text returned by the LLM.
        content_type: 'code' | 'text' | 'markdown'.
    Returns:
        Cleaned string ready for validation, formatting, or direct return.
    """
    if not raw:
        return ""
    text = _strip_fences(raw.strip(), content_type)
    text = _WRAPPER_RE.sub("", text, count=1).strip()
    if content_type == "code":
        text = _strip_trailing_signoffs(text)
    return _BLANK_COLLAPSE_RE.sub("\n\n", text).strip()

# ========== INPUT/OUTPUT MODELS ==========
class SubmitRequirementRequest(BaseModel):
    requirement_description: str = Field(..., description="Natural language description of the Python code requirement (minimum 10 characters)")
    output_format: Optional[str] = Field("code", description="Desired output format (e.g., 'code', 'markdown'). Default is 'code'.")

    @field_validator("requirement_description")
    @classmethod
    def validate_requirement_description(cls, v):
        if not v or not isinstance(v, str) or len(v.strip()) < 10:
            raise ValueError("Requirement description must be at least 10 characters and not empty.")
        return v.strip()

class SubmitRequirementResponse(BaseModel):
    success: bool = Field(..., description="Indicates if code generation was successful")
    code: Optional[str] = Field(None, description="Generated Python code (markdown code block)")
    clarification: Optional[str] = Field(None, description="Clarification question if requirement is unclear")
    fallback: Optional[str] = Field(None, description="Fallback message if code cannot be generated")
    quality_report: Optional[str] = Field(None, description="Code quality analysis report")
    error: Optional[str] = Field(None, description="Error message if any")

# ========== COMPONENTS ==========

class RequirementValidator:
    """Ensures requirement descriptions are clear, actionable, and meet minimum criteria."""
    def validate(self, requirement_description: str) -> bool:
        if not requirement_description or not isinstance(requirement_description, str):
            return False
        if len(requirement_description.strip()) < 10:
            return False
        # Could add more advanced checks (e.g., LLM-based intent detection) here
        return True

class LLMService:
    """Handles prompt construction, interacts with Azure OpenAI GPT-4.1, manages few-shot examples, and processes LLM responses."""
    def __init__(self):
        self.client = None

    @with_content_safety(config=GUARDRAILS_CONFIG)
    @trace_agent(agent_name=_obs_settings.AGENT_NAME, project_name=_obs_settings.PROJECT_NAME)
    async def generate_code(self, prompt: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Calls Azure OpenAI GPT-4.1 to generate Python code.
        """
        if not self.client:
            self.client = get_llm_client()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT + "\n\nOutput Format: " + OUTPUT_FORMAT},
        ]
        # Add few-shot examples as user/assistant pairs if present
        for example in FEW_SHOT_EXAMPLES:
            messages.append({"role": "user", "content": example})
            messages.append({"role": "assistant", "content": "# Example code for: " + example + "\n# ..."})
        # User prompt
        messages.append({"role": "user", "content": prompt})
        _llm_kwargs = Config.get_llm_kwargs()
        _t0 = _time.time()
        try:
            response = await self.client.chat.completions.create(
                model=Config.LLM_MODEL or "gpt-4.1",
                messages=messages,
                **_llm_kwargs
            )
            content = response.choices[0].message.content
            try:
                trace_model_call(
                    provider="azure",
                    model_name=Config.LLM_MODEL or "gpt-4.1",
                    prompt_tokens=getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0,
                    latency_ms=int((_time.time() - _t0) * 1000),
                    response_summary=content[:200] if content else "",
                )
            except Exception:
                pass
            return content
        except Exception as e:
            logging.error(f"LLMService.generate_code error: {e}")
            raise

class CodeFormatter:
    """Formats generated code to ensure PEP8 compliance and readability."""
    def format_code(self, generated_code: str) -> str:
        try:
            # Remove markdown code fences if present
            code = sanitize_llm_output(generated_code, content_type="code")
            # Use black to format code
            mode = black.Mode()
            formatted_code = black.format_str(code, mode=mode)
            return formatted_code
        except Exception as e:
            logging.warning(f"CodeFormatter: Formatting failed, returning original code. Error: {e}")
            return generated_code

class CodeQualityChecker:
    """Analyzes code for best practices, detects common errors, and provides quality feedback."""
    def check_quality(self, generated_code: str) -> str:
        try:
            # Remove markdown code fences if present
            code = sanitize_llm_output(generated_code, content_type="code")
            # Run flake8 as a subprocess for static analysis
            import tempfile
            with tempfile.NamedTemporaryFile("w+", suffix=".py", delete=True) as tmp:
                tmp.write(code)
                tmp.flush()
                result = subprocess.run(
                    ["flake8", tmp.name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    return "No issues found. Code is PEP8 compliant."
                else:
                    return f"Code quality issues:\n{result.stdout.strip()}"
        except Exception as e:
            logging.warning(f"CodeQualityChecker: Quality check failed. Error: {e}")
            return "Code quality check could not be completed."

class ErrorHandler:
    """Manages errors, retries, fallback behaviors, and logging."""
    def handle_error(self, error_code: str, context: Optional[dict] = None) -> dict:
        if error_code == "INVALID_REQUIREMENT":
            return {
                "success": False,
                "clarification": "Could you please clarify or expand your requirement? It appears to be too brief or unclear.",
                "error": "Requirement description is invalid or unclear."
            }
        elif error_code == "CODE_GENERATION_ERROR":
            return {
                "success": False,
                "fallback": FALLBACK_RESPONSE,
                "error": "Code generation failed after multiple attempts."
            }
        else:
            return {
                "success": False,
                "fallback": FALLBACK_RESPONSE,
                "error": f"An unexpected error occurred: {error_code}"
            }

class AuditLogger:
    """Logs requests, responses, and errors for auditing and monitoring."""
    def __init__(self):
        self.logger = logging.getLogger("agent.audit")
        self.logger.setLevel(logging.INFO)

    def log_event(self, event_type: str, details: Any):
        try:
            self.logger.info(f"{event_type}: {details}")
        except Exception:
            pass

# ========== ORCHESTRATOR ==========
class AgentOrchestrator:
    """Coordinates validation, LLM, formatting, quality check, and error handling."""
    def __init__(self):
        self.validator = RequirementValidator()
        self.llm_service = LLMService()
        self.formatter = CodeFormatter()
        self.quality_checker = CodeQualityChecker()
        self.error_handler = ErrorHandler()
        self.audit_logger = AuditLogger()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_request(self, requirement_description: str, output_format: str = "code") -> dict:
        """
        Validates input, orchestrates LLM call, formatting, and quality checks.
        Returns dict for SubmitRequirementResponse.
        """
        async with trace_step(
            "validate_requirement",
            step_type="parse",
            decision_summary="Validate requirement description for clarity and completeness",
            output_fn=lambda r: f"valid={r}"
        ) as step:
            is_valid = self.validator.validate(requirement_description)
            step.capture(is_valid)
            if not is_valid:
                self.audit_logger.log_event("validation_failed", {"requirement": requirement_description})
                return self.error_handler.handle_error("INVALID_REQUIREMENT", {"requirement": requirement_description})

        # LLM code generation with retries
        code = None
        error = None
        for attempt in range(3):
            async with trace_step(
                f"llm_generate_code_attempt_{attempt+1}",
                step_type="llm_call",
                decision_summary="Call LLM to generate Python code",
                output_fn=lambda r: f"code_len={len(r) if r else 0}"
            ) as step:
                try:
                    prompt = requirement_description.strip()
                    raw_code = await self.llm_service.generate_code(prompt)
                    code = sanitize_llm_output(raw_code, content_type="code")
                    if code and len(code.strip()) > 0:
                        break
                except Exception as e:
                    error = str(e)
                    self.audit_logger.log_event("llm_error", {"error": error, "attempt": attempt+1})
                    await _asyncio.sleep(2 ** attempt)
        if not code or not code.strip():
            self.audit_logger.log_event("code_generation_failed", {"requirement": requirement_description, "error": error})
            return self.error_handler.handle_error("CODE_GENERATION_ERROR", {"requirement": requirement_description, "error": error})

        # Format code
        async with trace_step(
            "format_code",
            step_type="process",
            decision_summary="Format code for PEP8 compliance",
            output_fn=lambda r: f"formatted_len={len(r) if r else 0}"
        ) as step:
            formatted_code = self.formatter.format_code(code)
            step.capture(formatted_code)

        # Code quality check (non-blocking)
        async with trace_step(
            "check_quality",
            step_type="process",
            decision_summary="Check code quality with flake8",
            output_fn=lambda r: f"quality_report={r[:50] if r else ''}"
        ) as step:
            quality_report = self.quality_checker.check_quality(formatted_code)
            step.capture(quality_report)

        # Assemble response
        response = {
            "success": True,
            "code": f"```python\n{formatted_code.strip()}\n```",
            "quality_report": quality_report
        }
        self.audit_logger.log_event("code_generated", {
            "requirement": requirement_description,
            "output_format": output_format,
            "code_len": len(formatted_code),
            "quality_report": quality_report
        })
        return response

# ========== MAIN AGENT ==========
class PythonCodeGenerationAgent:
    """Main agent entry point."""
    def __init__(self):
        self.orchestrator = AgentOrchestrator()

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def submit_requirement(self, requirement_description: str, output_format: str = "code") -> dict:
        """
        Entry point for user requests; receives requirement and output format.
        Returns dict for SubmitRequirementResponse.
        """
        async with trace_step(
            "submit_requirement",
            step_type="process",
            decision_summary="Receive user requirement and orchestrate code generation",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            try:
                result = await self.orchestrator.process_request(requirement_description, output_format)
                step.capture(result)
                return result
            except Exception as e:
                logging.error(f"PythonCodeGenerationAgent.submit_requirement error: {e}")
                return {
                    "success": False,
                    "fallback": FALLBACK_RESPONSE,
                    "error": str(e)
                }

# ========== OBSERVABILITY LIFESPAN ==========
@asynccontextmanager
async def _obs_lifespan(application):
    """Initialise observability on startup, clean up on shutdown."""
    try:
        _obs_startup_logger.info('')
        _obs_startup_logger.info('========== Agent Configuration Summary ==========')
        _obs_startup_logger.info(f'Environment: {getattr(Config, "ENVIRONMENT", "N/A")}')
        _obs_startup_logger.info(f'Agent: {getattr(Config, "AGENT_NAME", "N/A")}')
        _obs_startup_logger.info(f'Project: {getattr(Config, "PROJECT_NAME", "N/A")}')
        _obs_startup_logger.info(f'LLM Provider: {getattr(Config, "MODEL_PROVIDER", "N/A")}')
        _obs_startup_logger.info(f'LLM Model: {getattr(Config, "LLM_MODEL", "N/A")}')
        _cs_endpoint = getattr(Config, 'AZURE_CONTENT_SAFETY_ENDPOINT', None)
        _cs_key = getattr(Config, 'AZURE_CONTENT_SAFETY_KEY', None)
        if _cs_endpoint and _cs_key:
            _obs_startup_logger.info('Content Safety: Enabled (Azure Content Safety)')
            _obs_startup_logger.info(f'Content Safety Endpoint: {_cs_endpoint}')
        else:
            _obs_startup_logger.info('Content Safety: Not Configured')
        _obs_startup_logger.info('Observability Database: Azure SQL')
        _obs_startup_logger.info(f'Database Server: {getattr(Config, "OBS_AZURE_SQL_SERVER", "N/A")}')
        _obs_startup_logger.info(f'Database Name: {getattr(Config, "OBS_AZURE_SQL_DATABASE", "N/A")}')
        _obs_startup_logger.info('===============================================')
        _obs_startup_logger.info('')
    except Exception as _e:
        _obs_startup_logger.warning('Config summary failed: %s', _e)

    _obs_startup_logger.info('')
    _obs_startup_logger.info('========== Content Safety & Guardrails ==========')
    if GUARDRAILS_CONFIG.get('content_safety_enabled'):
        _obs_startup_logger.info('Content Safety: Enabled')
        _obs_startup_logger.info(f'  - Severity Threshold: {GUARDRAILS_CONFIG.get("content_safety_severity_threshold", "N/A")}')
        _obs_startup_logger.info(f'  - Check Toxicity: {GUARDRAILS_CONFIG.get("check_toxicity", False)}')
        _obs_startup_logger.info(f'  - Check Jailbreak: {GUARDRAILS_CONFIG.get("check_jailbreak", False)}')
        _obs_startup_logger.info(f'  - Check PII Input: {GUARDRAILS_CONFIG.get("check_pii_input", False)}')
        _obs_startup_logger.info(f'  - Check Credentials Output: {GUARDRAILS_CONFIG.get("check_credentials_output", False)}')
    else:
        _obs_startup_logger.info('Content Safety: Disabled')
    _obs_startup_logger.info('===============================================')
    _obs_startup_logger.info('')

    _obs_startup_logger.info('========== Initializing Agent Services ==========')
    # 1. Observability DB schema (imports are inside function — only needed at startup)
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
        _obs_startup_logger.info('✓ Observability database connected')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Observability database connection failed (metrics will not be saved)')
    # 2. OpenTelemetry tracer (initialize_tracer is pre-injected at top level)
    try:
        _t = initialize_tracer()
        if _t is not None:
            _obs_startup_logger.info('✓ Telemetry monitoring enabled')
        else:
            _obs_startup_logger.warning('✗ Telemetry monitoring disabled')
    except Exception as _e:
        _obs_startup_logger.warning('✗ Telemetry monitoring failed to initialize')
    _obs_startup_logger.info('=================================================')
    _obs_startup_logger.info('')
    yield

# ========== FASTAPI APP ==========
app = FastAPI(lifespan=_obs_lifespan,

    title="Python Code Generation Assistant",
    description="Professional Python code generation assistant. Interprets natural language requirements and generates high-quality, PEP8-compliant Python code.",
    version=Config.SERVICE_VERSION if hasattr(Config, "SERVICE_VERSION") else "1.0.0",
    # SYNTAX-FIX: lifespan=_obs_lifespan
)

agent = PythonCodeGenerationAgent()

# ========== ERROR HANDLING ==========
@app.exception_handler(RequestValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Malformed JSON or invalid request parameters.",
            "details": exc.errors(),
            "tips": [
                "Ensure your JSON is well-formed (check for missing commas, colons, or quotes).",
                "All required fields must be present and non-empty.",
                "Text fields must not exceed 50,000 characters.",
                "If pasting code, wrap it in triple quotes or escape special characters."
            ]
        }
    )

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def pydantic_validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation failed.",
            "details": exc.errors(),
            "tips": [
                "Check that all required fields are present and valid.",
                "Text fields must not be empty or too short.",
                "If pasting code, wrap it in triple quotes or escape special characters."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "An unexpected error occurred.",
            "details": str(exc),
            "tips": [
                "Try again later.",
                "If the problem persists, contact support."
            ]
        }
    )

# ========== ENDPOINTS ==========

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/submit-requirement", response_model=SubmitRequirementResponse)
async def submit_requirement_endpoint(req: SubmitRequirementRequest):
    """
    Submit a Python code requirement and receive generated code.
    """
    result = await agent.submit_requirement(
        requirement_description=req.requirement_description,
        output_format=req.output_format or "code"
    )
    # Sanitize LLM output before returning
    if result.get("code"):
        result["code"] = sanitize_llm_output(result["code"], content_type="code")
    return result

# ========== __main__ ENTRYPOINT ==========
async def _run_agent():
    """Entrypoint: runs the agent with observability (trace collection only)."""
    import uvicorn

    # Unified logging config — routes uvicorn, agent, and observability through
    # the same handler so all telemetry appears in a single consistent stream.
    _LOG_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(name)s: %(message)s",
                "use_colors": None,
            },
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn":        {"handlers": ["default"], "level": "INFO", "propagate": False},
            "uvicorn.error":  {"level": "INFO"},
            "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
            "agent":          {"handlers": ["default"], "level": "INFO", "propagate": False},
            "__main__":       {"handlers": ["default"], "level": "INFO", "propagate": False},
            "observability": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "config": {"handlers": ["default"], "level": "INFO", "propagate": False},
            "azure":   {"handlers": ["default"], "level": "WARNING", "propagate": False},
            "urllib3": {"handlers": ["default"], "level": "WARNING", "propagate": False},
        },
    }

    config = uvicorn.Config(
        "agent:app",
        host="0.0.0.0",
        port=8080,
        reload=False,
        log_level="info",
        log_config=_LOG_CONFIG,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    _asyncio.run(_run_agent())