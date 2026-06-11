"""
services/evaluator.py

Responsibility:
    - Evaluate a candidate CV against job requirements using Gemini.
    - Build prompt templates combining CV text + retrieved RAG context.
    - Request structured JSON responses (score, decision, strengths, weaknesses, summary).
    - Parse and validate Gemini output, transforming into typed results.

Out of scope:
    - FastAPI routes.
    - PDF parsing or text extraction.
    - ChromaDB / vector store operations.
    - Email or notification integrations.

Dependencies:
    - langchain 1.3.4
    - langchain-google-genai
    - pydantic >= 2.0
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pydantic models for structured Gemini output
# ---------------------------------------------------------------------------


class HiringDecision(str, Enum):
    """Possible hiring recommendations."""

    YES = "Yes"
    NO = "No"
    CONSIDER = "Consider"


class EvaluationResultModel(BaseModel):
    """
    Structured output that Gemini must return as a JSON object.

    This model is used with LangChain's with_structured_output() to
    guarantee a type-safe response.
    """

    score: int = Field(
        description="Numerical score from 0 to 100 indicating candidate fit.",
        ge=0,
        le=100,
    )
    decision: HiringDecision = Field(
        description="Final hiring recommendation: Yes, No, or Consider."
    )
    strengths: list[str] = Field(
        description="List of the candidate's strongest relevant skills or experiences.",
        min_items=1,
        max_items=5,
    )
    weaknesses: list[str] = Field(
        description="List of critical missing skills or gaps (max 4 items).",
        min_items=0,
        max_items=4,
    )
    summary: str = Field(
        description="One-paragraph (2-4 sentences) justification of the evaluation.",
        max_length=500,
    )


# ---------------------------------------------------------------------------
# Service-level exceptions
# ---------------------------------------------------------------------------


class EvaluatorServiceError(Exception):
    """Base exception for all evaluator failures."""


class PromptBuildError(EvaluatorServiceError):
    """Raised when the prompt template cannot be built or formatted."""


class GeminiCallError(EvaluatorServiceError):
    """Raised when the Gemini API call fails."""


class OutputParsingError(EvaluatorServiceError):
    """Raised when Gemini's response cannot be parsed or validated."""


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """
    Immutable result returned by :meth:`EvaluatorService.evaluate`.

    Attributes:
        score:          Numeric fit score (0–100).
        decision:       Hiring recommendation.
        strengths:      List of strengths.
        weaknesses:     List of weaknesses.
        summary:        Justification text.
        evaluated_at:   UTC timestamp of evaluation.
        raw_response:   Optional raw JSON string for debugging.
    """

    score: int
    decision: HiringDecision
    strengths: list[str]
    weaknesses: list[str]
    summary: str
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_response: str | None = None


# ---------------------------------------------------------------------------
# Core service
# ---------------------------------------------------------------------------


class EvaluatorService:
    """
    Orchestrates LLM‑based candidate evaluation using Gemini.

    The service is stateless and thread‑safe. It expects:
        - CV text (extracted via CVService)
        - RAG context (retrieved from knowledge base, e.g., job description)

    A singleton instance can be shared via FastAPI dependency injection.

    Args:
        llm: LangChain chat model supporting structured output.
             Defaults to Gemini 1.5 Flash.
        system_prompt: Optional custom system prompt.
    """

    DEFAULT_SYSTEM_PROMPT = """You are a strict and objective technical recruiter for HireMind AI.

CRITICAL RULES — you must follow these exactly:
1. Read the CV text carefully word by word. Extract ONLY skills and experiences that are EXPLICITLY written in the CV.
2. NEVER invent, assume, or infer skills that are not directly stated in the CV text.
3. Compare the extracted skills against the job requirements one by one.
4. A skill counts as present ONLY if it appears literally in the CV (e.g. "RAG" counts only if "RAG" or "Retrieval-Augmented Generation" is written in the CV).
5. Score honestly: a candidate missing 60% of required skills must score below 50.
6. Respond ONLY with valid JSON. No markdown, no explanation outside the JSON."""

    def __init__(
        self,
        llm: BaseChatModel | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm or self._default_llm()
        self._system_prompt = system_prompt or self.DEFAULT_SYSTEM_PROMPT
        self._parser = PydanticOutputParser(pydantic_object=EvaluationResultModel)

        # Create structured LLM (LangChain 1.3.4 method)
        self._structured_llm = self._llm.with_structured_output(
            EvaluationResultModel,
            method="json_mode",  # Forces JSON, works reliably with Gemini
        )

    @staticmethod
    def _default_llm() -> ChatGoogleGenerativeAI:
        """
        Instantiate a default Gemini model.

        Expects environment variable GOOGLE_API_KEY to be set.
        Uses gemini-1.5-flash for low latency and cost.
        """
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY not found. "
                "Add it to your .env file: GOOGLE_API_KEY=your_key_here"
            )
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
            max_retries=2,
            timeout=30,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, cv_text: str, rag_context: str) -> EvaluationResult:
        """
        Evaluate a candidate CV against job requirements.

        Args:
            cv_text: Plain text extracted from the candidate's CV.
            rag_context: Retrieved knowledge (job description, company info, etc.).

        Returns:
            EvaluationResult containing score, decision, strengths, weaknesses, summary.

        Raises:
            TypeError:         If inputs are not strings.
            PromptBuildError:  If prompt formatting fails.
            GeminiCallError:   If Gemini API request fails.
            OutputParsingError:If response is not valid JSON or fails validation.
        """
        if not isinstance(cv_text, str):
            raise TypeError(f"cv_text must be str, got {type(cv_text).__name__!r}.")
        if not isinstance(rag_context, str):
            raise TypeError(
                f"rag_context must be str, got {type(rag_context).__name__!r}."
            )
        if not cv_text.strip():
            raise ValueError("cv_text cannot be empty.")
        if not rag_context.strip():
            logger.warning("evaluate: rag_context is empty. Evaluation may be incomplete.")

        logger.info(
            "EvaluatorService.evaluate: starting  cv_chars=%d  rag_chars=%d",
            len(cv_text),
            len(rag_context),
        )

        # Build prompt messages
        messages = self._build_messages(cv_text, rag_context)

        # Call Gemini with structured output
        try:
            response_model: EvaluationResultModel = self._structured_llm.invoke(messages)
        except Exception as exc:
            logger.error("Gemini call failed: %s", exc, exc_info=True)
            raise GeminiCallError(f"Gemini API error: {exc}") from exc

        # Convert Pydantic model to our internal dataclass
        result = EvaluationResult(
            score=response_model.score,
            decision=response_model.decision,
            strengths=response_model.strengths,
            weaknesses=response_model.weaknesses,
            summary=response_model.summary,
        )

        logger.info(
            "EvaluatorService.evaluate: done  score=%d  decision=%s",
            result.score,
            result.decision.value,
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_messages(self, cv_text: str, rag_context: str) -> list:
        """
        Build the chat messages for the LLM.

        Returns a list containing:
            - SystemMessage with the system prompt
            - HumanMessage with the formatted user prompt (CV + RAG context)
        """
        user_prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "human",
                    """
## JOB REQUIREMENTS (from knowledge base):
{rag_context}

## CANDIDATE CV (extracted text — read carefully):
{cv_text}

## YOUR TASK:

Step 1 — Extract from the CV:
List every technical skill, tool, framework, and technology explicitly mentioned in the CV text above.

Step 2 — Compare against requirements:
For each required skill in the job requirements, check if it appears in the CV.
Mark it as: PRESENT, PARTIALLY PRESENT, or MISSING.

Step 3 — Score:
- Count required skills that are PRESENT or PARTIALLY PRESENT vs total required skills.
- Score = (matched / total_required) * 100, rounded to nearest integer.
- If fewer than 40% of required skills match → score must be below 40, decision = "No"
- If 40-70% match → score 40-69, decision = "Consider"
- If more than 70% match → score 70-100, decision = "Yes"

Step 4 — Return this exact JSON (no extra text):
{{
  "score": <integer 0-100>,
  "decision": <"Yes" | "No" | "Consider">,
  "strengths": [<list of skills from CV that match job requirements, max 5>],
  "weaknesses": [<list of required skills MISSING from CV, max 4>],
  "summary": "<2-3 sentences explaining the score based on specific skill matches and gaps, max 500 chars>"
}}

IMPORTANT: strengths must only contain skills that are WRITTEN IN THE CV. Do not invent skills.
""",
                )
            ]
        )
        formatted = user_prompt_template.invoke(
            {"rag_context": rag_context, "cv_text": cv_text}
        )
        # Return as list of BaseMessages (SystemMessage + HumanMessage)
        return [
            SystemMessage(content=self._system_prompt),
            formatted.to_messages()[0],  # the HumanMessage
        ]


# ---------------------------------------------------------------------------
# FastAPI dependency factory
# ---------------------------------------------------------------------------


# Lazy singleton — ينشأ عند أول طلب، ليس عند import
_shared_evaluator_service: "EvaluatorService | None" = None


def get_evaluator_service() -> EvaluatorService:
    """FastAPI dependency — returns the shared EvaluatorService singleton (lazy init)."""
    global _shared_evaluator_service
    if _shared_evaluator_service is None:
        _shared_evaluator_service = EvaluatorService()
    return _shared_evaluator_service