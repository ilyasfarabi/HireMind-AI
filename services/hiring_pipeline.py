"""
services/hiring_pipeline.py

Responsibility:
    - Orchestrate the complete candidate evaluation workflow.
    - Coordinate CVService, RAGService, EvaluatorService, and DatabaseManager.
    - Receive raw PDF bytes.
    - Extract CV text.
    - Retrieve relevant job requirements from RAG.
    - Evaluate the candidate with Gemini.
    - Persist the result to SQLite.
    - Return the final evaluation result.

Out of scope:
    - FastAPI routes.
    - PDF parsing implementation (delegated to CVService).
    - ChromaDB implementation (delegated to RAGService).
    - Gemini prompt implementation (delegated to EvaluatorService).
    - SQLite schema (delegated to DatabaseManager).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from services.cv_service import CVExtractionError, CVService, CVExtractionResult
from services.database import DatabaseManager, EvaluationRecord, PersistenceError
from services.evaluator import (
    EvaluatorService,
    EvaluationResult,
    GeminiCallError,
    OutputParsingError,
)
from services.rag import RAGService, SearchError, SearchResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class HiringPipelineError(Exception):
    """Base exception for all hiring pipeline failures."""


class CVExtractionFailed(HiringPipelineError):
    """Raised when CV text extraction fails."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


class ContextRetrievalFailed(HiringPipelineError):
    """Raised when RAG context retrieval fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


class EvaluationFailed(HiringPipelineError):
    """Raised when Gemini evaluation fails."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HiringPipelineResult:
    """
    Final result of the orchestrated evaluation process.

    Attributes:
        cv_extraction:      Result from CVService (extracted text + metadata).
        rag_context:        Retrieved knowledge context (as a single string).
        evaluation:         Evaluation result from EvaluatorService.
        job_query:          The query used to retrieve job requirements.
        evaluation_id:      Primary key of the persisted SQLite record (None if not saved).
    """

    cv_extraction: CVExtractionResult
    rag_context: str
    evaluation: EvaluationResult
    job_query: str
    evaluation_id: int | None = None

    @property
    def final_score(self) -> int:
        """Convenience property to get the numeric score."""
        return self.evaluation.score

    @property
    def hiring_decision(self) -> str:
        """Convenience property to get the hiring recommendation."""
        return self.evaluation.decision.value

    def to_dict(self) -> dict[str, Any]:
        """Convert the entire result to a JSON-serializable dictionary."""
        return {
            "evaluation_id": self.evaluation_id,
            "cv": {
                "char_count": self.cv_extraction.char_count,
                "extracted_at": self.cv_extraction.extracted_at.isoformat(),
                "text": self.cv_extraction.text,
            },
            "rag_context": self.rag_context,
            "evaluation": {
                "score": self.evaluation.score,
                "decision": self.evaluation.decision.value,
                "strengths": self.evaluation.strengths,
                "weaknesses": self.evaluation.weaknesses,
                "summary": self.evaluation.summary,
                "evaluated_at": self.evaluation.evaluated_at.isoformat(),
            },
            "job_query": self.job_query,
        }


# ---------------------------------------------------------------------------
# Orchestration service
# ---------------------------------------------------------------------------


class HiringPipeline:
    """
    Orchestrates the end-to-end candidate evaluation.

    The pipeline is stateless and thread-safe. It depends on four injected
    services, making it easy to unit-test each component in isolation.

    Args:
        cv_service:        Service to extract text from PDF bytes.
        rag_service:       Service to retrieve job requirements from the knowledge base.
        evaluator_service: Service that uses Gemini to score the candidate.
        db_manager:        DatabaseManager for persisting results to SQLite.
                           Pass None to skip persistence (useful for testing).

    Example::

        pipeline = HiringPipeline(cv_service, rag_service, evaluator_service, db_manager)
        result   = pipeline.run(pdf_bytes, job_query="AI Engineer")
        print(result.final_score, result.evaluation_id)
    """

    def __init__(
        self,
        cv_service: CVService,
        rag_service: RAGService,
        evaluator_service: EvaluatorService,
        db_manager: DatabaseManager | None = None,
    ) -> None:
        self._cv_service        = cv_service
        self._rag_service       = rag_service
        self._evaluator_service = evaluator_service
        self._db_manager        = db_manager

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, pdf_bytes: bytes, job_query: str) -> HiringPipelineResult:
        """
        Execute the complete evaluation pipeline.

        Steps:
        1. Extract plain text from the uploaded PDF CV.
        2. Retrieve relevant job requirements from the RAG knowledge base.
        3. Evaluate the candidate using Gemini.
        4. Persist the result to SQLite (if db_manager is configured).
        5. Return the packaged result.

        Args:
            pdf_bytes: Raw binary content of the CV (PDF format).
            job_query: Search query used to retrieve job requirements
                       (e.g., "AI Engineer job description").

        Returns:
            HiringPipelineResult containing all intermediate and final data,
            including the SQLite record ID (evaluation_id) if persistence succeeded.

        Raises:
            TypeError:              If inputs have incorrect types.
            ValueError:             If inputs are empty.
            CVExtractionFailed:     If CV text extraction fails.
            ContextRetrievalFailed: If RAG search fails.
            EvaluationFailed:       If Gemini evaluation fails.
        """
        self._validate_inputs(pdf_bytes, job_query)

        logger.info(
            "HiringPipeline: starting  job_query='%s'  pdf_size=%d bytes",
            job_query,
            len(pdf_bytes),
        )

        # Step 1: Extract CV text
        cv_result = self._extract_cv_text(pdf_bytes)

        # Step 2: Retrieve RAG context
        rag_context = self._retrieve_context(job_query)

        # Step 3: Evaluate with Gemini
        evaluation_result = self._evaluate(cv_result.text, rag_context)

        # Step 4: Persist to SQLite
        evaluation_id = self._persist(
            cv_result=cv_result,
            rag_context=rag_context,
            evaluation_result=evaluation_result,
            job_query=job_query,
        )

        # Step 5: Return result
        pipeline_result = HiringPipelineResult(
            cv_extraction=cv_result,
            rag_context=rag_context,
            evaluation=evaluation_result,
            job_query=job_query,
            evaluation_id=evaluation_id,
        )

        logger.info(
            "HiringPipeline: complete  score=%d  decision=%s  db_id=%s",
            pipeline_result.final_score,
            pipeline_result.hiring_decision,
            evaluation_id or "not persisted",
        )

        return pipeline_result

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _extract_cv_text(self, pdf_bytes: bytes) -> CVExtractionResult:
        """
        Delegate to CVService and translate its exceptions.

        Raises:
            CVExtractionFailed: On any CVService failure.
        """
        try:
            return self._cv_service.process_cv(pdf_bytes)
        except CVExtractionError as exc:
            logger.warning(
                "HiringPipeline: CV extraction failed  reason=%s", exc.reason
            )
            raise CVExtractionFailed(reason=exc.reason, detail=exc.detail) from exc
        except Exception as exc:
            logger.error("HiringPipeline: unexpected CV error", exc_info=True)
            raise CVExtractionFailed(
                reason="unexpected_error", detail=str(exc)
            ) from exc

    def _retrieve_context(self, job_query: str) -> str:
        """
        Delegate to RAGService and extract the concatenated context.

        Raises:
            ContextRetrievalFailed: On any RAGService failure.
        """
        try:
            search_result: SearchResult = self._rag_service.search(
                query=job_query, top_k=5
            )
        except SearchError as exc:
            logger.error(
                "HiringPipeline: RAG search failed  query='%s'", job_query, exc_info=True
            )
            raise ContextRetrievalFailed(detail=str(exc)) from exc
        except Exception as exc:
            logger.error(
                "HiringPipeline: unexpected RAG error  query='%s'", job_query, exc_info=True
            )
            raise ContextRetrievalFailed(detail=f"Unexpected error: {exc}") from exc

        context = search_result.context

        if not context or not context.strip():
            logger.warning(
                "HiringPipeline: RAG context is empty for query='%s'", job_query
            )

        logger.info(
            "HiringPipeline: retrieved %d chunks  context_len=%d",
            len(search_result.chunks),
            len(context),
        )
        return context

    def _evaluate(self, cv_text: str, rag_context: str) -> EvaluationResult:
        """
        Delegate to EvaluatorService and translate its exceptions.

        Raises:
            EvaluationFailed: On Gemini call or output parsing failure.
        """
        try:
            return self._evaluator_service.evaluate(cv_text, rag_context)
        except (GeminiCallError, OutputParsingError) as exc:
            logger.error(
                "HiringPipeline: evaluation failed  type=%s", type(exc).__name__, exc_info=True
            )
            raise EvaluationFailed(
                reason=type(exc).__name__, detail=str(exc)
            ) from exc
        except Exception as exc:
            logger.error("HiringPipeline: unexpected evaluation error", exc_info=True)
            raise EvaluationFailed(
                reason="unexpected_error", detail=str(exc)
            ) from exc

    def _persist(
        self,
        cv_result: CVExtractionResult,
        rag_context: str,
        evaluation_result: EvaluationResult,
        job_query: str,
    ) -> int | None:
        """
        Persist the evaluation result to SQLite.

        Failures here are logged but do NOT raise — persistence is
        best-effort and must not break the response to the caller.

        Returns:
            The new record's primary key, or None if persistence is
            disabled or fails.
        """
        if self._db_manager is None:
            logger.debug("HiringPipeline: db_manager not set — skipping persistence.")
            return None

        record = EvaluationRecord(
            job_title=job_query,
            cv_text=cv_result.text,
            cv_char_count=cv_result.char_count,
            rag_context=rag_context,
            rag_context_length=len(rag_context),
            final_score=evaluation_result.score,
            hiring_decision=evaluation_result.decision.value,
            strengths=evaluation_result.strengths,
            weaknesses=evaluation_result.weaknesses,
            summary=evaluation_result.summary,
        )

        try:
            evaluation_id = self._db_manager.save_evaluation(record)
            logger.info("HiringPipeline: persisted  db_id=%d", evaluation_id)
            return evaluation_id
        except PersistenceError as exc:
            # Non-fatal — log and continue
            logger.error(
                "HiringPipeline: persistence failed (non-fatal) — %s", exc, exc_info=True
            )
            return None

    @staticmethod
    def _validate_inputs(pdf_bytes: bytes, job_query: str) -> None:
        """
        Guard clauses for input parameters.

        Raises:
            TypeError:  If types are incorrect.
            ValueError: If strings are empty.
        """
        if not isinstance(pdf_bytes, bytes):
            raise TypeError(
                f"pdf_bytes must be bytes, got {type(pdf_bytes).__name__!r}."
            )
        if not isinstance(job_query, str):
            raise TypeError(
                f"job_query must be str, got {type(job_query).__name__!r}."
            )
        if not pdf_bytes:
            raise ValueError("pdf_bytes must not be empty.")
        if not job_query.strip():
            raise ValueError("job_query must not be empty or whitespace.")


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def create_hiring_pipeline(
    cv_service: CVService | None = None,
    rag_service: RAGService | None = None,
    evaluator_service: EvaluatorService | None = None,
    db_manager: DatabaseManager | None = None,
) -> HiringPipeline:
    """
    Convenience factory to build a HiringPipeline with default service instances.

    Args:
        cv_service:        Optional CVService instance.
        rag_service:       Optional RAGService instance.
        evaluator_service: Optional EvaluatorService instance.
        db_manager:        Optional DatabaseManager instance. Defaults to shared singleton.

    Returns:
        Configured HiringPipeline instance.
    """
    from services.cv_service import CVService as _CVService
    from services.evaluator import EvaluatorService as _EvaluatorService
    from services.rag import RAGService as _RAGService
    from services.database import get_db_manager

    return HiringPipeline(
        cv_service=cv_service or _CVService(),
        rag_service=rag_service or _RAGService(),
        evaluator_service=evaluator_service or _EvaluatorService(),
        db_manager=db_manager or get_db_manager(),
    )