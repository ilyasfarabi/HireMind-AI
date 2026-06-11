"""
routes/evaluation_routes.py

Responsibility:
    - Expose a POST /evaluate endpoint.
    - Accept a PDF CV upload and a job_title form field.
    - Inject HiringPipeline using FastAPI Depends.
    - Execute the complete evaluation workflow.
    - Return a structured JSON response including the evaluation_id from SQLite.

Out of scope:
    - PDF parsing (delegated to HiringPipeline → CVService).
    - RAG implementation (delegated to HiringPipeline → RAGService).
    - Gemini implementation (delegated to HiringPipeline → EvaluatorService).
    - SQLite operations (delegated to HiringPipeline → DatabaseManager).
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from services.cv_service import get_cv_service
from services.rag import get_rag_service
from services.evaluator import get_evaluator_service
from services.database import get_db_manager
from services.hiring_pipeline import (
    CVExtractionFailed,
    ContextRetrievalFailed,
    EvaluationFailed,
    HiringPipeline,
    HiringPipelineResult,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/pdf", "application/x-pdf"}
)
PDF_MAGIC_BYTES: bytes = b"%PDF"

_CV_REASON_TO_STATUS: dict[str, int] = {
    "invalid_pdf":      status.HTTP_400_BAD_REQUEST,
    "encrypted_pdf":    status.HTTP_422_UNPROCESSABLE_ENTITY,
    "empty_pdf":        status.HTTP_400_BAD_REQUEST,
    "parser_error":     status.HTTP_500_INTERNAL_SERVER_ERROR,
    "unexpected_error": status.HTTP_500_INTERNAL_SERVER_ERROR,
}

# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class EvaluationResponse(BaseModel):
    """Successful response returned after candidate evaluation."""

    evaluation_id: int | None = Field(
        None, description="SQLite record ID for this evaluation (use for history lookup)."
    )
    final_score: int = Field(..., description="Overall fit score (0-100).", ge=0, le=100)
    hiring_decision: str = Field(
        ...,
        description="Hiring recommendation: 'Yes', 'No', or 'Consider'.",
        pattern="^(Yes|No|Consider)$",
    )
    strengths: list[str] = Field(..., description="Candidate's strengths relevant to the job.")
    weaknesses: list[str] = Field(..., description="Gaps or missing skills.")
    summary: str = Field(..., description="Justification of the evaluation.")
    job_query_used: str = Field(..., description="Job title used for retrieval.")
    cv_char_count: int = Field(..., description="Characters extracted from the CV.")
    rag_context_length: int = Field(..., description="Length of the RAG context string.")

    model_config = {"json_schema_extra": {
        "example": {
            "evaluation_id": 42,
            "final_score": 82,
            "hiring_decision": "Yes",
            "strengths": ["Strong Python skills", "RAG experience"],
            "weaknesses": ["No Kubernetes experience"],
            "summary": "Candidate is a strong fit for the AI Engineer role.",
            "job_query_used": "AI Engineer",
            "cv_char_count": 3120,
            "rag_context_length": 940,
        }
    }}

    @classmethod
    def from_pipeline_result(
        cls, result: HiringPipelineResult, job_title: str
    ) -> "EvaluationResponse":
        return cls(
            evaluation_id=result.evaluation_id,
            final_score=result.final_score,
            hiring_decision=result.hiring_decision,
            strengths=result.evaluation.strengths,
            weaknesses=result.evaluation.weaknesses,
            summary=result.evaluation.summary,
            job_query_used=job_title,
            cv_char_count=result.cv_extraction.char_count,
            rag_context_length=len(result.rag_context),
        )


class ErrorResponse(BaseModel):
    detail: str
    error_type: str | None = None


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/evaluate",
    tags=["Evaluation"],
)

# ---------------------------------------------------------------------------
# Singleton pipeline
# ---------------------------------------------------------------------------

_shared_hiring_pipeline: HiringPipeline = HiringPipeline(
    cv_service=get_cv_service(),
    rag_service=get_rag_service(),
    evaluator_service=get_evaluator_service(),
    db_manager=get_db_manager(),        # ← SQLite persistence wired in
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


async def get_validated_pdf_and_job(
    file: Annotated[UploadFile, File(description="Candidate CV (PDF format, max 10 MB).")],
    job_title: Annotated[str, Form(description="Job title or role to evaluate against.")],
) -> tuple[bytes, str]:
    job_title_stripped = job_title.strip()
    if not job_title_stripped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Job title must not be empty.",
        )

    try:
        content: bytes = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not read the uploaded file.",
        ) from exc
    finally:
        await file.close()

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid content type '{file.content_type}'. Only application/pdf is accepted.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {len(content):,} bytes exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    if not content.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File does not appear to be a valid PDF (magic bytes mismatch).",
        )

    return content, job_title_stripped


def get_hiring_pipeline() -> HiringPipeline:
    return _shared_hiring_pipeline


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "",
    response_model=EvaluationResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate a candidate CV against a job title",
    description=(
        "Upload a candidate's CV (PDF) and a job title. "
        "The system extracts CV text, retrieves job requirements via RAG, "
        "uses Gemini to produce a structured hiring evaluation, "
        "and persists the result to SQLite."
    ),
    responses={
        400: {"model": ErrorResponse, "description": "Invalid PDF or empty job title."},
        413: {"model": ErrorResponse, "description": "File too large (> 10 MB)."},
        422: {"model": ErrorResponse, "description": "Encrypted / unreadable PDF."},
        500: {"model": ErrorResponse, "description": "Internal server error."},
    },
)
async def evaluate_candidate(
    validated: Annotated[tuple[bytes, str], Depends(get_validated_pdf_and_job)],
    pipeline: Annotated[HiringPipeline, Depends(get_hiring_pipeline)],
) -> EvaluationResponse:
    """
    Evaluate a candidate CV against a job title.

    Steps:
    1. Extract plain text from the uploaded PDF.
    2. Retrieve job requirements via RAG.
    3. Score the candidate with Gemini.
    4. Save result to SQLite and return evaluation_id.
    """
    pdf_bytes, job_title = validated

    logger.info("POST /evaluate  job_title='%s'  pdf_size=%d bytes", job_title, len(pdf_bytes))

    try:
        result: HiringPipelineResult = pipeline.run(pdf_bytes, job_title)

    except CVExtractionFailed as exc:
        http_status = _CV_REASON_TO_STATUS.get(exc.reason, status.HTTP_500_INTERNAL_SERVER_ERROR)
        logger.warning("CV extraction failed  job='%s'  reason=%s", job_title, exc.reason)
        raise HTTPException(status_code=http_status, detail=exc.detail) from exc

    except ContextRetrievalFailed as exc:
        logger.error("RAG retrieval failed  job='%s'", job_title, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve job requirements: {exc.detail}",
        ) from exc

    except EvaluationFailed as exc:
        logger.error("Gemini evaluation failed  job='%s'", job_title, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation error: {exc.detail}",
        ) from exc

    except Exception as exc:
        logger.exception("Unexpected error in /evaluate  job='%s'", job_title)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal error occurred. Please try again later.",
        ) from exc

    logger.info(
        "Evaluation complete  job='%s'  score=%d  decision=%s  db_id=%s",
        job_title,
        result.final_score,
        result.hiring_decision,
        result.evaluation_id or "not saved",
    )

    return EvaluationResponse.from_pipeline_result(result, job_title)