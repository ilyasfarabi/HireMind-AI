"""
routes/history_routes.py

Responsibility:
    - Expose GET endpoints for querying stored evaluations from SQLite.
    - Inject DatabaseManager via FastAPI Depends.
    - Map database exceptions to HTTP responses.
    - Return structured JSON responses.

Out of scope:
    - PDF parsing, RAG, or Gemini calls.
    - Writing evaluations (handled by evaluation_routes via HiringPipeline).
    - Business logic of any kind.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from services.database import (
    DatabaseError,
    DatabaseManager,
    EvaluationRecord,
    EvaluationSummary,
    RecordNotFoundError,
    get_db_manager,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------


class EvaluationSummaryResponse(BaseModel):
    """Lightweight evaluation summary for list views."""

    id: int
    job_title: str
    cv_char_count: int
    final_score: int
    hiring_decision: str
    strengths: list[str]
    weaknesses: list[str]
    summary: str
    created_at: str

    model_config = {"json_schema_extra": {
        "example": {
            "id": 1,
            "job_title": "AI Engineer",
            "cv_char_count": 3120,
            "final_score": 82,
            "hiring_decision": "Yes",
            "strengths": ["Python", "RAG Systems"],
            "weaknesses": ["Kubernetes"],
            "summary": "Strong candidate with relevant AI experience.",
            "created_at": "2024-11-01T10:30:00+00:00",
        }
    }}

    @classmethod
    def from_summary(cls, s: EvaluationSummary) -> "EvaluationSummaryResponse":
        return cls(
            id=s.id,
            job_title=s.job_title,
            cv_char_count=s.cv_char_count,
            final_score=s.final_score,
            hiring_decision=s.hiring_decision,
            strengths=s.strengths,
            weaknesses=s.weaknesses,
            summary=s.summary,
            created_at=s.created_at.isoformat(),
        )


class EvaluationDetailResponse(EvaluationSummaryResponse):
    """Full evaluation record including CV text and RAG context."""

    cv_text: str
    rag_context: str
    rag_context_length: int

    @classmethod
    def from_record(cls, r: EvaluationRecord) -> "EvaluationDetailResponse":
        return cls(
            id=r.id,  # type: ignore[arg-type]
            job_title=r.job_title,
            cv_char_count=r.cv_char_count,
            final_score=r.final_score,
            hiring_decision=r.hiring_decision,
            strengths=r.strengths,
            weaknesses=r.weaknesses,
            summary=r.summary,
            created_at=r.created_at.isoformat(),
            cv_text=r.cv_text,
            rag_context=r.rag_context,
            rag_context_length=r.rag_context_length,
        )


class PaginatedEvaluationsResponse(BaseModel):
    """Paginated list of evaluation summaries."""

    total: int = Field(..., description="Total matching records.")
    limit: int = Field(..., description="Page size.")
    offset: int = Field(..., description="Current offset.")
    items: list[EvaluationSummaryResponse]


class StatsResponse(BaseModel):
    """Aggregate statistics across all evaluations."""

    total_evaluations: int
    avg_score: float
    decisions: dict[str, int] = Field(
        ..., description="Count of Yes / No / Consider decisions."
    )
    top_jobs: list[dict] = Field(
        ..., description="Top job titles by evaluation count."
    )


class ErrorResponse(BaseModel):
    detail: str


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/history",
    tags=["History"],
)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "",
    response_model=PaginatedEvaluationsResponse,
    summary="List stored evaluations",
    responses={500: {"model": ErrorResponse}},
)
def list_evaluations(
    job_title: Annotated[str | None, Query(description="Filter by job title.")] = None,
    hiring_decision: Annotated[
        str | None,
        Query(description="Filter by decision: Yes, No, or Consider.", pattern="^(Yes|No|Consider)$"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100, description="Page size.")] = 20,
    offset: Annotated[int, Query(ge=0, description="Rows to skip.")] = 0,
    db: Annotated[DatabaseManager, Depends(get_db_manager)] = ...,
) -> PaginatedEvaluationsResponse:
    """
    List stored candidate evaluations with optional filters and pagination.

    - Filter by **job_title** (case-insensitive exact match).
    - Filter by **hiring_decision** (Yes / No / Consider).
    - Results are ordered newest-first.
    """
    try:
        summaries = db.list_evaluations(
            job_title=job_title,
            hiring_decision=hiring_decision,
            limit=limit,
            offset=offset,
        )
        total = db.count_evaluations(
            job_title=job_title,
            hiring_decision=hiring_decision,
        )
    except DatabaseError as exc:
        logger.error("GET /history failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluations.",
        ) from exc

    return PaginatedEvaluationsResponse(
        total=total,
        limit=limit,
        offset=offset,
        items=[EvaluationSummaryResponse.from_summary(s) for s in summaries],
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="Aggregate evaluation statistics",
    responses={500: {"model": ErrorResponse}},
)
def get_stats(
    db: Annotated[DatabaseManager, Depends(get_db_manager)] = ...,
) -> StatsResponse:
    """
    Return aggregate statistics: total evaluations, average score,
    decision breakdown, and top job titles.
    """
    try:
        stats = db.get_stats()
    except DatabaseError as exc:
        logger.error("GET /history/stats failed: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to compute statistics.",
        ) from exc

    return StatsResponse(**stats)


@router.get(
    "/{evaluation_id}",
    response_model=EvaluationDetailResponse,
    summary="Get a single evaluation by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Evaluation not found."},
        500: {"model": ErrorResponse},
    },
)
def get_evaluation(
    evaluation_id: int,
    db: Annotated[DatabaseManager, Depends(get_db_manager)] = ...,
) -> EvaluationDetailResponse:
    """
    Retrieve the full details of a single evaluation, including the
    original CV text and RAG context.
    """
    try:
        record = db.get_evaluation(evaluation_id)
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatabaseError as exc:
        logger.error("GET /history/%d failed: %s", evaluation_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve evaluation.",
        ) from exc

    return EvaluationDetailResponse.from_record(record)


@router.delete(
    "/{evaluation_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an evaluation by ID",
    responses={
        404: {"model": ErrorResponse, "description": "Evaluation not found."},
        500: {"model": ErrorResponse},
    },
)
def delete_evaluation(
    evaluation_id: int,
    db: Annotated[DatabaseManager, Depends(get_db_manager)] = ...,
) -> None:
    """
    Permanently delete a stored evaluation and its related data.
    """
    try:
        db.delete_evaluation(evaluation_id)
        logger.info("DELETE /history/%d: success", evaluation_id)
    except RecordNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except DatabaseError as exc:
        logger.error("DELETE /history/%d failed: %s", evaluation_id, exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete evaluation.",
        ) from exc
