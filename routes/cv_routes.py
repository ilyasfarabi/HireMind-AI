"""
routes/cv_routes.py

Responsibility:
    - Accept and validate PDF file uploads.
    - Inject CVService via FastAPI Depends.
    - Delegate all extraction logic to CVService.
    - Map service-level exceptions to meaningful HTTP responses.
    - Return structured extraction results to the caller.

Out of scope:
    - PDF parsing or text extraction.
    - Business logic of any kind.
    - Gemini / AI / RAG / database calls.
"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from services.cv_service import CVExtractionError, CVService, get_cv_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALLOWED_CONTENT_TYPES: frozenset[str] = frozenset(
    {"application/pdf", "application/x-pdf"}
)
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
PDF_MAGIC_BYTES: bytes = b"%PDF"

# Map CVExtractionError.reason → HTTP status code
_REASON_TO_STATUS: dict[str, int] = {
    "invalid_pdf":   status.HTTP_400_BAD_REQUEST,
    "encrypted_pdf": status.HTTP_422_UNPROCESSABLE_ENTITY,
    "empty_pdf":     status.HTTP_400_BAD_REQUEST,
    "parser_error":  status.HTTP_500_INTERNAL_SERVER_ERROR,
}

# ---------------------------------------------------------------------------
# Response schema
# ---------------------------------------------------------------------------


class CVUploadResponse(BaseModel):
    """Successful response returned after CV text extraction."""

    filename: str = Field(..., description="Original name of the uploaded file.")
    size_bytes: int = Field(..., description="Size of the uploaded file in bytes.")
    char_count: int = Field(..., description="Number of characters extracted.")
    extracted_at: str = Field(..., description="UTC ISO-8601 timestamp of extraction.")
    text: str = Field(..., description="Full plain text extracted from the CV.")

    model_config = {"json_schema_extra": {
        "example": {
            "filename": "john_doe_cv.pdf",
            "size_bytes": 204800,
            "char_count": 3120,
            "extracted_at": "2024-11-01T10:30:00+00:00",
            "text": "John Doe\nSenior Python Engineer\n...",
        }
    }}


# ---------------------------------------------------------------------------
# Error response schema (for OpenAPI docs only)
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    """Standard error envelope."""

    detail: str = Field(..., description="Human-readable error message.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/cv",
    tags=["CV"],
)

# ---------------------------------------------------------------------------
# Dependency: validated file bytes
# ---------------------------------------------------------------------------


async def get_validated_pdf_bytes(
    file: Annotated[
        UploadFile,
        File(description="CV document — must be a PDF, max 10 MB."),
    ],
) -> tuple[str, bytes]:
    """
    FastAPI dependency that reads and validates an uploaded PDF.

    Centralising validation here keeps the endpoint handler free of
    any file-handling logic and makes validation independently testable.

    Args:
        file: Multipart upload provided by the client.

    Returns:
        A ``(filename, content)`` tuple ready for the service layer.

    Raises:
        HTTPException 400: On extension, MIME type, magic-bytes, or empty-file failures.
        HTTPException 413: If the file exceeds the size limit.
        HTTPException 500: If the upload stream cannot be read.
    """
    try:
        content: bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to read the uploaded file.",
        ) from exc
    finally:
        await file.close()

    filename: str = file.filename or ""

    _assert_valid_filename(filename)
    _assert_valid_mime(file.content_type, filename)
    _assert_valid_size(content, filename)
    _assert_valid_magic(content, filename)

    return filename, content


# ---------------------------------------------------------------------------
# Validation helpers (pure functions — no HTTP context)
# ---------------------------------------------------------------------------


def _assert_valid_filename(filename: str) -> None:
    if not filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename was provided with the upload.",
        )
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' must have a .pdf extension.",
        )


def _assert_valid_mime(content_type: str | None, filename: str) -> None:
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File '{filename}' has an invalid content type "
                f"'{content_type}'. Only PDF files are accepted."
            ),
        )


def _assert_valid_size(content: bytes, filename: str) -> None:
    if len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File '{filename}' is empty.",
        )
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File '{filename}' is {len(content):,} bytes, which exceeds "
                f"the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit."
            ),
        )


def _assert_valid_magic(content: bytes, filename: str) -> None:
    if not content.startswith(PDF_MAGIC_BYTES):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File '{filename}' does not appear to be a valid PDF "
                "(magic bytes check failed)."
            ),
        )


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/upload",
    response_model=CVUploadResponse,
    status_code=status.HTTP_200_OK,
    summary="Upload and extract text from a CV",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid or unreadable PDF."},
        413: {"model": ErrorResponse, "description": "File exceeds size limit."},
        422: {"model": ErrorResponse, "description": "PDF is password-protected."},
        500: {"model": ErrorResponse, "description": "Internal extraction failure."},
    },
)
async def upload_cv(
    validated: Annotated[
        tuple[str, bytes],
        Depends(get_validated_pdf_bytes),
    ],
    service: Annotated[
        CVService,
        Depends(get_cv_service),
    ],
) -> CVUploadResponse:
    """
    Upload a CV as a PDF file and receive its extracted plain text.

    **Flow**

    1. ``get_validated_pdf_bytes`` dependency reads and validates the upload.
    2. ``CVService.process_cv`` extracts the text via ``PDFParser``.
    3. The extracted result is returned as a structured JSON response.

    **Validation rules**

    - Filename must end with `.pdf`.
    - Content-Type must be `application/pdf`.
    - File must not be empty and must not exceed 10 MB.
    - File must begin with the PDF magic bytes (`%PDF`).
    """
    filename, content = validated

    logger.info("upload_cv: processing file='%s'  size=%d", filename, len(content))

    try:
        result = service.process_cv(content)
    except CVExtractionError as exc:
        http_status = _REASON_TO_STATUS.get(
            exc.reason, status.HTTP_500_INTERNAL_SERVER_ERROR
        )
        logger.warning(
            "upload_cv: extraction failed  reason=%s  file='%s'",
            exc.reason,
            filename,
        )
        raise HTTPException(status_code=http_status, detail=exc.detail) from exc

    logger.info(
        "upload_cv: success  file='%s'  chars=%d", filename, result.char_count
    )

    return CVUploadResponse(
        filename=filename,
        size_bytes=len(content),
        char_count=result.char_count,
        extracted_at=result.extracted_at.isoformat(),
        text=result.text,
    )
