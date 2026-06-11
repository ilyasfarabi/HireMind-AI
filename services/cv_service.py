"""
services/cv_service.py

Responsibility:
    - Orchestrate the flow between the API route and PDFParser.
    - Receive raw PDF bytes from the route layer.
    - Delegate text extraction to an injected PDFParser instance.
    - Translate parser-level exceptions into service-level exceptions.
    - Return clean extracted text to the caller.

Out of scope:
    - Gemini / AI calls.
    - RAG pipeline.
    - Database operations.
    - Email or notification integrations.
    - Business rules of any kind.

Dependencies:
    services/pdf_parser.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from services.pdf_parser import (
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFParser,
    PDFParserError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Service-level exceptions
# ---------------------------------------------------------------------------


class CVServiceError(Exception):
    """Base exception for all cv_service failures."""


class CVExtractionError(CVServiceError):
    """
    Raised when text extraction fails for any reason.

    Attributes:
        reason: Machine-readable failure category.
        detail: Human-readable explanation forwarded from the parser.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reason={self.reason!r}, detail={self.detail!r})"


# ---------------------------------------------------------------------------
# Result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CVExtractionResult:
    """
    Immutable value object returned by :meth:`CVService.process_cv`.

    Attributes:
        text:           Extracted plain text from the CV.
        char_count:     Number of characters in the extracted text.
        extracted_at:   UTC timestamp of when extraction completed.
    """

    text: str
    char_count: int = field(init=False)
    extracted_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class CVService:
    """
    Orchestration layer between the CV upload route and :class:`PDFParser`.

    The service is intentionally thin: it delegates all extraction work to
    the injected ``PDFParser`` and owns only the translation of parser
    exceptions into typed service exceptions.

    The class is stateless and thread-safe; a single instance can be shared
    across all requests (e.g. via FastAPI's dependency injection).

    Args:
        parser: A :class:`PDFParser` instance. Defaults to a new instance
                if not provided, supporting both DI and direct construction.

    Example::

        service = CVService()
        result  = service.process_cv(pdf_bytes)
        print(result.text)
    """

    def __init__(self, parser: PDFParser | None = None) -> None:
        self._parser: PDFParser = parser or PDFParser()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_cv(self, pdf_bytes: bytes) -> CVExtractionResult:
        """
        Extract text from a PDF CV supplied as raw bytes.

        This is the single public entry point for the service.  The route
        layer calls this method and receives a :class:`CVExtractionResult`
        on success, or a :class:`CVExtractionError` on failure.

        Args:
            pdf_bytes: Raw binary content of a PDF file.

        Returns:
            :class:`CVExtractionResult` containing the extracted text and
            metadata.

        Raises:
            TypeError:         If ``pdf_bytes`` is not ``bytes``.
            CVExtractionError: For every parser-level failure, with a
                               ``reason`` field indicating the category:

                               - ``"invalid_pdf"``    — corrupt / non-PDF input
                               - ``"encrypted_pdf"``  — password-protected file
                               - ``"empty_pdf"``      — no extractable text
                               - ``"parser_error"``   — unexpected parser failure
        """
        self._validate_input(pdf_bytes)

        logger.info("CVService: starting extraction  size=%d bytes", len(pdf_bytes))

        text = self._run_extraction(pdf_bytes)

        result = CVExtractionResult(text=text)

        logger.info(
            "CVService: extraction complete  chars=%d  extracted_at=%s",
            result.char_count,
            result.extracted_at.isoformat(),
        )

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(pdf_bytes: bytes) -> None:
        """
        Guard clause — ensures the input type is correct before proceeding.

        Args:
            pdf_bytes: Value to validate.

        Raises:
            TypeError: If the value is not ``bytes``.
        """
        if not isinstance(pdf_bytes, bytes):
            raise TypeError(
                f"pdf_bytes must be bytes, got {type(pdf_bytes).__name__!r}."
            )

    def _run_extraction(self, pdf_bytes: bytes) -> str:
        """
        Delegate to the parser and translate exceptions into service errors.

        Args:
            pdf_bytes: Validated raw PDF bytes.

        Returns:
            Extracted text string.

        Raises:
            CVExtractionError: On any :class:`PDFParserError` subclass.
        """
        try:
            return self._parser.extract(pdf_bytes)

        except InvalidPDFError as exc:
            logger.warning("CVService: invalid PDF — %s", exc)
            raise CVExtractionError(
                reason="invalid_pdf",
                detail="The uploaded file is not a valid PDF or is corrupt.",
            ) from exc

        except EncryptedPDFError as exc:
            logger.warning("CVService: encrypted PDF — %s", exc)
            raise CVExtractionError(
                reason="encrypted_pdf",
                detail="The PDF is password-protected. Please upload an unlocked version.",
            ) from exc

        except EmptyPDFError as exc:
            logger.warning("CVService: empty PDF — %s", exc)
            raise CVExtractionError(
                reason="empty_pdf",
                detail=(
                    "No text could be extracted from the PDF. "
                    "It may contain only scanned images."
                ),
            ) from exc

        except PDFParserError as exc:
            logger.error("CVService: unexpected parser error — %s", exc, exc_info=True)
            raise CVExtractionError(
                reason="parser_error",
                detail="An unexpected error occurred while processing the PDF.",
            ) from exc


# ---------------------------------------------------------------------------
# FastAPI dependency factory
# ---------------------------------------------------------------------------


def get_cv_service() -> CVService:
    """
    FastAPI dependency that provides a shared :class:`CVService` instance.

    Declare it once at module level to get singleton-like behaviour across
    all requests without needing a DI container.

    Example::

        from fastapi import Depends
        from services.cv_service import CVService, get_cv_service

        @router.post("/upload")
        async def upload_cv(
            file: UploadFile,
            service: CVService = Depends(get_cv_service),
        ):
            result = service.process_cv(await file.read())
            ...
    """
    return _shared_cv_service


# Module-level singleton — instantiated once at import time.
_shared_cv_service: CVService = CVService()