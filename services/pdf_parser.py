"""
services/pdf_parser.py

Responsibility:
    - Extract plain text from a PDF supplied as raw bytes.
    - Surface meaningful, typed exceptions for every failure mode.

Out of scope:
    - Gemini / AI calls.
    - RAG pipeline.
    - Database interaction.
    - Business logic of any kind.

Dependencies:
    pip install PyMuPDF
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import fitz  # PyMuPDF

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class PDFParserError(Exception):
    """Base exception for all pdf_parser failures."""


class InvalidPDFError(PDFParserError):
    """Raised when the bytes cannot be opened as a valid PDF."""


class EmptyPDFError(PDFParserError):
    """Raised when the PDF is valid but contains no extractable text."""


class EncryptedPDFError(PDFParserError):
    """Raised when the PDF is password-protected and cannot be read."""


# ---------------------------------------------------------------------------
# Internal data container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _ParseResult:
    """
    Intermediate result produced by the low-level page loop.

    Attributes:
        text:       Concatenated text of all pages.
        page_count: Total number of pages in the document.
        char_count: Total number of characters extracted.
    """

    text: str
    page_count: int
    char_count: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "char_count", len(self.text))


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------


class PDFParser:
    """
    Stateless PDF text extractor backed by PyMuPDF (fitz).

    Usage
    -----
    ::

        parser = PDFParser()
        text = parser.extract(pdf_bytes)

    The class is stateless — a single instance can safely be reused
    across many requests (e.g. as a FastAPI dependency singleton).
    """

    # Page-separator inserted between consecutive pages in the output.
    _PAGE_SEPARATOR: str = "\n\n"

    def extract(self, pdf_bytes: bytes) -> str:
        """
        Extract all text from a PDF document supplied as bytes.

        Args:
            pdf_bytes: Raw binary content of a PDF file.

        Returns:
            A single string containing the text of every page,
            pages separated by ``\\n\\n``.

        Raises:
            TypeError:        If ``pdf_bytes`` is not ``bytes``.
            InvalidPDFError:  If the bytes are not a valid PDF or are corrupt.
            EncryptedPDFError: If the document requires a password.
            EmptyPDFError:    If the document yields no extractable text.
        """
        if not isinstance(pdf_bytes, bytes):
            raise TypeError(
                f"pdf_bytes must be bytes, got {type(pdf_bytes).__name__!r}."
            )

        doc = self._open_document(pdf_bytes)

        try:
            result = self._extract_pages(doc)
        finally:
            doc.close()

        logger.info(
            "PDF parsed: pages=%d  chars=%d",
            result.page_count,
            result.char_count,
        )

        return result.text

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _open_document(pdf_bytes: bytes) -> fitz.Document:
        """
        Open raw bytes as a :class:`fitz.Document`.

        Args:
            pdf_bytes: Raw PDF bytes.

        Returns:
            An open fitz.Document instance.

        Raises:
            InvalidPDFError:   On corrupt or non-PDF input.
            EncryptedPDFError: If the document is password-protected.
        """
        try:
            doc: fitz.Document = fitz.open(stream=pdf_bytes, filetype="pdf")
        except fitz.FileDataError as exc:
            raise InvalidPDFError(
                "Could not open the provided bytes as a PDF. "
                "The file may be corrupt or not a PDF at all."
            ) from exc
        except Exception as exc:
            raise InvalidPDFError(
                f"Unexpected error while opening PDF: {exc}"
            ) from exc

        if doc.is_encrypted:
            doc.close()
            raise EncryptedPDFError(
                "The PDF is password-protected. "
                "Provide a decrypted version for text extraction."
            )

        return doc

    def _extract_pages(self, doc: fitz.Document) -> _ParseResult:
        """
        Iterate over every page and accumulate extracted text.

        Args:
            doc: An open, unencrypted :class:`fitz.Document`.

        Returns:
            A :class:`_ParseResult` with the concatenated text and metadata.

        Raises:
            EmptyPDFError: If no text was found across all pages.
        """
        pages: list[str] = []

        for page_index in range(len(doc)):
            page: fitz.Page = doc.load_page(page_index)
            page_text: str = page.get_text("text")  # type: ignore[arg-type]

            stripped = page_text.strip()
            if stripped:
                pages.append(stripped)
            else:
                logger.debug("Page %d yielded no text — skipping.", page_index + 1)

        if not pages:
            raise EmptyPDFError(
                "No extractable text was found in the PDF. "
                "It may consist entirely of scanned images."
            )

        full_text = self._PAGE_SEPARATOR.join(pages)
        return _ParseResult(text=full_text, page_count=len(doc))


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """
    Module-level shortcut for one-off extractions.

    Internally creates a :class:`PDFParser` instance and delegates to
    :meth:`PDFParser.extract`. Prefer injecting :class:`PDFParser`
    directly in long-running services to avoid repeated instantiation.

    Args:
        pdf_bytes: Raw binary content of a PDF file.

    Returns:
        Extracted text as a single string.

    Raises:
        TypeError:         If ``pdf_bytes`` is not ``bytes``.
        InvalidPDFError:   If the bytes are not a valid PDF.
        EncryptedPDFError: If the PDF is password-protected.
        EmptyPDFError:     If the PDF contains no extractable text.

    Example::

        with open("cv.pdf", "rb") as f:
            text = extract_text_from_pdf(f.read())
    """
    return PDFParser().extract(pdf_bytes)
