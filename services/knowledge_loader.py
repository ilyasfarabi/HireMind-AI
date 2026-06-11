"""
services/knowledge_loader.py

Responsibility:
    - Load all .txt knowledge files from the knowledge/ directory.
    - Ingest each document into the RAGService vector store.
    - Expose a callable load_knowledge() function for use in main.py.

Out of scope:
    - FastAPI routes.
    - PDF parsing.
    - Gemini / evaluation logic.
"""

from __future__ import annotations

import logging
from pathlib import Path

from services.rag import KnowledgeDocument, RAGService, get_rag_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWLEDGE_DIR: Path = Path("knowledge")
KNOWLEDGE_GLOB: str = "*.txt"
FILE_ENCODING: str = "utf-8"


# ---------------------------------------------------------------------------
# Public function  (FIX: دالة صريحة بدل كود يشتغل عند import)
# ---------------------------------------------------------------------------


def load_knowledge(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    rag_service: RAGService | None = None,
) -> int:
    """
    Scan *knowledge_dir* for ``*.txt`` files and ingest each one into RAG.

    Args:
        knowledge_dir: Directory containing knowledge ``.txt`` files.
                       Defaults to ``./knowledge``.
        rag_service:   RAGService instance to use. Defaults to the
                       module-level singleton from :func:`get_rag_service`.

    Returns:
        Number of documents successfully ingested.

    Raises:
        FileNotFoundError: If *knowledge_dir* does not exist.
    """
    if not knowledge_dir.exists():
        raise FileNotFoundError(
            f"Knowledge directory not found: '{knowledge_dir.resolve()}'. "
            "Create it and add .txt job description files before starting."
        )

    service: RAGService = rag_service or get_rag_service()
    files = sorted(knowledge_dir.glob(KNOWLEDGE_GLOB))

    if not files:
        logger.warning(
            "load_knowledge: no .txt files found in '%s'. "
            "RAG context will be empty.",
            knowledge_dir,
        )
        return 0

    logger.info(
        "load_knowledge: found %d file(s) in '%s'", len(files), knowledge_dir
    )

    ingested = 0
    for file_path in files:
        try:
            content = file_path.read_text(encoding=FILE_ENCODING)
            service.add_document(
                KnowledgeDocument(
                    content=content,
                    source=file_path.name,
                )
            )
            logger.info("load_knowledge: ingested '%s'", file_path.name)
            ingested += 1
        except Exception as exc:
            # Log and continue — one bad file should not abort the entire load
            logger.error(
                "load_knowledge: failed to ingest '%s' — %s",
                file_path.name,
                exc,
                exc_info=True,
            )

    logger.info("load_knowledge: done  ingested=%d / %d", ingested, len(files))
    return ingested
