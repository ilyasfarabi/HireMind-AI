"""
services/rag.py

Responsibility:
    - Manage company knowledge documents (add, list, delete).
    - Split documents into overlapping chunks.
    - Generate embeddings via LangChain embedding models.
    - Persist and retrieve chunks using ChromaDB.
    - Expose similarity search for downstream Gemini/LLM calls.

Out of scope:
    - Gemini / LLM calls.
    - Candidate evaluation or scoring.
    - FastAPI routes.
    - Database persistence outside ChromaDB.

Dependencies:
    pip install langchain langchain-community langchain-huggingface chromadb
    (or langchain-google-genai for Gemini embeddings)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE: int = 500
DEFAULT_CHUNK_OVERLAP: int = 100
DEFAULT_COLLECTION_NAME: str = "hiremind_knowledge"
DEFAULT_TOP_K: int = 5

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class RAGServiceError(Exception):
    """Base exception for all rag service failures."""


class DocumentIngestionError(RAGServiceError):
    """Raised when a document cannot be ingested into the vector store."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason = reason
        self.detail = detail

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reason={self.reason!r}, detail={self.detail!r})"


class SearchError(RAGServiceError):
    """Raised when a similarity search fails."""

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    """
    Input document to be ingested into the knowledge base.

    Attributes:
        content:    Raw text content of the document.
        source:     Identifier for the origin (e.g. filename, URL, job title).
        doc_id:     Unique document ID. Auto-generated if not provided.
        metadata:   Arbitrary key-value pairs forwarded to ChromaDB.
    """

    content: str
    source: str
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content or not self.content.strip():
            raise ValueError("KnowledgeDocument.content must not be empty.")
        if not self.source or not self.source.strip():
            raise ValueError("KnowledgeDocument.source must not be empty.")


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """
    Immutable result returned after successfully ingesting a document.

    Attributes:
        doc_id:         ID of the ingested document.
        source:         Source identifier.
        chunk_count:    Number of chunks stored in ChromaDB.
        ingested_at:    UTC timestamp of ingestion.
    """

    doc_id: str
    source: str
    chunk_count: int
    ingested_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """
    A single chunk returned by a similarity search.

    Attributes:
        content:    Text of the chunk.
        source:     Source document identifier.
        doc_id:     Parent document ID.
        score:      Cosine similarity score (higher = more relevant).
        metadata:   Full metadata dict from ChromaDB.
    """

    content: str
    source: str
    doc_id: str
    score: float
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SearchResult:
    """
    Aggregated result of a similarity search.

    Attributes:
        query:      The original search query.
        chunks:     Ranked list of retrieved chunks.
        context:    Chunks concatenated into a single context string,
                    ready to be injected into an LLM prompt.
    """

    query: str
    chunks: list[RetrievedChunk]
    context: str = field(init=False)

    def __post_init__(self) -> None:
        joined = "\n\n---\n\n".join(c.content for c in self.chunks)
        object.__setattr__(self, "context", joined)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


class DocumentChunker:
    """
    Splits raw text into overlapping chunks using LangChain's
    :class:`RecursiveCharacterTextSplitter`.

    The chunker is stateless and can be shared across requests.

    Args:
        chunk_size:     Target character length per chunk.
        chunk_overlap:  Character overlap between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def split(self, document: KnowledgeDocument) -> list[Document]:
        """
        Split a :class:`KnowledgeDocument` into LangChain ``Document`` chunks.

        Each chunk carries the parent document's metadata enriched with
        ``doc_id`` and ``source`` so every vector in ChromaDB is traceable.

        Args:
            document: The knowledge document to split.

        Returns:
            List of LangChain :class:`Document` objects, one per chunk.
        """
        base_metadata: dict[str, Any] = {
            **document.metadata,
            "doc_id": document.doc_id,
            "source": document.source,
        }

        chunks: list[Document] = self._splitter.create_documents(
            texts=[document.content],
            metadatas=[base_metadata],
        )

        logger.debug(
            "DocumentChunker: split '%s' into %d chunks  (size=%d, overlap=%d)",
            document.source,
            len(chunks),
            self._splitter._chunk_size,
            self._splitter._chunk_overlap,
        )

        return chunks


# ---------------------------------------------------------------------------
# Vector store wrapper
# ---------------------------------------------------------------------------


class ChromaStore:
    """
    Thin wrapper around LangChain's :class:`Chroma` vector store.

    Handles all ChromaDB interactions — add, search, and delete — while
    keeping the rest of the service decoupled from the underlying store.

    Args:
        embeddings:       LangChain :class:`Embeddings` implementation.
        collection_name:  ChromaDB collection name.
        persist_directory: Optional path for persistent storage.
                           Pass ``None`` for an in-memory (ephemeral) store.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        persist_directory: str | None = None,
    ) -> None:
        self._embeddings = embeddings
        self._collection_name = collection_name

        chroma_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "embedding_function": embeddings,
        }
        if persist_directory:
            chroma_kwargs["persist_directory"] = persist_directory

        self._store: Chroma = Chroma(**chroma_kwargs)

        logger.info(
            "ChromaStore: initialised  collection='%s'  persist='%s'",
            collection_name,
            persist_directory or "in-memory",
        )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add_chunks(self, chunks: list[Document]) -> list[str]:
        """
        Embed and persist a list of LangChain Documents.

        Args:
            chunks: Pre-split document chunks to store.

        Returns:
            List of ChromaDB-assigned vector IDs.

        Raises:
            DocumentIngestionError: On any ChromaDB write failure.
        """
        try:
            ids: list[str] = self._store.add_documents(chunks)
        except Exception as exc:
            logger.error("ChromaStore.add_chunks: write failed — %s", exc, exc_info=True)
            raise DocumentIngestionError(
                reason="chroma_write_error",
                detail=f"Failed to persist chunks to ChromaDB: {exc}",
            ) from exc

        logger.debug("ChromaStore: stored %d vectors", len(ids))
        return ids

    def delete_by_doc_id(self, doc_id: str) -> None:
        """
        Remove all vectors that belong to a given document ID.

        Uses ChromaDB's ``where`` filter on the ``doc_id`` metadata field.

        Args:
            doc_id: The parent document's ID to purge.
        """
        try:
            self._store._collection.delete(where={"doc_id": {"$eq": doc_id}})
        except Exception as exc:
            logger.warning(
                "ChromaStore.delete_by_doc_id: could not delete doc_id='%s' — %s",
                doc_id,
                exc,
            )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[tuple[Document, float]]:
        """
        Return the *top_k* most similar chunks for the given query.

        Args:
            query:           Natural-language search query.
            top_k:           Number of results to return.
            filter_metadata: Optional ChromaDB ``where`` filter dict.

        Returns:
            List of ``(Document, score)`` tuples ordered by descending similarity.

        Raises:
            SearchError: On any ChromaDB read failure.
        """
        kwargs: dict[str, Any] = {"k": top_k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        try:
            results: list[tuple[Document, float]] = (
                self._store.similarity_search_with_relevance_scores(query, **kwargs)
            )
        except Exception as exc:
            logger.error("ChromaStore.similarity_search: query failed — %s", exc, exc_info=True)
            raise SearchError(
                detail=f"Similarity search failed: {exc}"
            ) from exc

        return results


# ---------------------------------------------------------------------------
# RAG Service
# ---------------------------------------------------------------------------


class RAGService:
    """
    Orchestration layer for the HireMind knowledge base.

    Wires together :class:`DocumentChunker` and :class:`ChromaStore` to
    provide a single, clean API for the rest of the application.

    The service is stateless (all state lives in ChromaDB) and thread-safe.
    A module-level singleton is provided via :func:`get_rag_service`.

    Args:
        chunker:      Pre-configured :class:`DocumentChunker`.
        store:        Pre-configured :class:`ChromaStore`.

    Example::

        service = RAGService()

        result = service.add_document(
            KnowledgeDocument(content="...", source="job_ai_engineer.txt")
        )

        search = service.search("machine learning experience")
        print(search.context)
    """

    def __init__(
        self,
        chunker: DocumentChunker | None = None,
        store: ChromaStore | None = None,
    ) -> None:
        self._chunker: DocumentChunker = chunker or DocumentChunker()
        self._store: ChromaStore = store or ChromaStore(
            embeddings=_build_default_embeddings()
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_document(self, document: KnowledgeDocument) -> IngestionResult:
        """
        Ingest a knowledge document into the vector store.

        Splits the document into chunks, generates embeddings, and
        persists everything to ChromaDB.

        Args:
            document: The knowledge document to ingest.

        Returns:
            :class:`IngestionResult` with chunk count and timestamp.

        Raises:
            TypeError:               If ``document`` is not a :class:`KnowledgeDocument`.
            DocumentIngestionError:  On chunking or ChromaDB write failure.
        """
        if not isinstance(document, KnowledgeDocument):
            raise TypeError(
                f"document must be KnowledgeDocument, got {type(document).__name__!r}."
            )

        logger.info(
            "RAGService.add_document: ingesting  doc_id='%s'  source='%s'  chars=%d",
            document.doc_id,
            document.source,
            len(document.content),
        )

        chunks = self._chunker.split(document)

        if not chunks:
            raise DocumentIngestionError(
                reason="empty_document",
                detail=f"Document '{document.source}' produced no chunks after splitting.",
            )

        self._store.add_chunks(chunks)

        result = IngestionResult(
            doc_id=document.doc_id,
            source=document.source,
            chunk_count=len(chunks),
        )

        logger.info(
            "RAGService.add_document: done  doc_id='%s'  chunks=%d  at=%s",
            result.doc_id,
            result.chunk_count,
            result.ingested_at.isoformat(),
        )

        return result

    def add_documents(self, documents: list[KnowledgeDocument]) -> list[IngestionResult]:
        """
        Batch-ingest multiple knowledge documents.

        Processes each document individually and collects results.
        Failures on individual documents are logged and re-raised
        without aborting the entire batch.

        Args:
            documents: List of :class:`KnowledgeDocument` instances.

        Returns:
            List of :class:`IngestionResult` objects, one per document.

        Raises:
            DocumentIngestionError: On the first document that fails.
        """
        if not documents:
            raise ValueError("documents list must not be empty.")

        results: list[IngestionResult] = []
        for doc in documents:
            result = self.add_document(doc)
            results.append(result)

        logger.info("RAGService.add_documents: ingested %d documents", len(results))
        return results

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        filter_metadata: dict[str, Any] | None = None,
    ) -> SearchResult:
        """
        Retrieve the most relevant knowledge chunks for a query.

        Embeds the query, runs a cosine similarity search in ChromaDB,
        and returns a :class:`SearchResult` whose ``.context`` attribute
        can be injected directly into an LLM prompt.

        Args:
            query:           Natural-language question or keyword string.
            top_k:           Maximum number of chunks to return.
            filter_metadata: Optional metadata filter (e.g. ``{"source": "jd_backend.txt"}``).

        Returns:
            :class:`SearchResult` with ranked chunks and a concatenated context.

        Raises:
            ValueError:  If ``query`` is blank.
            SearchError: On ChromaDB read failure.
        """
        if not query or not query.strip():
            raise ValueError("query must not be empty.")

        logger.info("RAGService.search: query='%.80s'  top_k=%d", query, top_k)

        raw: list[tuple[Document, float]] = self._store.similarity_search(
            query=query,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        chunks: list[RetrievedChunk] = [
            RetrievedChunk(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                doc_id=doc.metadata.get("doc_id", "unknown"),
                score=score,
                metadata=doc.metadata,
            )
            for doc, score in raw
        ]

        result = SearchResult(query=query, chunks=chunks)

        logger.info(
            "RAGService.search: returned %d chunks  top_score=%.4f",
            len(chunks),
            chunks[0].score if chunks else 0.0,
        )

        return result

    def delete_document(self, doc_id: str) -> None:
        """
        Remove all vector embeddings associated with a document.

        Args:
            doc_id: The document ID previously returned by :meth:`add_document`.
        """
        logger.info("RAGService.delete_document: removing doc_id='%s'", doc_id)
        self._store.delete_by_doc_id(doc_id)
        logger.info("RAGService.delete_document: done  doc_id='%s'", doc_id)


# ---------------------------------------------------------------------------
# Embedding factory (swap for Gemini / OpenAI embeddings as needed)
# ---------------------------------------------------------------------------


def _build_default_embeddings() -> Embeddings:
    """
    Build the default embedding model.

    Uses HuggingFace's ``all-MiniLM-L6-v2`` — a compact, fast sentence
    transformer that runs locally with no API key required.

    To swap for Gemini embeddings::

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    To swap for OpenAI embeddings::

        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-small")

    Returns:
        A LangChain :class:`Embeddings` instance.
    """
    logger.info("Building default HuggingFace embeddings (all-MiniLM-L6-v2)")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# FastAPI dependency factory
# ---------------------------------------------------------------------------


def get_rag_service() -> RAGService:
    """
    FastAPI dependency that returns the shared :class:`RAGService` singleton.

    Declare once at module level; FastAPI will reuse the same instance
    across all requests, giving you a single persistent ChromaDB connection.

    Example::

        from fastapi import Depends
        from services.rag import RAGService, get_rag_service

        @router.post("/knowledge")
        async def add_knowledge(
            service: RAGService = Depends(get_rag_service),
        ):
            ...
    """
    return _shared_rag_service


# Module-level singleton — instantiated once at import time.
_shared_rag_service: RAGService = RAGService()