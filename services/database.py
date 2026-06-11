"""
services/database.py

Responsibility:
    - Initialize and manage the SQLite database connection.
    - Persist evaluation results after each hiring pipeline run.
    - Provide clean query methods for retrieving stored evaluations.
    - Expose a FastAPI dependency factory.

Out of scope:
    - PDF parsing.
    - RAG / ChromaDB operations.
    - Gemini / LLM calls.
    - FastAPI routes.

Schema:
    evaluations — one row per candidate evaluation.
    evaluation_strengths — normalized strengths (one row per item).
    evaluation_weaknesses — normalized weaknesses (one row per item).
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DB_PATH: Path = Path("hiremind.db")

# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class DatabaseError(Exception):
    """Base exception for all database failures."""


class RecordNotFoundError(DatabaseError):
    """Raised when a requested record does not exist."""

    def __init__(self, evaluation_id: int) -> None:
        super().__init__(f"Evaluation with id={evaluation_id} not found.")
        self.evaluation_id = evaluation_id


class PersistenceError(DatabaseError):
    """Raised when a write operation fails."""


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EvaluationRecord:
    """
    Immutable record representing a single stored evaluation.

    Attributes:
        id:                 Auto-incremented primary key (None before insert).
        job_title:          Job title used for retrieval.
        cv_text:            Full extracted CV text.
        cv_char_count:      Number of characters in the CV.
        rag_context:        RAG context string used for evaluation.
        rag_context_length: Length of the RAG context.
        final_score:        Numeric fit score (0–100).
        hiring_decision:    'Yes', 'No', or 'Consider'.
        strengths:          List of candidate strengths.
        weaknesses:         List of missing skills / gaps.
        summary:            Justification paragraph.
        created_at:         UTC timestamp of record creation.
    """

    job_title: str
    cv_text: str
    cv_char_count: int
    rag_context: str
    rag_context_length: int
    final_score: int
    hiring_decision: str
    strengths: list[str]
    weaknesses: list[str]
    summary: str
    id: int | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    """
    Lightweight summary for list views (no CV text or RAG context).

    Attributes:
        id:               Primary key.
        job_title:        Job title.
        cv_char_count:    Characters in the CV.
        final_score:      Fit score (0–100).
        hiring_decision:  'Yes', 'No', or 'Consider'.
        strengths:        List of strengths.
        weaknesses:       List of weaknesses.
        summary:          Evaluation justification.
        created_at:       UTC timestamp.
    """

    id: int
    job_title: str
    cv_char_count: int
    final_score: int
    hiring_decision: str
    strengths: list[str]
    weaknesses: list[str]
    summary: str
    created_at: datetime


# ---------------------------------------------------------------------------
# Database manager
# ---------------------------------------------------------------------------


class DatabaseManager:
    """
    Manages the SQLite connection and all database operations.

    The manager is thread-safe when used with ``check_same_thread=False``
    and a module-level singleton. Each public method opens a short-lived
    connection context to avoid long-held locks.

    Args:
        db_path: Path to the SQLite file. Created on first use.

    Example::

        db = DatabaseManager()
        db.initialize()

        record_id = db.save_evaluation(record)
        summaries = db.list_evaluations()
        record    = db.get_evaluation(record_id)
    """

    def __init__(self, db_path: Path = DEFAULT_DB_PATH) -> None:
        self._db_path = db_path
        logger.info("DatabaseManager: db_path='%s'", self._db_path.resolve())

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Create tables if they do not already exist.

        Safe to call multiple times (idempotent). Call once at app startup.

        Raises:
            DatabaseError: If table creation fails.
        """
        logger.info("DatabaseManager: initializing schema...")
        with self._connect() as conn:
            self._create_tables(conn)
        logger.info("DatabaseManager: schema ready.")

    @staticmethod
    def _create_tables(conn: sqlite3.Connection) -> None:
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS evaluations (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                job_title           TEXT    NOT NULL,
                cv_text             TEXT    NOT NULL,
                cv_char_count       INTEGER NOT NULL,
                rag_context         TEXT    NOT NULL,
                rag_context_length  INTEGER NOT NULL,
                final_score         INTEGER NOT NULL CHECK (final_score BETWEEN 0 AND 100),
                hiring_decision     TEXT    NOT NULL CHECK (hiring_decision IN ('Yes', 'No', 'Consider')),
                summary             TEXT    NOT NULL,
                created_at          TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_strengths (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id   INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                strength        TEXT    NOT NULL,
                position        INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS evaluation_weaknesses (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                evaluation_id   INTEGER NOT NULL REFERENCES evaluations(id) ON DELETE CASCADE,
                weakness        TEXT    NOT NULL,
                position        INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_evaluations_job_title
                ON evaluations(job_title);

            CREATE INDEX IF NOT EXISTS idx_evaluations_hiring_decision
                ON evaluations(hiring_decision);

            CREATE INDEX IF NOT EXISTS idx_evaluations_created_at
                ON evaluations(created_at);

            CREATE INDEX IF NOT EXISTS idx_strengths_evaluation_id
                ON evaluation_strengths(evaluation_id);

            CREATE INDEX IF NOT EXISTS idx_weaknesses_evaluation_id
                ON evaluation_weaknesses(evaluation_id);
        """)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def save_evaluation(self, record: EvaluationRecord) -> int:
        """
        Persist an evaluation record and its strengths/weaknesses.

        Runs inside a single transaction — either all rows are written
        or none are (rollback on failure).

        Args:
            record: The :class:`EvaluationRecord` to persist.

        Returns:
            The auto-generated primary key of the new evaluation row.

        Raises:
            PersistenceError: If the insert fails for any reason.
        """
        try:
            with self._connect() as conn:
                evaluation_id = self._insert_evaluation(conn, record)
                self._insert_strengths(conn, evaluation_id, record.strengths)
                self._insert_weaknesses(conn, evaluation_id, record.weaknesses)
                conn.commit()

            logger.info(
                "DatabaseManager.save_evaluation: saved  id=%d  job='%s'  score=%d  decision=%s",
                evaluation_id,
                record.job_title,
                record.final_score,
                record.hiring_decision,
            )
            return evaluation_id

        except sqlite3.Error as exc:
            logger.error("DatabaseManager.save_evaluation: failed — %s", exc, exc_info=True)
            raise PersistenceError(f"Failed to save evaluation: {exc}") from exc

    @staticmethod
    def _insert_evaluation(conn: sqlite3.Connection, record: EvaluationRecord) -> int:
        cursor = conn.execute(
            """
            INSERT INTO evaluations (
                job_title, cv_text, cv_char_count,
                rag_context, rag_context_length,
                final_score, hiring_decision,
                summary, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.job_title,
                record.cv_text,
                record.cv_char_count,
                record.rag_context,
                record.rag_context_length,
                record.final_score,
                record.hiring_decision,
                record.summary,
                record.created_at.isoformat(),
            ),
        )
        return cursor.lastrowid  # type: ignore[return-value]

    @staticmethod
    def _insert_strengths(
        conn: sqlite3.Connection,
        evaluation_id: int,
        strengths: list[str],
    ) -> None:
        conn.executemany(
            "INSERT INTO evaluation_strengths (evaluation_id, strength, position) VALUES (?, ?, ?)",
            [(evaluation_id, s, i) for i, s in enumerate(strengths)],
        )

    @staticmethod
    def _insert_weaknesses(
        conn: sqlite3.Connection,
        evaluation_id: int,
        weaknesses: list[str],
    ) -> None:
        conn.executemany(
            "INSERT INTO evaluation_weaknesses (evaluation_id, weakness, position) VALUES (?, ?, ?)",
            [(evaluation_id, w, i) for i, w in enumerate(weaknesses)],
        )

    def delete_evaluation(self, evaluation_id: int) -> None:
        """
        Delete an evaluation and its related strengths/weaknesses.

        Foreign key CASCADE handles child rows automatically.

        Args:
            evaluation_id: Primary key of the evaluation to delete.

        Raises:
            RecordNotFoundError: If no record with that ID exists.
            PersistenceError:    On database write failure.
        """
        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    "DELETE FROM evaluations WHERE id = ?", (evaluation_id,)
                )
                conn.commit()

            if cursor.rowcount == 0:
                raise RecordNotFoundError(evaluation_id)

            logger.info("DatabaseManager.delete_evaluation: deleted id=%d", evaluation_id)

        except RecordNotFoundError:
            raise
        except sqlite3.Error as exc:
            logger.error("DatabaseManager.delete_evaluation: failed — %s", exc, exc_info=True)
            raise PersistenceError(f"Failed to delete evaluation {evaluation_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_evaluation(self, evaluation_id: int) -> EvaluationRecord:
        """
        Fetch a single evaluation by primary key, including strengths/weaknesses.

        Args:
            evaluation_id: Primary key.

        Returns:
            Fully populated :class:`EvaluationRecord`.

        Raises:
            RecordNotFoundError: If no record exists with that ID.
            DatabaseError:       On read failure.
        """
        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT * FROM evaluations WHERE id = ?", (evaluation_id,)
                ).fetchone()

                if row is None:
                    raise RecordNotFoundError(evaluation_id)

                strengths  = self._fetch_strengths(conn, evaluation_id)
                weaknesses = self._fetch_weaknesses(conn, evaluation_id)

            return self._row_to_record(row, strengths, weaknesses)

        except RecordNotFoundError:
            raise
        except sqlite3.Error as exc:
            logger.error("DatabaseManager.get_evaluation: failed — %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to fetch evaluation {evaluation_id}: {exc}") from exc

    def list_evaluations(
        self,
        job_title: str | None = None,
        hiring_decision: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[EvaluationSummary]:
        """
        List evaluations with optional filters, ordered by most recent first.

        This method returns :class:`EvaluationSummary` objects (no CV text or
        RAG context) to keep response payloads small for list views.

        Args:
            job_title:        Filter by exact job title (case-insensitive).
            hiring_decision:  Filter by decision ('Yes', 'No', 'Consider').
            limit:            Maximum rows to return (default 50).
            offset:           Rows to skip for pagination (default 0).

        Returns:
            List of :class:`EvaluationSummary` ordered newest-first.

        Raises:
            DatabaseError: On read failure.
        """
        try:
            clauses: list[str] = []
            params: list[object] = []

            if job_title:
                clauses.append("LOWER(job_title) = LOWER(?)")
                params.append(job_title)

            if hiring_decision:
                clauses.append("hiring_decision = ?")
                params.append(hiring_decision)

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

            query = f"""
                SELECT id, job_title, cv_char_count, final_score,
                       hiring_decision, summary, created_at
                FROM evaluations
                {where}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            with self._connect() as conn:
                rows = conn.execute(query, params).fetchall()
                summaries: list[EvaluationSummary] = []

                for row in rows:
                    eval_id = row["id"]
                    strengths  = self._fetch_strengths(conn, eval_id)
                    weaknesses = self._fetch_weaknesses(conn, eval_id)
                    summaries.append(self._row_to_summary(row, strengths, weaknesses))

            logger.info(
                "DatabaseManager.list_evaluations: returned %d records  "
                "job_title='%s'  decision='%s'",
                len(summaries),
                job_title or "*",
                hiring_decision or "*",
            )
            return summaries

        except sqlite3.Error as exc:
            logger.error("DatabaseManager.list_evaluations: failed — %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to list evaluations: {exc}") from exc

    def count_evaluations(
        self,
        job_title: str | None = None,
        hiring_decision: str | None = None,
    ) -> int:
        """
        Count evaluations matching optional filters.

        Useful for pagination metadata.

        Args:
            job_title:       Filter by job title.
            hiring_decision: Filter by decision.

        Returns:
            Total count of matching rows.
        """
        try:
            clauses: list[str] = []
            params: list[object] = []

            if job_title:
                clauses.append("LOWER(job_title) = LOWER(?)")
                params.append(job_title)

            if hiring_decision:
                clauses.append("hiring_decision = ?")
                params.append(hiring_decision)

            where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

            with self._connect() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) as cnt FROM evaluations {where}", params
                ).fetchone()

            return row["cnt"] if row else 0

        except sqlite3.Error as exc:
            logger.error("DatabaseManager.count_evaluations: failed — %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to count evaluations: {exc}") from exc

    def get_stats(self) -> dict[str, object]:
        """
        Return aggregate statistics across all stored evaluations.

        Returns a dict with:
            - total_evaluations
            - avg_score
            - decisions: {'Yes': N, 'No': N, 'Consider': N}
            - top_jobs: list of {job_title, count} ordered by count desc

        Raises:
            DatabaseError: On read failure.
        """
        try:
            with self._connect() as conn:
                totals = conn.execute("""
                    SELECT
                        COUNT(*)               AS total,
                        ROUND(AVG(final_score), 1) AS avg_score
                    FROM evaluations
                """).fetchone()

                decisions_rows = conn.execute("""
                    SELECT hiring_decision, COUNT(*) AS cnt
                    FROM evaluations
                    GROUP BY hiring_decision
                """).fetchall()

                top_jobs_rows = conn.execute("""
                    SELECT job_title, COUNT(*) AS cnt
                    FROM evaluations
                    GROUP BY job_title
                    ORDER BY cnt DESC
                    LIMIT 10
                """).fetchall()

            decisions = {row["hiring_decision"]: row["cnt"] for row in decisions_rows}
            top_jobs  = [
                {"job_title": row["job_title"], "count": row["cnt"]}
                for row in top_jobs_rows
            ]

            return {
                "total_evaluations": totals["total"] or 0,
                "avg_score": totals["avg_score"] or 0.0,
                "decisions": {
                    "Yes":     decisions.get("Yes", 0),
                    "No":      decisions.get("No", 0),
                    "Consider": decisions.get("Consider", 0),
                },
                "top_jobs": top_jobs,
            }

        except sqlite3.Error as exc:
            logger.error("DatabaseManager.get_stats: failed — %s", exc, exc_info=True)
            raise DatabaseError(f"Failed to compute stats: {exc}") from exc

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection, None, None]:
        """Open a short-lived connection with row_factory set to dict-like rows."""
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    @staticmethod
    def _fetch_strengths(conn: sqlite3.Connection, evaluation_id: int) -> list[str]:
        rows = conn.execute(
            "SELECT strength FROM evaluation_strengths WHERE evaluation_id = ? ORDER BY position",
            (evaluation_id,),
        ).fetchall()
        return [row["strength"] for row in rows]

    @staticmethod
    def _fetch_weaknesses(conn: sqlite3.Connection, evaluation_id: int) -> list[str]:
        rows = conn.execute(
            "SELECT weakness FROM evaluation_weaknesses WHERE evaluation_id = ? ORDER BY position",
            (evaluation_id,),
        ).fetchall()
        return [row["weakness"] for row in rows]

    @staticmethod
    def _row_to_record(
        row: sqlite3.Row,
        strengths: list[str],
        weaknesses: list[str],
    ) -> EvaluationRecord:
        return EvaluationRecord(
            id=row["id"],
            job_title=row["job_title"],
            cv_text=row["cv_text"],
            cv_char_count=row["cv_char_count"],
            rag_context=row["rag_context"],
            rag_context_length=row["rag_context_length"],
            final_score=row["final_score"],
            hiring_decision=row["hiring_decision"],
            strengths=strengths,
            weaknesses=weaknesses,
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    @staticmethod
    def _row_to_summary(
        row: sqlite3.Row,
        strengths: list[str],
        weaknesses: list[str],
    ) -> EvaluationSummary:
        return EvaluationSummary(
            id=row["id"],
            job_title=row["job_title"],
            cv_char_count=row["cv_char_count"],
            final_score=row["final_score"],
            hiring_decision=row["hiring_decision"],
            strengths=strengths,
            weaknesses=weaknesses,
            summary=row["summary"],
            created_at=datetime.fromisoformat(row["created_at"]),
        )


# ---------------------------------------------------------------------------
# FastAPI dependency factory
# ---------------------------------------------------------------------------

_shared_db_manager: DatabaseManager = DatabaseManager()


def get_db_manager() -> DatabaseManager:
    """
    FastAPI dependency — returns the shared :class:`DatabaseManager` singleton.

    Example::

        from fastapi import Depends
        from services.database import DatabaseManager, get_db_manager

        @router.get("/evaluations")
        def list_evals(db: DatabaseManager = Depends(get_db_manager)):
            return db.list_evaluations()
    """
    return _shared_db_manager
