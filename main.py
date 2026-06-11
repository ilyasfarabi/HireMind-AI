"""
main.py — HireMind AI entry point.

IMPORTANT: load_dotenv() must run before ANY service import.
Services create singletons at import time that need GOOGLE_API_KEY.
"""

from __future__ import annotations

# ── Load .env FIRST — before any service imports ──────────────────────────
import logging
import os
from dotenv import load_dotenv

load_dotenv()  # يقرأ .env قبل أي import للـ services

# ── Validate critical env vars early ──────────────────────────────────────
_GOOGLE_KEY = os.getenv("GOOGLE_API_KEY")
if not _GOOGLE_KEY:
    raise RuntimeError(
        "\n\n[HireMind] GOOGLE_API_KEY not set!\n"
        "Add this line to your .env file:\n"
        "GOOGLE_API_KEY=your_actual_key_here\n"
    )

# ── Now safe to import services ────────────────────────────────────────────
from fastapi import FastAPI
from routes.evaluation_routes import router as evaluation_router
from routes.history_routes import router as history_router
from services.database import get_db_manager
from services.knowledge_loader import load_knowledge

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="HireMind AI",
    version="1.0.0",
    description="AI-powered CV screening and candidate evaluation system.",
)

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


@app.on_event("startup")
async def startup_event() -> None:
    """Initialize database schema and load knowledge base on startup."""

    # 1. Initialize SQLite schema (idempotent — safe to run every startup)
    logger.info("startup: initializing database...")
    db = get_db_manager()
    db.initialize()
    logger.info("startup: database ready.")

    # 2. Load knowledge base into ChromaDB
    logger.info("startup: loading knowledge base...")
    try:
        count = load_knowledge()
        logger.info("startup: knowledge base ready  documents=%d", count)
    except FileNotFoundError as exc:
        logger.warning("startup: %s", exc)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

app.include_router(evaluation_router)
app.include_router(history_router)

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/", tags=["Health"])
def root() -> dict[str, str]:
    return {"status": "ok", "service": "HireMind AI"}