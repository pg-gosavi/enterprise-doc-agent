"""
FastAPI Backend — Enterprise Document Intelligence Agent
========================================================

Endpoints
---------
  GET  /health                  — liveness + stats
  POST /documents/index         — upload & index a PDF
  GET  /documents/              — list indexed documents
  DELETE /documents/{filename}  — remove a document
  POST /query/                  — RAG query
  POST /evaluate/               — run QA evaluation dataset

Run
---
  uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when running from any working directory
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.routers import documents, query, evaluate
from backend.dependencies import get_pipeline, get_store
from backend.schemas import HealthResponse
from config import LLM_MODEL
from utils.logger import get_logger

logger = get_logger("api")

# ── App factory ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise Document Intelligence Agent",
    description=(
        "RAG pipeline backed by ChromaDB + Llama 3.1 8B via Groq. "
        "Upload PDFs, ask questions, run evaluations."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Streamlit (port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to ["http://localhost:8501"] in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(evaluate.router)


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """Liveness + readiness probe."""
    try:
        store        = get_store()
        stats        = store.stats()
        total_chunks = stats.get("total_chunks", 0)
    except Exception:
        total_chunks = 0

    return HealthResponse(
        status="ok",
        model=LLM_MODEL,
        total_chunks=total_chunks,
    )


# ── Startup event ──────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Pre-warm the pipeline so the first request is not slow."""
    logger.info("Warming up RAG pipeline…")
    try:
        get_pipeline()
        logger.info("Pipeline ready.")
    except Exception as exc:
        logger.error(f"Pipeline warm-up failed: {exc}")


# ── Global error handler ───────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )


if __name__ == "__main__":
    import uvicorn
    from config import API_HOST, API_PORT
    uvicorn.run("backend.api:app", host=API_HOST, port=API_PORT, reload=True)
