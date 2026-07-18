"""
Router: /query
  POST / — run a full RAG query against indexed documents
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from backend.schemas import ChunkInfo, QueryRequest, QueryResponse
from backend.dependencies import get_pipeline
from rag.pipeline import RAGPipeline
from utils.logger import get_logger

logger = get_logger("router.query")
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "/",
    response_model=QueryResponse,
    summary="RAG query — ask a question about indexed documents",
)
async def rag_query(req: QueryRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    logger.info(f"Query received: '{req.question[:80]}'")

    # Swap prompt version on the fly without re-creating the whole pipeline
    from rag.prompt_manager import PromptManager
    try:
        pipeline.prompt_manager = PromptManager(version=req.prompt_version)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))

    try:
        resp = pipeline.query(
            question=req.question,
            top_k=req.top_k,
            rerank_top_k=req.rerank_top_k,
            use_mmr=req.use_mmr,
        )
    except Exception as exc:
        logger.error(f"Query failed: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))

    chunks = [
        ChunkInfo(
            id=c["id"],
            document=c["document"],
            metadata=c["metadata"],
            relevance_score=c.get("relevance_score", 0.0),
            distance=c.get("distance", 0.0),
        )
        for c in resp.retrieved_chunks
    ]

    return QueryResponse(
        question=resp.query,
        answer=resp.answer,
        retrieved_chunks=chunks,
        prompt_version=resp.prompt_version,
        latency_sec=round(resp.latency_sec, 3),
        model=resp.model,
        tokens_used=resp.tokens_used,
    )
