"""
Router: /documents
  POST /index        — upload a PDF and index it
  GET  /             — list all indexed documents
  DELETE /{filename} — remove a document from the store
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.schemas import DocumentInfo, DocumentListResponse, IndexResponse
from backend.dependencies import get_pipeline, get_store
from utils.logger import get_logger

logger = get_logger("router.documents")
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/index",
    response_model=IndexResponse,
    summary="Upload and index a PDF",
)
async def index_document(
    file:     UploadFile = File(..., description="PDF file to index"),
    pipeline  = Depends(get_pipeline),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only PDF files are supported.",
        )

    # Save upload to a temp file so PDFExtractor can open it by path
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        count = pipeline.index_document(tmp_path, source_name=file.filename)
        logger.info(f"Indexed {count} chunks from '{file.filename}'")
        return IndexResponse(
            filename=file.filename,
            chunks=count,
            message=f"Successfully indexed {count} chunks.",
        )
    except Exception as exc:
        logger.error(f"Indexing failed for {file.filename}: {exc}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        )
    finally:
        tmp_path.unlink(missing_ok=True)


@router.get(
    "/",
    response_model=DocumentListResponse,
    summary="List indexed documents",
)
async def list_documents(store = Depends(get_store)):
    """
    Returns all unique source documents stored in ChromaDB plus their chunk counts.
    """
    try:
        result = store._collection.get(include=["metadatas"])
        metadatas = result.get("metadatas") or []

        counts: dict[str, int] = {}
        for m in metadatas:
            src = m.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1

        docs = [DocumentInfo(filename=fn, chunks=c) for fn, c in counts.items()]
        return DocumentListResponse(
            documents=docs,
            total_chunks=sum(counts.values()),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.delete(
    "/{filename}",
    summary="Delete all chunks for a document",
)
async def delete_document(filename: str, store = Depends(get_store)):
    try:
        store.delete_source(filename)
        return {"deleted": filename, "status": "ok"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
