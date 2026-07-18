"""
Pydantic schemas for all FastAPI request and response bodies.
Keeping them in one file makes it trivial to generate an OpenAPI client.
"""
from __future__ import annotations
from typing import Any
from pydantic import BaseModel, Field


# ── /index ────────────────────────────────────────────────────────────────────

class IndexResponse(BaseModel):
    filename:    str
    chunks:      int
    status:      str = "indexed"
    message:     str = ""


# ── /documents ────────────────────────────────────────────────────────────────

class DocumentInfo(BaseModel):
    filename:   str
    chunks:     int

class DocumentListResponse(BaseModel):
    documents:    list[DocumentInfo]
    total_chunks: int


# ── /query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:     str  = Field(..., min_length=3, description="Natural language question")
    top_k:        int  = Field(8,  ge=1, le=20,  description="Candidates from ChromaDB")
    rerank_top_k: int  = Field(4,  ge=1, le=10,  description="Chunks kept after MMR")
    use_mmr:      bool = Field(True,              description="Enable MMR reranking")
    prompt_version: str = Field("v2",             description="Prompt template version")

class ChunkInfo(BaseModel):
    id:              str
    document:        str
    metadata:        dict[str, Any]
    relevance_score: float
    distance:        float

class QueryResponse(BaseModel):
    question:         str
    answer:           str
    retrieved_chunks: list[ChunkInfo]
    prompt_version:   str
    latency_sec:      float
    model:            str
    tokens_used:      int


# ── /evaluate ─────────────────────────────────────────────────────────────────

class EvalCase(BaseModel):
    question:            str
    relevant_chunk_ids:  list[str] = []
    reference_answer:    str       = ""

class EvalRequest(BaseModel):
    cases:          list[EvalCase]
    run_name:       str  = "eval_run"
    top_k:          int  = 8
    rerank_top_k:   int  = 4
    use_mmr:        bool = True
    prompt_version: str  = "v2"

class EvalCaseResult(BaseModel):
    question:       str
    precision_at_k: float
    recall_at_k:    float
    mrr:            float
    rouge_l:        float
    faithfulness:   float
    latency_sec:    float
    answer:         str
    retrieved_ids:  list[str]

class EvalAggregate(BaseModel):
    n:              int
    precision_at_k: float
    recall_at_k:    float
    mrr:            float
    rouge_l:        float
    faithfulness:   float
    avg_latency_sec:float
    prompt_version: str

class EvalResponse(BaseModel):
    run_name:  str
    aggregate: EvalAggregate
    details:   list[EvalCaseResult]
    report_path: str


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:       str
    model:        str
    total_chunks: int
    api_version:  str = "1.0.0"
