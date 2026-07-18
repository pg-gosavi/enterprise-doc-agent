"""
RAG Pipeline (Groq / Llama 3.1 8B)
====================================
Uses groq.Groq client (OpenAI-compatible chat completions).
System prompt is passed as role="system" in the messages list.
Token count from response.usage.prompt_tokens + completion_tokens.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from groq import Groq

from config import (
    GROQ_API_KEY,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    RERANK_TOP_K,
    TOP_K,
    GROQ_TPM_LIMIT,
)
from embeddings.embedder import Embedder
from ingestion.chunker import SemanticChunker
from ingestion.pdf_extractor import PDFExtractor
from rag.prompt_manager import PromptManager
from vector_store.chroma_store import ChromaStore
from utils.logger import get_logger

logger = get_logger("pipeline")


@dataclass
class RagResponse:
    query:             str
    answer:            str
    retrieved_chunks:  list[dict]
    prompt_version:    str
    latency_sec:       float
    model:             str
    tokens_used:       int = 0
    metadata:          dict = field(default_factory=dict)

    def __str__(self) -> str:
        sep = "─" * 60
        chunk_lines = "\n".join(
            f"  [{i+1}] {c['metadata'].get('source','?')} p.{c['metadata'].get('page_num','?')} "
            f"(relevance={c.get('relevance_score', 0):.3f})"
            for i, c in enumerate(self.retrieved_chunks)
        )
        return (
            f"\n{sep}\n"
            f"QUERY   : {self.query}\n"
            f"MODEL   : {self.model} via Groq  |  "
            f"LATENCY: {self.latency_sec:.2f}s  |  TOKENS: {self.tokens_used}\n"
            f"CHUNKS  :\n{chunk_lines}\n"
            f"{sep}\n"
            f"ANSWER  :\n{self.answer}\n"
            f"{sep}"
        )


class RAGPipeline:
    def __init__(self, prompt_version: str | None = None):
        self.embedder       = Embedder()
        self.chunker        = SemanticChunker()
        self.store          = ChromaStore()
        self.prompt_manager = PromptManager(version=prompt_version)
        self._llm           = Groq(api_key=GROQ_API_KEY)
        logger.info(f"RAG Pipeline ready — LLM: {LLM_MODEL} via Groq")

    def index_document(self, pdf_path: str | Path) -> int:
        pdf_path  = Path(pdf_path)
        extractor = PDFExtractor(pdf_path)
        pages     = extractor.extract_pages()
        chunks    = self.chunker.chunk_pages(pages)
        self.store.upsert_chunks(chunks, self.embedder)
        logger.info(f"Indexed {len(chunks)} chunks from {pdf_path.name}")
        return len(chunks)

    def index_directory(self, dir_path: str | Path) -> dict[str, int]:
        dir_path = Path(dir_path)
        results: dict[str, int] = {}
        for pdf in dir_path.glob("*.pdf"):
            try:
                results[pdf.name] = self.index_document(pdf)
            except Exception as exc:
                logger.error(f"Failed: {pdf.name}: {exc}")
                results[pdf.name] = -1
        return results

    def query(
        self,
        question:     str,
        top_k:        int = TOP_K,
        rerank_top_k: int = RERANK_TOP_K,
        where_filter: dict | None = None,
        use_mmr:      bool = True,
    ) -> RagResponse:
        t0 = time.perf_counter()

        chunks = self.store.search(
            query=question, embedder=self.embedder,
            top_k=top_k, rerank_top_k=rerank_top_k,
            where=where_filter, use_mmr=use_mmr,
        )

        if not chunks:
            return RagResponse(
                query=question,
                answer="INSUFFICIENT CONTEXT: No documents indexed yet.",
                retrieved_chunks=[],
                prompt_version=self.prompt_manager.version,
                latency_sec=time.perf_counter() - t0,
                model=LLM_MODEL,
            )

        context      = PromptManager.format_chunks_as_context(chunks)
        system, user = self.prompt_manager.render(context, question)

        # Rate-limit guard (~1 token per 4 chars)
        est = (len(system) + len(user)) // 4
        if est > GROQ_TPM_LIMIT * 0.8:
            logger.warning(f"Large context (~{est} tokens) — may hit Groq TPM limit.")

        response = self._llm.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )

        answer      = response.choices[0].message.content
        tokens_used = (response.usage.prompt_tokens or 0) + (response.usage.completion_tokens or 0)
        latency     = time.perf_counter() - t0

        return RagResponse(
            query=question, answer=answer, retrieved_chunks=chunks,
            prompt_version=self.prompt_manager.version,
            latency_sec=latency, model=LLM_MODEL, tokens_used=tokens_used,
        )

    def query_batch(self, questions: list[str], **kw: Any) -> list[RagResponse]:
        return [self.query(q, **kw) for q in questions]

    def stats(self) -> dict:
        return self.store.stats()
