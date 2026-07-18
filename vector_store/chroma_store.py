"""
ChromaDB Vector Store
=====================
Manages a persistent ChromaDB collection for document chunks.

Responsibilities
----------------
- Upsert chunks (idempotent — re-indexing a doc updates existing records)
- Cosine similarity search (top-k)
- Maximal Marginal Relevance (MMR) reranking to balance relevance + diversity
- Metadata filtering (e.g. restrict to a specific source file or page range)
- Collection stats / health check
"""

from __future__ import annotations

from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings

from config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DISTANCE_FN,
    MMR_LAMBDA,
    RERANK_TOP_K,
    TOP_K,
)
from embeddings.embedder import Embedder
from ingestion.chunker import Chunk
from utils.logger import get_logger

logger = get_logger("chroma_store")


def _mmr(
    query_vec:   np.ndarray,
    candidates:  list[dict],
    top_k:       int,
    lambda_mult: float,
) -> list[dict]:
    """
    Maximal Marginal Relevance reranking.

    Iteratively selects the chunk that maximises:
        λ · sim(query, chunk) − (1−λ) · max_sim(chunk, selected)

    Parameters
    ----------
    query_vec   : 1-D query embedding (L2-normalised)
    candidates  : list of dicts with keys 'embedding', 'document', 'metadata', 'id', 'distance'
    top_k       : number of chunks to return
    lambda_mult : relevance-diversity trade-off (1.0 = pure relevance)
    """
    if not candidates:
        return []

    embeddings = np.stack([c["embedding"] for c in candidates])  # (N, D)
    rel_scores = embeddings @ query_vec                           # cosine sim (normalised)

    selected_idx: list[int] = []
    remaining    = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        if not selected_idx:
            # First pick: highest relevance
            pick = int(np.argmax([rel_scores[i] for i in remaining]))
            idx  = remaining[pick]
        else:
            # Subsequent picks: MMR score
            sel_embs   = embeddings[selected_idx]         # (S, D)
            div_scores = (embeddings[remaining] @ sel_embs.T).max(axis=1)  # (R,)
            mmr_scores = (
                lambda_mult * rel_scores[remaining]
                - (1 - lambda_mult) * div_scores
            )
            pick = int(np.argmax(mmr_scores))
            idx  = remaining[pick]

        selected_idx.append(idx)
        remaining.remove(idx)

    return [candidates[i] for i in selected_idx]


class ChromaStore:
    """
    Persistent ChromaDB wrapper with upsert, search, and MMR.

    Usage
    -----
    store = ChromaStore()
    store.upsert_chunks(chunks, embedder)
    results = store.search("What is the overdue invoice amount?", embedder)
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": DISTANCE_FN},
        )
        logger.info(
            f"ChromaDB ready — collection '[bold]{COLLECTION_NAME}[/bold]' "
            f"({self._collection.count()} existing chunks)"
        )

    # ── Ingestion ───────────────────────────────────────────────────────────────

    def upsert_chunks(self, chunks: list[Chunk], embedder: Embedder) -> None:
        """Embed and upsert chunks (idempotent via chunk ID)."""
        if not chunks:
            logger.warning("upsert_chunks called with empty list.")
            return

        texts      = [c.embedding_text for c in chunks]
        embeddings = embedder.embed_texts(texts)

        self._collection.upsert(
            ids        = [c.id for c in chunks],
            embeddings = embeddings.tolist(),
            documents  = [c.text for c in chunks],
            metadatas  = [
                {k: (str(v) if isinstance(v, bool) else v)
                 for k, v in c.metadata.items()}
                for c in chunks
            ],
        )
        logger.info(f"Upserted {len(chunks)} chunks into ChromaDB.")

    # ── Retrieval ───────────────────────────────────────────────────────────────

    def search(
        self,
        query:          str,
        embedder:       Embedder,
        top_k:          int = TOP_K,
        rerank_top_k:   int = RERANK_TOP_K,
        where:          dict | None = None,   # ChromaDB metadata filter
        use_mmr:        bool = True,
        lambda_mult:    float = MMR_LAMBDA,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the most relevant chunks for a query.

        Returns a list of dicts (sorted by relevance) with keys:
            id, document, metadata, distance, relevance_score
        """
        query_vec = embedder.embed_query(query)

        kwargs: dict[str, Any] = dict(
            query_embeddings=[query_vec.tolist()],
            n_results=min(top_k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)

        # Flatten results into a list of dicts
        candidates = []
        for i, doc_id in enumerate(raw["ids"][0]):
            candidates.append(
                {
                    "id":        doc_id,
                    "document":  raw["documents"][0][i],
                    "metadata":  raw["metadatas"][0][i],
                    "distance":  raw["distances"][0][i],
                    "embedding": np.array(raw["embeddings"][0][i], dtype=np.float32),
                    "relevance_score": 1.0 - raw["distances"][0][i],
                }
            )

        logger.info(f"Raw retrieval: {len(candidates)} candidates for query '{query[:60]}…'")

        if use_mmr and len(candidates) > rerank_top_k:
            candidates = _mmr(query_vec, candidates, rerank_top_k, lambda_mult)
            logger.info(f"After MMR reranking: {len(candidates)} chunks selected.")

        # Strip numpy embedding from returned dicts (not JSON-serialisable)
        for c in candidates:
            c.pop("embedding", None)

        return candidates

    # ── Utilities ───────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        count = self._collection.count()
        return {"collection": COLLECTION_NAME, "total_chunks": count}

    def delete_source(self, source_filename: str) -> None:
        """Remove all chunks belonging to a specific source document."""
        self._collection.delete(where={"source": source_filename})
        logger.info(f"Deleted all chunks for source: {source_filename}")

    def reset(self) -> None:
        """Danger: wipe the entire collection."""
        self._client.delete_collection(COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": DISTANCE_FN},
        )
        logger.warning("ChromaDB collection reset.")
