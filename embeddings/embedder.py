"""
Embedder
========
Wraps SentenceTransformer to produce L2-normalised embeddings.

Features
--------
- Singleton model instance (loaded once per process)
- Batched encoding with configurable batch size
- In-process LRU cache for repeated query embeddings (avoids re-encoding)
- Returns numpy float32 arrays — ChromaDB accepts these directly
"""

from __future__ import annotations

import functools
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger("embedder")

_MODEL: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info(f"Loading embedding model: [bold]{EMBEDDING_MODEL}[/bold]")
        _MODEL = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    return _MODEL


class Embedder:
    """
    Thin wrapper around SentenceTransformer.

    Usage
    -----
    embedder = Embedder()
    vecs = embedder.embed_texts(["Invoice total: $1,200", "Due date: 2024-06-30"])
    query_vec = embedder.embed_query("What is the invoice total?")
    """

    def __init__(self, batch_size: int = 64):
        self.batch_size = batch_size
        self._model     = _get_model()

    # ── Public API ──────────────────────────────────────────────────────────────

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """
        Embed a list of texts (e.g. chunk embedding_text fields).
        Returns shape (N, D) float32, L2-normalised.
        """
        if not texts:
            return np.empty((0, self._model.get_sentence_embedding_dimension()), dtype=np.float32)

        logger.info(f"Embedding {len(texts)} texts in batches of {self.batch_size}…")
        vecs = self._model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 200,
            convert_to_numpy=True,
        )
        logger.info(f"Embeddings ready — shape: {vecs.shape}")
        return vecs.astype(np.float32)

    @functools.lru_cache(maxsize=256)
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a single query string with LRU caching.
        Returns shape (D,) float32, L2-normalised.
        """
        vec = self._model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.astype(np.float32)

    @property
    def dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()
