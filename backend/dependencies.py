"""
FastAPI dependency injection — shared singleton instances.
The pipeline and store are created once and reused across all requests.
"""
from __future__ import annotations
from functools import lru_cache

from rag.pipeline import RAGPipeline
from vector_store.chroma_store import ChromaStore


@lru_cache(maxsize=1)
def _pipeline_singleton() -> RAGPipeline:
    return RAGPipeline()


@lru_cache(maxsize=1)
def _store_singleton() -> ChromaStore:
    return ChromaStore()


def get_pipeline() -> RAGPipeline:
    """FastAPI dependency: returns the shared RAGPipeline instance."""
    return _pipeline_singleton()


def get_store() -> ChromaStore:
    """FastAPI dependency: returns the shared ChromaStore instance."""
    return _store_singleton()
