"""
Semantic Chunker
================
Converts extracted page content into overlapping chunks optimised for RAG:

Strategy
--------
1. Split text into sentences using a regex (avoids NLTK dependency).
2. Accumulate sentences into a window until CHUNK_SIZE characters are reached.
3. Slide the window forward by (CHUNK_SIZE - CHUNK_OVERLAP) characters,
   retaining the overlap tail as context for the next chunk.
4. Prepend a context header (source + page) to every chunk so the LLM can
   cite the origin without the embedding needing to encode it.
5. Discard chunks shorter than MIN_CHUNK_CHARS.

Result
------
Each Chunk carries: id, text, embedding_text, source, page_num, chunk_index,
char_start, char_end, metadata.
"""

from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_CHARS
from ingestion.pdf_extractor import PageContent
from utils.logger import get_logger

logger = get_logger("chunker")

# Sentence boundary: end with . ! ? followed by space or newline
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    """A single text chunk ready for embedding."""
    id:            str       # deterministic SHA-256 prefix
    text:          str       # full chunk text (with context header)
    embedding_text: str      # text sent to the embedding model (no header)
    source:        str
    page_num:      int
    chunk_index:   int
    char_start:    int
    char_end:      int
    metadata:      dict = field(default_factory=dict)


def _make_chunk_id(source: str, page: int, index: int) -> str:
    """Deterministic, collision-resistant chunk ID."""
    raw = f"{source}|p{page}|c{index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _build_context_header(source: str, page: int) -> str:
    return f"[Source: {source} | Page: {page}]\n"


class SemanticChunker:
    """
    Produces overlapping sentence-aware chunks from a list of PageContent objects.

    Parameters
    ----------
    chunk_size   : target chunk size in characters (~4 chars/token)
    chunk_overlap: character overlap between successive chunks
    min_chars    : discard chunks smaller than this
    """

    def __init__(
        self,
        chunk_size:    int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        min_chars:     int = MIN_CHUNK_CHARS,
    ):
        self.chunk_size    = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chars     = min_chars

    # ── Public API ──────────────────────────────────────────────────────────────

    def chunk_pages(self, pages: list[PageContent]) -> list[Chunk]:
        """Convert a list of PageContent objects into Chunk objects."""
        all_chunks: list[Chunk] = []
        for page in pages:
            if not page.text.strip():
                continue
            page_chunks = self._chunk_page(page)
            all_chunks.extend(page_chunks)
        logger.info(
            f"Chunking complete — {len(pages)} pages → {len(all_chunks)} chunks"
        )
        return all_chunks

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _chunk_page(self, page: PageContent) -> list[Chunk]:
        sentences = _split_sentences(page.text)
        if not sentences:
            return []

        chunks: list[Chunk]  = []
        buffer: list[str]    = []
        buf_len              = 0
        chunk_index          = 0
        char_cursor          = 0

        for sentence in sentences:
            sen_len = len(sentence) + 1  # +1 for the space/newline

            if buf_len + sen_len > self.chunk_size and buffer:
                chunk = self._finalise_chunk(
                    buffer, page, chunk_index, char_cursor
                )
                if chunk:
                    chunks.append(chunk)
                    chunk_index += 1

                # Compute overlap: keep tail of buffer that fits in overlap window
                overlap_buf: list[str] = []
                overlap_len = 0
                for s in reversed(buffer):
                    if overlap_len + len(s) + 1 > self.chunk_overlap:
                        break
                    overlap_buf.insert(0, s)
                    overlap_len += len(s) + 1

                char_cursor += buf_len - overlap_len
                buffer  = overlap_buf
                buf_len = overlap_len

            buffer.append(sentence)
            buf_len += sen_len

        # Flush remaining sentences
        if buffer:
            chunk = self._finalise_chunk(buffer, page, chunk_index, char_cursor)
            if chunk:
                chunks.append(chunk)

        return chunks

    def _finalise_chunk(
        self,
        buffer: list[str],
        page:   PageContent,
        index:  int,
        char_start: int,
    ) -> Chunk | None:
        embedding_text = " ".join(buffer).strip()

        if len(embedding_text) < self.min_chars:
            return None

        header    = _build_context_header(page.source, page.page_num)
        full_text = header + embedding_text
        char_end  = char_start + len(embedding_text)

        return Chunk(
            id             = _make_chunk_id(page.source, page.page_num, index),
            text           = full_text,
            embedding_text = embedding_text,
            source         = page.source,
            page_num       = page.page_num,
            chunk_index    = index,
            char_start     = char_start,
            char_end       = char_end,
            metadata       = {
                **page.metadata,
                "chunk_index": index,
                "char_start":  char_start,
                "char_end":    char_end,
                "has_tables":  page.has_tables,
            },
        )
