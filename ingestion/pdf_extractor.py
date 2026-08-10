"""
PDF Extractor
=============
Uses PyMuPDF (fitz) to extract text page-by-page with rich metadata.
Handles:
  - Multi-column layouts via text-block sorting
  - Table detection heuristic (lines of short words)
  - Header / footer stripping by vertical position threshold
  - OCR fallback flag (True if page has almost no text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import fitz  # PyMuPDF

from utils.logger import get_logger

logger = get_logger("pdf_extractor")


@dataclass
class PageContent:
    """Structured output for a single PDF page."""
    page_num:   int
    text:       str
    source:     str                          # PDF filename
    has_tables: bool = False
    ocr_needed: bool = False
    metadata:   dict = field(default_factory=dict)


def _sort_blocks(blocks: list[dict]) -> list[dict]:
    """Sort text blocks top-to-bottom, then left-to-right (handles 2-column PDFs)."""
    return sorted(blocks, key=lambda b: (round(b["bbox"][1] / 10) * 10, b["bbox"][0]))


def _strip_header_footer(blocks: list[dict], page_height: float) -> list[dict]:
    """Remove blocks within the top 7% or bottom 7% of the page (headers/footers)."""
    margin_top    = page_height * 0.07
    margin_bottom = page_height * 0.93
    return [b for b in blocks if margin_top <= b["bbox"][1] <= margin_bottom]


def _detect_tables(text: str) -> bool:
    """Heuristic: flag pages where many short tab/pipe-separated tokens appear."""
    lines = text.splitlines()
    table_lines = sum(1 for line in lines if len(line.split()) > 3 and "\t" in line)
    return table_lines > 3


def _clean_text(raw: str) -> str:
    """Normalise whitespace; collapse hyphenated line-breaks; strip ligatures."""
    text = re.sub(r"-\n", "", raw)           # re-join hyphenated words
    text = re.sub(r"\n{3,}", "\n\n", text)   # collapse blank lines
    text = re.sub(r"[ \t]+", " ", text)      # collapse spaces
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")  # common ligatures
    return text.strip()


class PDFExtractor:
    """Extract structured page content from a PDF file."""

    MIN_CHARS_PER_PAGE = 80   # below this → likely scanned / image-only page

    def __init__(self, filepath: str | Path, source_name: str | None = None):
        self.filepath = Path(filepath)
        self.source_name = source_name or self.filepath.name
        if not self.filepath.exists():
            raise FileNotFoundError(f"PDF not found: {self.filepath}")
        logger.info(f"Loaded PDF: [bold]{self.filepath.name}[/bold]")

    # ── Public API ──────────────────────────────────────────────────────────────

    @property
    def metadata(self) -> dict:
        """Document-level metadata (title, author, creation date …)."""
        with fitz.open(self.filepath) as doc:
            return doc.metadata

    def extract_pages(self) -> list[PageContent]:
        """Return a list of PageContent objects, one per page."""
        pages = list(self._iter_pages())
        logger.info(f"Extracted {len(pages)} pages from {self.filepath.name}")
        return pages

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _iter_pages(self) -> Iterator[PageContent]:
        doc_meta = {}
        with fitz.open(self.filepath) as doc:
            doc_meta = {
                "title":    doc.metadata.get("title", ""),
                "author":   doc.metadata.get("author", ""),
                "num_pages": doc.page_count,
                "source":   self.source_name,
            }

            for page in doc:
                page_height = page.rect.height

                # Extract text blocks with coordinates
                blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
                text_blocks = [b for b in blocks if b["type"] == 0]  # type 0 = text

                text_blocks = _strip_header_footer(text_blocks, page_height)
                text_blocks = _sort_blocks(text_blocks)

                # Concatenate spans into a single string per page
                raw_text = "\n".join(
                    " ".join(span["text"] for line in blk["lines"] for span in line["spans"])
                    for blk in text_blocks
                )

                cleaned = _clean_text(raw_text)
                ocr_needed = len(cleaned) < self.MIN_CHARS_PER_PAGE

                yield PageContent(
                    page_num=page.number + 1,
                    text=cleaned,
                    source=self.source_name,
                    has_tables=_detect_tables(cleaned),
                    ocr_needed=ocr_needed,
                    metadata={**doc_meta, "page_num": page.number + 1},
                )

                if ocr_needed:
                    logger.warning(
                        f"Page {page.number+1} of {self.filepath.name} "
                        "has very little text — may be a scanned image."
                    )
