"""
Prompt Manager
==============
A lightweight prompt versioning system.

- Prompts are defined as Python dataclasses (easy to diff in git).
- Each version has a unique slug, description, and template string.
- The active version is set in config.py (ACTIVE_PROMPT_VERSION).
- All versions are logged so evaluation results are always reproducible.

Template variables
------------------
  {context}  — retrieved chunks, formatted with source citations
  {query}    — the user's question
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from config import ACTIVE_PROMPT_VERSION
from utils.logger import get_logger

logger = get_logger("prompt_manager")


@dataclass(frozen=True)
class PromptTemplate:
    version:     str
    description: str
    system:      str
    user_tmpl:   str   # must contain {context} and {query}

    def render(self, context: str, query: str) -> tuple[str, str]:
        """Return (system_prompt, user_message)."""
        return self.system, self.user_tmpl.format(context=context, query=query)


# ── Prompt Registry ────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, PromptTemplate] = {}


def _register(template: PromptTemplate) -> PromptTemplate:
    _REGISTRY[template.version] = template
    return template


# v1 — baseline (minimal instruction)
_register(PromptTemplate(
    version="v1",
    description="Baseline: answer from context only, short instruction.",
    system=(
        "You are a document intelligence assistant. "
        "Answer ONLY from the provided context. "
        "If the answer is not in the context, say 'I could not find that information.'"
    ),
    user_tmpl=(
        "Context:\n{context}\n\n"
        "Question: {query}\n\n"
        "Answer:"
    ),
))

# v2 — structured citation prompt (current active)
_register(PromptTemplate(
    version="v2",
    description=(
        "Structured output with citations, confidence signal, "
        "and explicit 'not found' handling."
    ),
    system=(
        "You are an enterprise document intelligence assistant specialised in "
        "financial documents such as invoices, purchase orders, and AR reports.\n\n"
        "Rules:\n"
        "1. Answer ONLY from the provided context passages. Do NOT use prior knowledge.\n"
        "2. Cite the source of every claim using [Source: <filename>, Page: <N>].\n"
        "3. If the answer requires calculations, show the calculation steps.\n"
        "4. If the context does not contain enough information, respond with:\n"
        "   'INSUFFICIENT CONTEXT: <what is missing>'\n"
        "5. Be concise — one clear paragraph unless the question requires structured output.\n"
        "6. For monetary values always include the currency symbol as found in the source."
    ),
    user_tmpl=(
        "### Context Passages\n"
        "{context}\n\n"
        "### Question\n"
        "{query}\n\n"
        "### Answer (with citations)"
    ),
))

# v3 — chain-of-thought variant (for complex multi-step financial questions)
_register(PromptTemplate(
    version="v3",
    description="Chain-of-thought reasoning before final answer; best for multi-hop questions.",
    system=(
        "You are an enterprise document intelligence assistant specialised in "
        "financial documents.\n\n"
        "For each question:\n"
        "1. Write <thinking> ... </thinking> with step-by-step reasoning.\n"
        "2. Write <answer> ... </answer> with the final answer and citations.\n\n"
        "Base your reasoning ONLY on the provided context. "
        "If insufficient, state what is missing inside <answer>."
    ),
    user_tmpl=(
        "### Context Passages\n"
        "{context}\n\n"
        "### Question\n"
        "{query}"
    ),
))


# ── Public API ─────────────────────────────────────────────────────────────────

class PromptManager:
    """
    Retrieve and render versioned prompt templates.

    Usage
    -----
    pm = PromptManager()                # uses ACTIVE_PROMPT_VERSION
    pm = PromptManager(version="v3")   # explicit version
    system, user = pm.render(context_str, query_str)
    """

    def __init__(self, version: str | None = None):
        self.version = version or ACTIVE_PROMPT_VERSION
        if self.version not in _REGISTRY:
            raise ValueError(
                f"Unknown prompt version '{self.version}'. "
                f"Available: {list(_REGISTRY.keys())}"
            )
        self._template = _REGISTRY[self.version]
        logger.info(
            f"PromptManager initialised — version [bold]{self.version}[/bold]: "
            f"{self._template.description}"
        )

    def render(self, context: str, query: str) -> tuple[str, str]:
        """Return (system_prompt, user_message) ready for the LLM API."""
        return self._template.render(context, query)

    @staticmethod
    def list_versions() -> list[str]:
        return list(_REGISTRY.keys())

    @staticmethod
    def format_chunks_as_context(chunks: list[dict]) -> str:
        """
        Convert retrieved chunk dicts into a numbered context string.
        Each passage includes its citation header.
        """
        parts = []
        for i, chunk in enumerate(chunks, start=1):
            meta   = chunk.get("metadata", {})
            source = meta.get("source", "unknown")
            page   = meta.get("page_num", "?")
            text   = chunk.get("document", "").strip()
            parts.append(
                f"[{i}] [Source: {source}, Page: {page}]\n{text}"
            )
        return "\n\n---\n\n".join(parts)
