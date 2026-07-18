"""
Evaluator
=========
Measures RAG system performance across two dimensions:

Retrieval Quality
-----------------
- Precision@k  : fraction of top-k retrieved chunks that are relevant
- MRR          : Mean Reciprocal Rank — how high the first relevant chunk ranks
- Recall@k     : fraction of all relevant chunks that appear in top-k

Response Quality
----------------
- ROUGE-L      : longest-common-subsequence overlap with reference answer
- Faithfulness : heuristic — does the answer cite the retrieved chunks?
  (In production replace this with an LLM-judge call.)

Usage
-----
    evaluator = Evaluator()
    result = evaluator.evaluate(
        rag_response=resp,
        relevant_chunk_ids=["a1b2c3", "d4e5f6"],   # ground truth
        reference_answer="The total is $1,200.",     # optional
    )
    evaluator.evaluate_dataset(eval_cases, pipeline)
    evaluator.save_report(results, "run_01")
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from rouge_score import rouge_scorer

from config import EVAL_DIR
from rag.pipeline import RAGPipeline, RagResponse
from utils.logger import get_logger

logger = get_logger("evaluator")

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


@dataclass
class EvalCase:
    """A single ground-truth evaluation case."""
    question:            str
    relevant_chunk_ids:  list[str]          # set of chunk IDs that are relevant
    reference_answer:    str = ""           # optional reference answer
    metadata:            dict = field(default_factory=dict)


@dataclass
class EvalResult:
    """Metrics for a single query evaluation."""
    question:         str
    precision_at_k:   float
    recall_at_k:      float
    mrr:              float
    rouge_l:          float
    faithfulness:     float      # 0.0 – 1.0
    latency_sec:      float
    prompt_version:   str
    retrieved_ids:    list[str]
    answer:           str


class Evaluator:
    """Compute retrieval and response quality metrics for RAG responses."""

    # ── Single-response evaluation ──────────────────────────────────────────────

    def evaluate(
        self,
        rag_response:        RagResponse,
        relevant_chunk_ids:  Sequence[str],
        reference_answer:    str = "",
    ) -> EvalResult:
        """Compute all metrics for one RagResponse."""
        retrieved_ids = [c["id"] for c in rag_response.retrieved_chunks]
        relevant_set  = set(relevant_chunk_ids)

        precision = self._precision_at_k(retrieved_ids, relevant_set)
        recall    = self._recall_at_k(retrieved_ids, relevant_set)
        mrr       = self._mrr(retrieved_ids, relevant_set)
        rouge_l   = self._rouge_l(rag_response.answer, reference_answer)
        faith     = self._faithfulness(rag_response)

        result = EvalResult(
            question       = rag_response.query,
            precision_at_k = precision,
            recall_at_k    = recall,
            mrr            = mrr,
            rouge_l        = rouge_l,
            faithfulness   = faith,
            latency_sec    = rag_response.latency_sec,
            prompt_version = rag_response.prompt_version,
            retrieved_ids  = retrieved_ids,
            answer         = rag_response.answer,
        )

        logger.info(
            f"Eval | P@k={precision:.3f} | Recall={recall:.3f} | "
            f"MRR={mrr:.3f} | ROUGE-L={rouge_l:.3f} | Faith={faith:.3f}"
        )
        return result

    # ── Dataset evaluation ──────────────────────────────────────────────────────

    def evaluate_dataset(
        self,
        eval_cases: list[EvalCase],
        pipeline:   RAGPipeline,
    ) -> list[EvalResult]:
        """Run evaluation over a list of EvalCase objects."""
        results: list[EvalResult] = []
        for case in eval_cases:
            logger.info(f"Evaluating: '{case.question[:70]}'")
            resp   = pipeline.query(case.question)
            result = self.evaluate(resp, case.relevant_chunk_ids, case.reference_answer)
            results.append(result)
        return results

    # ── Aggregate report ────────────────────────────────────────────────────────

    def aggregate(self, results: list[EvalResult]) -> dict:
        """Compute mean metrics across all EvalResult objects."""
        if not results:
            return {}

        def mean(attr: str) -> float:
            vals = [getattr(r, attr) for r in results]
            return sum(vals) / len(vals)

        return {
            "n":              len(results),
            "precision_at_k": round(mean("precision_at_k"), 4),
            "recall_at_k":    round(mean("recall_at_k"), 4),
            "mrr":            round(mean("mrr"), 4),
            "rouge_l":        round(mean("rouge_l"), 4),
            "faithfulness":   round(mean("faithfulness"), 4),
            "avg_latency_sec":round(mean("latency_sec"), 3),
            "prompt_version": results[0].prompt_version if results else "N/A",
        }

    def save_report(self, results: list[EvalResult], run_name: str) -> Path:
        """Persist individual results + aggregate to a JSON file."""
        report = {
            "run_name":  run_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "aggregate": self.aggregate(results),
            "details": [
                {
                    "question":       r.question,
                    "precision_at_k": r.precision_at_k,
                    "recall_at_k":    r.recall_at_k,
                    "mrr":            r.mrr,
                    "rouge_l":        r.rouge_l,
                    "faithfulness":   r.faithfulness,
                    "latency_sec":    r.latency_sec,
                    "prompt_version": r.prompt_version,
                    "retrieved_ids":  r.retrieved_ids,
                    "answer":         r.answer,
                }
                for r in results
            ],
        }
        out = EVAL_DIR / f"{run_name}.json"
        out.write_text(json.dumps(report, indent=2))
        logger.info(f"Evaluation report saved → {out}")
        return out

    def compare_runs(self, run_a: str, run_b: str) -> dict:
        """Load two saved reports and diff their aggregate metrics."""
        a = json.loads((EVAL_DIR / f"{run_a}.json").read_text())["aggregate"]
        b = json.loads((EVAL_DIR / f"{run_b}.json").read_text())["aggregate"]
        metrics = ["precision_at_k", "recall_at_k", "mrr", "rouge_l", "faithfulness"]
        return {
            m: {
                run_a: a.get(m),
                run_b: b.get(m),
                "delta": round(b.get(m, 0) - a.get(m, 0), 4),
            }
            for m in metrics
        }

    # ── Metric implementations ──────────────────────────────────────────────────

    @staticmethod
    def _precision_at_k(retrieved: list[str], relevant: set[str]) -> float:
        if not retrieved:
            return 0.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(retrieved)

    @staticmethod
    def _recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
        if not relevant:
            return 1.0
        hits = sum(1 for r in retrieved if r in relevant)
        return hits / len(relevant)

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _rouge_l(generated: str, reference: str) -> float:
        if not reference.strip():
            return 0.0
        scores = _ROUGE.score(reference, generated)
        return round(scores["rougeL"].fmeasure, 4)

    @staticmethod
    def _faithfulness(resp: RagResponse) -> float:
        """
        Heuristic faithfulness: what fraction of retrieved chunk sources are
        cited in the answer?

        In production, replace with an LLM judge:
            "Given this context and answer, is every claim in the answer
             supported by the context? Score 0–1."
        """
        if not resp.retrieved_chunks:
            return 0.0

        answer_lower = resp.answer.lower()
        cited = 0
        for chunk in resp.retrieved_chunks:
            source = chunk["metadata"].get("source", "")
            if source and source.lower() in answer_lower:
                cited += 1
            # Also accept [Source: ...] citation pattern
            elif "[source:" in answer_lower and source.lower()[:8] in answer_lower:
                cited += 1

        return cited / len(resp.retrieved_chunks)
