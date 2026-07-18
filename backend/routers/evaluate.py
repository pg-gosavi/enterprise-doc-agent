"""
Router: /evaluate
  POST / — run a full QA evaluation over a dataset of cases
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from backend.schemas import (
    EvalAggregate, EvalCaseResult, EvalRequest, EvalResponse,
)
from backend.dependencies import get_pipeline
from evaluation.evaluator import EvalCase as CoreEvalCase, Evaluator
from rag.pipeline import RAGPipeline
from rag.prompt_manager import PromptManager
from utils.logger import get_logger

logger = get_logger("router.evaluate")
router = APIRouter(prefix="/evaluate", tags=["Evaluate"])


@router.post(
    "/",
    response_model=EvalResponse,
    summary="Run QA evaluation on a dataset of ground-truth cases",
)
async def run_evaluation(req: EvalRequest, pipeline: RAGPipeline = Depends(get_pipeline)):
    if not req.cases:
        raise HTTPException(status_code=422, detail="No evaluation cases provided.")

    # Swap prompt version
    try:
        pipeline.prompt_manager = PromptManager(version=req.prompt_version)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    eval_cases = [
        CoreEvalCase(
            question=c.question,
            relevant_chunk_ids=c.relevant_chunk_ids,
            reference_answer=c.reference_answer,
        )
        for c in req.cases
    ]

    evaluator = Evaluator()
    results   = []

    for case in eval_cases:
        try:
            resp   = pipeline.query(
                question=case.question,
                top_k=req.top_k,
                rerank_top_k=req.rerank_top_k,
                use_mmr=req.use_mmr,
            )
            result = evaluator.evaluate(resp, case.relevant_chunk_ids, case.reference_answer)
            results.append(result)
        except Exception as exc:
            logger.error(f"Eval case failed '{case.question[:50]}': {exc}")

    if not results:
        raise HTTPException(status_code=500, detail="All evaluation cases failed.")

    agg          = evaluator.aggregate(results)
    report_path  = evaluator.save_report(results, req.run_name)

    return EvalResponse(
        run_name=req.run_name,
        aggregate=EvalAggregate(**agg),
        details=[
            EvalCaseResult(
                question=r.question,
                precision_at_k=r.precision_at_k,
                recall_at_k=r.recall_at_k,
                mrr=r.mrr,
                rouge_l=r.rouge_l,
                faithfulness=r.faithfulness,
                latency_sec=r.latency_sec,
                answer=r.answer,
                retrieved_ids=r.retrieved_ids,
            )
            for r in results
        ],
        report_path=str(report_path),
    )
