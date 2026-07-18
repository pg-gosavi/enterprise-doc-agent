"""
Enterprise Document Intelligence Agent
======================================
CLI entry point. Supports three modes:

  python main.py index   --path data/            Index all PDFs in a directory
  python main.py query   --q "What is the total amount due?"
  python main.py eval    --dataset data/eval.json

Environment
-----------
  ANTHROPIC_API_KEY  — required for query / eval modes
  Copy .env.example to .env and fill in your key.

Quick demo (no real PDFs needed)
---------------------------------
  python main.py demo
"""

import argparse
import json
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from config import ACTIVE_PROMPT_VERSION, EVAL_DIR
from evaluation.evaluator import EvalCase, Evaluator
from rag.pipeline import RAGPipeline
from rag.prompt_manager import PromptManager
from utils.logger import get_logger

console = Console()
logger  = get_logger("main")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _print_response(resp) -> None:
    console.print(str(resp))


def _print_aggregate(agg: dict) -> None:
    table = Table(title="Evaluation Results", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    for k, v in agg.items():
        table.add_row(str(k), str(v))
    console.print(table)


# ── Commands ───────────────────────────────────────────────────────────────────

def cmd_index(args) -> None:
    pipeline = RAGPipeline(prompt_version=args.prompt)
    path     = Path(args.path)

    if path.is_dir():
        results = pipeline.index_directory(path)
        total   = sum(v for v in results.values() if v > 0)
        console.print(f"\n✅  Indexed [bold]{total}[/bold] chunks from {len(results)} PDFs.")
    elif path.is_file() and path.suffix.lower() == ".pdf":
        count = pipeline.index_document(path)
        console.print(f"\n✅  Indexed [bold]{count}[/bold] chunks from {path.name}")
    else:
        console.print(f"[red]Path not found or not a PDF/directory: {path}[/red]")
        sys.exit(1)


def cmd_query(args) -> None:
    pipeline = RAGPipeline(prompt_version=args.prompt)
    resp     = pipeline.query(args.q)
    _print_response(resp)


def cmd_eval(args) -> None:
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        console.print(f"[red]Dataset not found: {dataset_path}[/red]")
        sys.exit(1)

    raw_cases = json.loads(dataset_path.read_text())
    eval_cases = [
        EvalCase(
            question           = c["question"],
            relevant_chunk_ids = c.get("relevant_chunk_ids", []),
            reference_answer   = c.get("reference_answer", ""),
        )
        for c in raw_cases
    ]

    pipeline  = RAGPipeline(prompt_version=args.prompt)
    evaluator = Evaluator()
    results   = evaluator.evaluate_dataset(eval_cases, pipeline)
    agg       = evaluator.aggregate(results)
    report    = evaluator.save_report(results, args.run_name)

    _print_aggregate(agg)
    console.print(f"\nFull report → [bold]{report}[/bold]")


def cmd_compare(args) -> None:
    evaluator = Evaluator()
    diff = evaluator.compare_runs(args.run_a, args.run_b)

    table = Table(
        title=f"Run comparison: {args.run_a} vs {args.run_b}",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric")
    table.add_column(args.run_a, style="yellow")
    table.add_column(args.run_b, style="green")
    table.add_column("Δ",        style="bold magenta")
    for metric, vals in diff.items():
        delta = vals["delta"]
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        table.add_row(metric, str(vals[args.run_a]), str(vals[args.run_b]), delta_str)
    console.print(table)


def cmd_demo(args) -> None:
    """
    End-to-end demo using a tiny synthetic invoice text (no real PDF needed).
    Creates an in-memory chunk, upserts it, then runs a query.
    """
    console.rule("[bold green]Enterprise RAG Agent — Demo Mode[/bold green]")

    from ingestion.chunker import Chunk
    from embeddings.embedder import Embedder
    from vector_store.chroma_store import ChromaStore

    embedder = Embedder()
    store    = ChromaStore()

    # Synthetic invoice document
    synthetic_text = (
        "INVOICE #INV-2024-0042\n"
        "Vendor: Acme Supplies Ltd.\n"
        "Bill To: GlobalTech Corp, 100 Main St, New York, NY 10001\n"
        "Invoice Date: 2024-03-15   Due Date: 2024-04-14   Terms: Net 30\n\n"
        "Line Items:\n"
        "  1. Cloud Hosting Services (Q1 2024) ...... $4,800.00\n"
        "  2. Professional Services (40 hrs @ $150) .. $6,000.00\n"
        "  3. Software Licences (5 seats) ............ $2,500.00\n"
        "  Subtotal: $13,300.00\n"
        "  Tax (8.875% NY): $1,180.43\n"
        "  TOTAL DUE: $14,480.43\n\n"
        "Payment: Wire transfer to Bank of America, ABA: 026009593, "
        "Account: 483920145.  Late fee 1.5%/month after due date."
    )

    # Build a single synthetic chunk
    chunk = Chunk(
        id             = "demo_chunk_001",
        text           = "[Source: demo_invoice.pdf | Page: 1]\n" + synthetic_text,
        embedding_text = synthetic_text,
        source         = "demo_invoice.pdf",
        page_num       = 1,
        chunk_index    = 0,
        char_start     = 0,
        char_end       = len(synthetic_text),
        metadata       = {
            "source":   "demo_invoice.pdf",
            "page_num": 1,
            "title":    "Demo Invoice",
            "num_pages": 1,
            "author":   "",
            "chunk_index": 0,
            "char_start":  0,
            "char_end":    len(synthetic_text),
            "has_tables":  False,
        },
    )

    store.upsert_chunks([chunk], embedder)
    console.print("✅  Synthetic invoice indexed.\n")

    # Run some demo queries
    pipeline = RAGPipeline()
    demo_questions = [
        "What is the total amount due on this invoice?",
        "When is the payment due date and what are the late fee terms?",
        "What are the individual line items and their costs?",
    ]

    for q in demo_questions:
        resp = pipeline.query(q, top_k=2, rerank_top_k=1)
        _print_response(resp)

    console.rule("[bold green]Demo complete[/bold green]")


# ── Argument parsing ───────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Enterprise Document Intelligence Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # index
    p_index = sub.add_parser("index", help="Index PDF(s) into ChromaDB")
    p_index.add_argument("--path",   required=True, help="PDF file or directory of PDFs")
    p_index.add_argument("--prompt", default=ACTIVE_PROMPT_VERSION)

    # query
    p_query = sub.add_parser("query", help="Run a RAG query")
    p_query.add_argument("--q",      required=True, help="Question to answer")
    p_query.add_argument("--prompt", default=ACTIVE_PROMPT_VERSION)

    # eval
    p_eval = sub.add_parser("eval", help="Evaluate on a JSON dataset")
    p_eval.add_argument("--dataset",  required=True, help="Path to eval JSON file")
    p_eval.add_argument("--run-name", dest="run_name", default="eval_run",
                        help="Name for the saved report")
    p_eval.add_argument("--prompt",   default=ACTIVE_PROMPT_VERSION)

    # compare
    p_cmp = sub.add_parser("compare", help="Compare two evaluation runs")
    p_cmp.add_argument("--run-a", dest="run_a", required=True)
    p_cmp.add_argument("--run-b", dest="run_b", required=True)

    # demo
    sub.add_parser("demo", help="Self-contained demo with a synthetic invoice")

    return parser


def main() -> None:
    parser = build_parser()
    args   = parser.parse_args()

    commands = {
        "index":   cmd_index,
        "query":   cmd_query,
        "eval":    cmd_eval,
        "compare": cmd_compare,
        "demo":    cmd_demo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
