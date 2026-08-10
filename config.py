"""
Central configuration — single source of truth for all parameters.
"""
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_store"
LOG_DIR    = BASE_DIR / "logs"
EVAL_DIR   = BASE_DIR / "evaluation_results"
UPLOAD_DIR = BASE_DIR / "uploads"

for d in (DATA_DIR, CHROMA_DIR, LOG_DIR, EVAL_DIR, UPLOAD_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Embedding model ────────────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 80
MIN_CHUNK_CHARS = 100

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "enterprise_docs"
DISTANCE_FN     = "cosine"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K        = 8
RERANK_TOP_K = 4
MMR_LAMBDA   = 0.6

# ── LLM (Groq / Llama 3.1 8B) ────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
LLM_MODEL       = "llama-3.1-8b-instant"
LLM_MAX_TOKENS  = 1024
LLM_TEMPERATURE = 0.0
GROQ_TPM_LIMIT  = 6_000

# ── Prompt versioning ─────────────────────────────────────────────────────────
ACTIVE_PROMPT_VERSION = "v2"

# ── FastAPI ───────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "localhost")
API_PORT = int(os.getenv("PORT", os.getenv("API_PORT", "8000")))
API_BASE_URL = f"http://{API_HOST}:{API_PORT}"

# ── Evaluation ────────────────────────────────────────────────────────────────
EVAL_METRICS = ["precision_at_k", "mrr", "rouge_l", "faithfulness"]
