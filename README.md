# Enterprise Document Intelligence Agent

Production RAG system with a **FastAPI backend** + **Streamlit frontend**.

## Project Structure

```
enterprise_doc_agent/
├── backend/
│   ├── api.py               ← FastAPI app (entry point)
│   ├── dependencies.py      ← Singleton DI (pipeline, store)
│   ├── schemas.py           ← All Pydantic request/response models
│   └── routers/
│       ├── documents.py     ← POST /documents/index, GET, DELETE
│       ├── query.py         ← POST /query/
│       └── evaluate.py      ← POST /evaluate/
├── frontend/
│   └── app.py               ← Streamlit UI (calls FastAPI via HTTP)
├── ingestion/
│   ├── pdf_extractor.py     ← PyMuPDF, multi-column, metadata
│   └── chunker.py           ← Sentence-aware overlapping chunks
├── embeddings/
│   └── embedder.py          ← SentenceTransformer + LRU cache
├── vector_store/
│   └── chroma_store.py      ← ChromaDB upsert, search, MMR
├── rag/
│   ├── pipeline.py          ← End-to-end RAG orchestrator (Groq)
│   └── prompt_manager.py    ← Versioned prompt templates v1/v2/v3
├── evaluation/
│   └── evaluator.py         ← P@k, MRR, ROUGE-L, faithfulness
├── config.py                ← All parameters (single source of truth)
├── requirements.txt
├── start.sh                 ← Launch both servers in parallel
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add: GROQ_API_KEY=gsk_...
bash start.sh               # starts backend :8000 + frontend :8501
```

Or run them separately:

```bash
# Terminal 1 — backend
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 — frontend
streamlit run frontend/app.py
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs  | Swagger UI — test every endpoint |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:8501       | Streamlit UI |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness + total chunk count |
| `POST` | `/documents/index` | Upload & index a PDF |
| `GET` | `/documents/` | List all indexed documents |
| `DELETE` | `/documents/{filename}` | Remove a document |
| `POST` | `/query/` | RAG query → answer + chunks |
| `POST` | `/evaluate/` | QA evaluation dataset |

## Groq Rate Limits (llama-3.1-8b-instant)

- 30 req/min · 14.4K req/day
- **6,000 tokens/min** · 500K tokens/day
