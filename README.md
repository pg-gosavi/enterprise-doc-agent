# Enterprise Document Intelligence Agent

Production RAG system with a **FastAPI backend** and two frontends:
- **Next.js UI** in `app/`
- **Streamlit UI** in `frontend/`

## Project Structure

```
enterprise_doc_agent/
├── app/                    ← Next.js UI (client-side document assistant)
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
├── package.json             ← Next.js frontend dependencies
├── start.sh                 ← Launch both servers in parallel
└── .env.example
```

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env        # add: GROQ_API_KEY=gsk_...
```

Start the backend:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
```

Then start the Next.js frontend in a second terminal:

```bash
npm install
npm run dev
```

Alternatively, launch the Streamlit frontend instead:

```bash
streamlit run frontend/app.py
```

Or use the bundled helper script:

```bash
bash start.sh
```

| URL | Purpose |
|-----|---------|
| http://localhost:8000/docs  | Swagger UI — test every endpoint |
| http://localhost:8000/redoc | ReDoc documentation |
| http://localhost:3000       | Next.js UI |
| http://localhost:8501       | Streamlit UI |

## Demo video

If you want to include a demo video in the README, upload the file manually into the repository and then embed it with a relative path.

1. Add the video file to the repository, for example:
   - `demo.mp4`
   - `docs/demo.mp4`
   - `assets/demo.mp4`
2. Commit the file to GitHub.
3. In this README, embed it using HTML so GitHub can render the video:

```md
<video controls width="800">
  <source src="./demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
```

If you upload the file to a subfolder, update the `src` path accordingly, for example:

```md
<video controls width="800">
  <source src="./assets/demo.mp4" type="video/mp4">
</video>
```

Alternatively, if you host the video on YouTube or another external site, use a linked thumbnail instead:

```md
[![Watch demo](https://img.youtube.com/vi/<VIDEO_ID>/hqdefault.jpg)](https://youtu.be/<VIDEO_ID>)
```

## Deployment

This project is best deployed as two connected services:

- **Frontend:** Vercel for the Next.js app in `app/`.
- **Backend:** A Python host such as Render or Fly.io for the FastAPI service.

### Recommended deployment setup

1. Push the repo to GitHub (`https://github.com/pg-gosavi/enterprise-doc-agent`).
2. On Vercel, create a new project from this GitHub repo. Vercel will detect the Next.js app automatically.
3. On Render, create a new Web Service from the same GitHub repo.
   - Environment: Python 3.x
   - Start command: `uvicorn backend.api:app --host 0.0.0.0 --port $PORT`
   - Set environment variable: `GROQ_API_KEY`
4. In Vercel project settings, add:
   - `NEXT_PUBLIC_API_BASE_URL=https://<your-backend-url>`

### Why this setup?

- Vercel is ideal for the Next.js frontend.
- The FastAPI backend cannot run directly on Vercel as a full Python service, so it needs a dedicated Python host.
- Render/Fly.io supports a persistent web service and can run the FastAPI app with the same repo.

### Local vs production

For local development, keep using:

```bash
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload
npm run dev
```

When deployed, the frontend will call the rendered backend URL via `NEXT_PUBLIC_API_BASE_URL`.

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
