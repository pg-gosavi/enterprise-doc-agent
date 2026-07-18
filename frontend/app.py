"""
Streamlit Frontend — Enterprise Document Intelligence Agent
===========================================================
All data comes from the FastAPI backend via HTTP.
Run:  streamlit run frontend/app.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import requests
import streamlit as st

# ── Ensure project root is importable ─────────────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import API_BASE_URL

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="DocIntel Agent",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Layout ── */
#MainMenu, header, footer { visibility: hidden; }
[data-testid="stSidebar"] { background: #0d0f1a; border-right: 1px solid #1e2235; }
section.main { background: #10121e; }

/* ── Cards ── */
.card {
    background: #161929;
    border: 1px solid #252d4a;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
}
.answer-card {
    background: #161929;
    border: 1px solid #252d4a;
    border-left: 4px solid #6366f1;
    border-radius: 0 14px 14px 0;
    padding: 18px 22px;
    color: #dde1f0;
    line-height: 1.75;
    font-size: 15px;
}
.user-bubble {
    background: #1e1b4b;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px;
    color: #c7d2fe;
    font-size: 15px;
    margin-left: auto;
    width: fit-content;
    max-width: 80%;
}

/* ── Badges ── */
.badge {
    display: inline-block;
    border-radius: 20px;
    padding: 3px 11px;
    font-size: 11px;
    font-weight: 600;
    margin: 2px;
}
.badge-groq  { background:#0d2e22; color:#34d399; border:1px solid #065f46; }
.badge-llama { background:#1e1b4b; color:#a5b4fc; border:1px solid #3730a3; }
.badge-ok    { background:#0a2e1a; color:#4ade80; border:1px solid #166534; }
.badge-err   { background:#2e0a0a; color:#f87171; border:1px solid #991b1b; }

/* ── Source pills ── */
.pill {
    display:inline-block; background:#1a2040; border:1px solid #2d3a60;
    border-radius:20px; padding:3px 12px; font-size:12px;
    color:#a5b4fc; margin:3px 3px;
}

/* ── Relevance bar ── */
.rel-wrap { background:#1e2235; border-radius:6px; height:6px; margin:4px 0 10px; overflow:hidden; }
.rel-fill { background:linear-gradient(90deg,#6366f1,#a78bfa); height:100%; border-radius:6px; }

/* ── Metric box ── */
.mbox { background:#161929; border:1px solid #252d4a; border-radius:12px;
        padding:14px 12px; text-align:center; }
.mval { font-size:22px; font-weight:700; color:#a5b4fc; }
.mlbl { font-size:11px; color:#64748b; margin-top:3px; }

/* ── Sidebar items ── */
.sfile { background:#161929; border:1px solid #1e2a40; border-radius:8px;
         padding:8px 12px; margin:5px 0; font-size:13px; color:#94a3b8; }

/* ── Chunk text ── */
.chunk-text { font-size:13px; color:#8892b0; font-family:monospace;
              background:#0d0f1a; border-radius:8px; padding:10px 14px;
              white-space:pre-wrap; word-break:break-word; }

/* ── Quick chips ── */
.stButton > button {
    background:#161929 !important;
    border:1px solid #2d3a60 !important;
    color:#a5b4fc !important;
    border-radius:20px !important;
    font-size:13px !important;
}
.stButton > button:hover {
    background:#1e2235 !important;
    border-color:#6366f1 !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS — API CALLS
# ══════════════════════════════════════════════════════════════════════════════

def _api(method: str, path: str, **kwargs):
    """Thin wrapper around requests that surfaces errors cleanly."""
    url = f"{API_BASE_URL}{path}"
    try:
        r = requests.request(method, url, timeout=120, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Cannot reach the backend. "
            "Start it with: `uvicorn backend.api:app --port 8000 --reload`"
        )
        return None
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", "")
        except Exception:
            pass
        st.error(f"❌ API error {e.response.status_code}: {detail or str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        return None


def api_health():
    return _api("GET", "/health")

def api_list_docs():
    return _api("GET", "/documents/")

def api_index_pdf(file_bytes: bytes, filename: str):
    return _api("POST", "/documents/index",
                files={"file": (filename, file_bytes, "application/pdf")})

def api_delete_doc(filename: str):
    return _api("DELETE", f"/documents/{filename}")

def api_query(payload: dict):
    return _api("POST", "/query/", json=payload)

def api_evaluate(payload: dict):
    return _api("POST", "/evaluate/", json=payload)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
defaults = {
    "chat": [],          # [{role, content, chunks, meta}]
    "query_count":  0,
    "total_tokens": 0,
    "total_latency": 0.0,
    "eval_result": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🔍 DocIntel Agent")
    st.markdown(
        '<span class="badge badge-groq">⚡ Groq</span>'
        '<span class="badge badge-llama">🦙 Llama 3.1 8B</span>',
        unsafe_allow_html=True,
    )

    # ── Backend status ────────────────────────────────────────────────────────
    st.divider()
    health = api_health()
    if health:
        st.markdown(
            f'<span class="badge badge-ok">● Backend online</span> '
            f'<span style="font-size:12px;color:#64748b">{health["total_chunks"]} chunks indexed</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<span class="badge badge-err">● Backend offline</span>', unsafe_allow_html=True)

    # ── Settings ──────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### ⚙️ Settings")
    prompt_label = st.selectbox(
        "Prompt strategy",
        ["v2 — Structured + Citations", "v1 — Minimal", "v3 — Chain of Thought"],
    )
    prompt_version = prompt_label.split(" ")[0]

    col1, col2 = st.columns(2)
    top_k      = col1.number_input("Retrieve k", 1, 20, 8)
    rerank_k   = col2.number_input("Rerank k",   1, 10, 4)
    use_mmr    = st.toggle("MMR Reranking", value=True,
                           help="Maximise relevance + diversity in retrieved chunks")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📄 Upload PDFs")
    uploads = st.file_uploader(
        "Drop PDFs here", type=["pdf"],
        accept_multiple_files=True, label_visibility="collapsed",
    )
    if uploads and st.button("📥 Index", use_container_width=True, type="primary"):
        prog = st.progress(0)
        for i, f in enumerate(uploads):
            with st.spinner(f"Indexing {f.name}…"):
                result = api_index_pdf(f.read(), f.name)
                if result:
                    st.success(f"✅ {f.name} — {result['chunks']} chunks")
            prog.progress((i + 1) / len(uploads))
        st.rerun()

    # ── Indexed documents ─────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📚 Knowledge Base")
    docs_resp = api_list_docs()
    if docs_resp and docs_resp["documents"]:
        for doc in docs_resp["documents"]:
            cols = st.columns([4, 1])
            cols[0].markdown(
                f'<div class="sfile">📄 {doc["filename"]}<br>'
                f'<span style="color:#6366f1;font-size:11px">{doc["chunks"]} chunks</span></div>',
                unsafe_allow_html=True,
            )
            if cols[1].button("🗑", key=f"del_{doc['filename']}", help="Delete"):
                api_delete_doc(doc["filename"])
                st.rerun()
    else:
        st.caption("No documents indexed yet.")

    # ── Session stats ─────────────────────────────────────────────────────────
    st.divider()
    st.markdown("### 📊 Session")
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="mbox"><div class="mval">{st.session_state.query_count}</div>'
        f'<div class="mlbl">Queries</div></div>', unsafe_allow_html=True,
    )
    if st.session_state.query_count:
        avg_lat = st.session_state.total_latency / st.session_state.query_count
        c2.markdown(
            f'<div class="mbox"><div class="mval">{avg_lat:.1f}s</div>'
            f'<div class="mlbl">Avg latency</div></div>', unsafe_allow_html=True,
        )
    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_chat, tab_eval, tab_docs, tab_about = st.tabs(
    ["💬 Chat", "📈 Evaluate", "📋 API Docs", "ℹ️ About"]
)


# ╔══════════════════════════════════════════════════════════════════════════════
# ║ TAB 1 — CHAT
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.markdown("### 💬 Ask your documents anything")

    has_docs = docs_resp and docs_resp.get("documents")
    if not has_docs:
        st.info("📄 Upload and index at least one PDF to start asking questions.")

    # ── Render history ────────────────────────────────────────────────────────
    for msg in st.session_state.chat:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">🙋 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="answer-card">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Source pills
            if msg.get("chunks"):
                pills = "".join(
                    f'<span class="pill">📄 {c["metadata"].get("source","?")} '
                    f'p.{c["metadata"].get("page_num","?")}</span>'
                    for c in msg["chunks"]
                )
                st.markdown(pills, unsafe_allow_html=True)

                with st.expander("🔍 Retrieved chunks", expanded=False):
                    for i, c in enumerate(msg["chunks"]):
                        rel  = c.get("relevance_score", 0)
                        src  = c["metadata"].get("source", "?")
                        page = c["metadata"].get("page_num", "?")
                        text = c.get("document", "")[:500]
                        st.markdown(
                            f"**[{i+1}]** `{src}` — Page {page} &nbsp; "
                            f'<span style="color:#a5b4fc">rel: {rel:.3f}</span>',
                            unsafe_allow_html=True,
                        )
                        pct = int(rel * 100)
                        st.markdown(
                            f'<div class="rel-wrap">'
                            f'<div class="rel-fill" style="width:{pct}%"></div></div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(f'<div class="chunk-text">{text}</div>',
                                    unsafe_allow_html=True)
                        st.divider()

            if msg.get("meta"):
                m = msg["meta"]
                st.caption(
                    f"⏱ {m['latency']:.2f}s · 🔢 {m['tokens']} tokens · "
                    f"📋 prompt {m['pv']} · 🦙 {m['model']}"
                )

    # ── Quick chips ───────────────────────────────────────────────────────────
    if not st.session_state.chat and has_docs:
        st.markdown("**Quick questions:**")
        chips = [
            "What is the total amount due?",
            "List all vendor names.",
            "Summarise payment terms.",
            "What line items are on the invoice?",
        ]
        cols = st.columns(len(chips))
        for col, chip in zip(cols, chips):
            if col.button(chip, use_container_width=True):
                st.session_state["_chip"] = chip
                st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask a question about your documents…", disabled=not has_docs)
    if "_chip" in st.session_state:
        user_input = st.session_state.pop("_chip")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        with st.spinner("🦙 Llama is thinking…"):
            result = api_query({
                "question":       user_input,
                "top_k":          top_k,
                "rerank_top_k":   rerank_k,
                "use_mmr":        use_mmr,
                "prompt_version": prompt_version,
            })

        if result:
            st.session_state.chat.append({
                "role":    "assistant",
                "content": result["answer"],
                "chunks":  result["retrieved_chunks"],
                "meta": {
                    "latency": result["latency_sec"],
                    "tokens":  result["tokens_used"],
                    "pv":      result["prompt_version"],
                    "model":   result["model"],
                },
            })
            st.session_state.query_count  += 1
            st.session_state.total_tokens += result["tokens_used"]
            st.session_state.total_latency+= result["latency_sec"]

        st.rerun()


# ╔══════════════════════════════════════════════════════════════════════════════
# ║ TAB 2 — EVALUATE
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_eval:
    st.markdown("### 📈 QA Evaluation")
    st.markdown(
        "Paste a JSON array of test cases. Each case needs `question`, "
        "`relevant_chunk_ids`, and optionally `reference_answer`."
    )

    left, right = st.columns([2, 3])

    with left:
        run_name = st.text_input("Run name", "eval_run_01")
        eval_json = st.text_area(
            "Evaluation dataset (JSON)",
            height=280,
            value=json.dumps([
                {
                    "question": "What is the total amount due?",
                    "relevant_chunk_ids": ["demo_chunk_001"],
                    "reference_answer": "The total amount due is $14,480.43."
                },
                {
                    "question": "When is the payment due?",
                    "relevant_chunk_ids": ["demo_chunk_001"],
                    "reference_answer": "Payment is due on 2024-04-14."
                },
            ], indent=2),
        )

        run_btn = st.button("▶ Run Evaluation", type="primary",
                            disabled=not (docs_resp and docs_resp.get("documents")))

    with right:
        if run_btn:
            try:
                cases = json.loads(eval_json)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

            with st.spinner("Running evaluation…"):
                result = api_evaluate({
                    "cases":          cases,
                    "run_name":       run_name,
                    "top_k":          top_k,
                    "rerank_top_k":   rerank_k,
                    "use_mmr":        use_mmr,
                    "prompt_version": prompt_version,
                })

            if result:
                st.session_state.eval_result = result

        if st.session_state.eval_result:
            ev   = st.session_state.eval_result
            agg  = ev["aggregate"]
            dets = ev["details"]

            # ── Metric cards ──────────────────────────────────────────────────
            st.markdown("#### Aggregate Metrics")
            metric_names  = ["Precision@k", "Recall@k", "MRR", "ROUGE-L", "Faithfulness"]
            metric_values = [
                agg["precision_at_k"], agg["recall_at_k"],
                agg["mrr"], agg["rouge_l"], agg["faithfulness"],
            ]
            mcols = st.columns(5)
            for col, name, val in zip(mcols, metric_names, metric_values):
                col.metric(name, f"{val:.3f}")
            st.metric("Avg Latency", f"{agg['avg_latency_sec']:.2f}s")

            # ── Radar chart ───────────────────────────────────────────────────
            fig_radar = go.Figure(go.Scatterpolar(
                r=metric_values + [metric_values[0]],
                theta=metric_names + [metric_names[0]],
                fill="toself",
                fillcolor="rgba(99,102,241,0.2)",
                line=dict(color="#6366f1", width=2),
                name=run_name,
            ))
            fig_radar.update_layout(
                polar=dict(
                    bgcolor="#161929",
                    radialaxis=dict(visible=True, range=[0, 1],
                                    tickfont=dict(color="#64748b", size=10),
                                    gridcolor="#252d4a"),
                    angularaxis=dict(tickfont=dict(color="#a5b4fc", size=12),
                                     gridcolor="#252d4a"),
                ),
                paper_bgcolor="#10121e",
                plot_bgcolor="#10121e",
                showlegend=False,
                margin=dict(l=40, r=40, t=40, b=40),
                height=320,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            # ── Per-case bar chart ────────────────────────────────────────────
            if len(dets) > 1:
                q_labels = [d["question"][:35] + "…" for d in dets]
                fig_bar = go.Figure()
                for metric_key, label, color in [
                    ("precision_at_k", "Precision@k", "#6366f1"),
                    ("rouge_l",        "ROUGE-L",     "#a78bfa"),
                    ("faithfulness",   "Faithfulness","#34d399"),
                ]:
                    fig_bar.add_trace(go.Bar(
                        name=label,
                        x=q_labels,
                        y=[d[metric_key] for d in dets],
                        marker_color=color,
                    ))
                fig_bar.update_layout(
                    barmode="group",
                    paper_bgcolor="#10121e",
                    plot_bgcolor="#161929",
                    font=dict(color="#a5b4fc"),
                    xaxis=dict(tickfont=dict(size=10), gridcolor="#252d4a"),
                    yaxis=dict(range=[0, 1.05], gridcolor="#252d4a"),
                    legend=dict(bgcolor="#161929", bordercolor="#252d4a"),
                    margin=dict(l=10, r=10, t=30, b=10),
                    height=300,
                )
                st.plotly_chart(fig_bar, use_container_width=True)

            # ── Latency scatter ───────────────────────────────────────────────
            fig_lat = px.scatter(
                x=[d["question"][:30] + "…" for d in dets],
                y=[d["latency_sec"] for d in dets],
                size=[d["precision_at_k"] * 30 + 5 for d in dets],
                color=[d["rouge_l"] for d in dets],
                color_continuous_scale="Bluyl",
                labels={"x": "Question", "y": "Latency (s)", "color": "ROUGE-L"},
                title="Latency per question (bubble size = Precision@k)",
            )
            fig_lat.update_layout(
                paper_bgcolor="#10121e",
                plot_bgcolor="#161929",
                font=dict(color="#a5b4fc"),
                xaxis=dict(gridcolor="#252d4a"),
                yaxis=dict(gridcolor="#252d4a"),
                margin=dict(l=10, r=10, t=40, b=10),
                height=280,
                showlegend=False,
            )
            st.plotly_chart(fig_lat, use_container_width=True)

            # ── Per-case table ────────────────────────────────────────────────
            st.markdown("#### Per-case Results")
            st.dataframe(
                [
                    {
                        "Question":    d["question"][:55],
                        "P@k":  f"{d['precision_at_k']:.3f}",
                        "Recall":      f"{d['recall_at_k']:.3f}",
                        "MRR":         f"{d['mrr']:.3f}",
                        "ROUGE-L":     f"{d['rouge_l']:.3f}",
                        "Faith.":      f"{d['faithfulness']:.3f}",
                        "Latency (s)": f"{d['latency_sec']:.2f}",
                    }
                    for d in dets
                ],
                use_container_width=True,
            )

            # ── Download ──────────────────────────────────────────────────────
            st.download_button(
                "⬇ Download JSON Report",
                data=json.dumps(ev, indent=2),
                file_name=f"{ev['run_name']}.json",
                mime="application/json",
            )

        else:
            st.info("Run an evaluation to see results and Plotly charts here.")


# ╔══════════════════════════════════════════════════════════════════════════════
# ║ TAB 3 — API DOCS LINK
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.markdown("### 📋 Interactive API Documentation")
    st.markdown(
        f"The FastAPI backend auto-generates interactive docs at:\n\n"
        f"- **Swagger UI**: [{API_BASE_URL}/docs]({API_BASE_URL}/docs)\n"
        f"- **ReDoc**:      [{API_BASE_URL}/redoc]({API_BASE_URL}/redoc)\n\n"
        "You can test every endpoint directly from your browser."
    )

    st.markdown("#### Endpoint Summary")
    st.markdown("""
| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/health` | Liveness probe + chunk count |
| `POST` | `/documents/index` | Upload & index a PDF |
| `GET`  | `/documents/` | List all indexed documents |
| `DELETE` | `/documents/{filename}` | Remove a document |
| `POST` | `/query/` | RAG query |
| `POST` | `/evaluate/` | Run QA evaluation dataset |
    """)

    st.markdown("#### Example: cURL query")
    st.code(
        'curl -X POST http://localhost:8000/query/ \\\n'
        '  -H "Content-Type: application/json" \\\n'
        '  -d \'{"question": "What is the total due?", "top_k": 8, '
        '"rerank_top_k": 4, "use_mmr": true, "prompt_version": "v2"}\'',
        language="bash",
    )

    st.markdown("#### Example: Upload PDF")
    st.code(
        'curl -X POST http://localhost:8000/documents/index \\\n'
        '  -F "file=@invoice.pdf"',
        language="bash",
    )


# ╔══════════════════════════════════════════════════════════════════════════════
# ║ TAB 4 — ABOUT
# ╚══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
### Enterprise Document Intelligence Agent

Production-grade RAG system with a **FastAPI backend** and **Streamlit frontend**.

---

#### 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Streamlit Frontend  (port 8501)                            │
│  - Upload PDFs → POST /documents/index                      │
│  - Chat UI    → POST /query/                                │
│  - Evaluation → POST /evaluate/                             │
└──────────────────────┬──────────────────────────────────────┘
                       │  HTTP (requests library)
┌──────────────────────▼──────────────────────────────────────┐
│  FastAPI Backend  (port 8000)                               │
│  /documents   /query   /evaluate   /health                  │
│                       │                                     │
│   ChromaDB ◄──────────┤                                     │
│   SentenceTransformers│                                     │
│   Groq / Llama 3.1 8B │                                     │
│   ROUGE-Score Eval    │                                     │
└─────────────────────────────────────────────────────────────┘
```

---

#### 🚀 How to run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
cp .env.example .env   # edit and add GROQ_API_KEY=gsk_...

# 3. Start backend  (terminal 1)
uvicorn backend.api:app --host 0.0.0.0 --port 8000 --reload

# 4. Start frontend (terminal 2)
streamlit run frontend/app.py

# OR use the launcher script
bash start.sh
```

---

#### ⚡ Groq Rate Limits (llama-3.1-8b-instant)

| Limit | Value |
|---|---|
| Requests / min | 30 |
| Requests / day | 14,400 |
| **Tokens / min** | **6,000** |
| Tokens / day | 500,000 |

The backend automatically warns when estimated context exceeds 80% of the TPM limit.

---

#### 📋 Prompt Versions

| Version | Strategy | Best for |
|---|---|---|
| v1 | Minimal instruction | Prototyping |
| v2 | Structured + citations | Production invoice / AR QA |
| v3 | Chain-of-thought | Multi-hop financial questions |
    """)
