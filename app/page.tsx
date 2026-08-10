"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  Bot, ChevronDown, FileText, FolderOpen, LayoutDashboard, LoaderCircle,
  Menu, PanelLeftClose, Plus, Send, Settings2, ShieldCheck, SlidersHorizontal,
  Sparkles, Trash2, UploadCloud
} from "lucide-react";
import { DocumentScene } from "../components/DocumentScene";
import { api, Citation, DocumentInfo } from "../lib/api";

type ChatMessage = { role: "assistant" | "user"; text: string; citations?: Citation[]; latency?: number };

export default function CommandCenter() {
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [files, setFiles] = useState<File[]>([]);
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(8);
  const [rerankTopK, setRerankTopK] = useState(4);
  const [useMmr, setUseMmr] = useState(true);
  const [isIndexing, setIsIndexing] = useState(false);
  const [isQuerying, setIsQuerying] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [apiStatus, setApiStatus] = useState<"loading" | "online" | "offline">("loading");
  const [error, setError] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "assistant", text: "Connect a source document to begin an evidence-led analysis session." },
  ]);
  const fileInput = useRef<HTMLInputElement>(null);

  const indexedChunks = useMemo(() => documents.reduce((total, document) => total + document.chunks, 0), [documents]);

  const refreshDocuments = async () => {
    const result = await api.documents();
    setDocuments(result);
  };

  useEffect(() => {
    void Promise.all([api.health(), refreshDocuments()])
      .then(([health]) => {
        setApiStatus(health.status === "ok" ? "online" : "offline");
      })
      .catch((requestError: unknown) => {
        setApiStatus("offline");
        setError(requestError instanceof Error ? requestError.message : "The API could not be reached.");
      });
  }, []);

  const selectFiles = (event: ChangeEvent<HTMLInputElement>) => {
    setFiles(Array.from(event.target.files ?? []));
  };

  const indexFiles = async () => {
    if (!files.length) return;
    setError(null);
    setIsIndexing(true);
    try {
      for (const file of files) await api.index(file);
      await refreshDocuments();
      setFiles([]);
      if (fileInput.current) fileInput.current.value = "";
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to index the selected document.");
    } finally {
      setIsIndexing(false);
    }
  };

  const deleteDocument = async (filename: string) => {
    setDeleting(filename);
    setError(null);
    try {
      await api.remove(filename);
      await refreshDocuments();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to remove the document.");
    } finally {
      setDeleting(null);
    }
  };

  const submitQuestion = async (event: FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed || isQuerying) return;
    setError(null);
    setMessages((current) => [...current, { role: "user", text: trimmed }]);
    setQuery("");
    setIsQuerying(true);
    try {
      const result = await api.query(trimmed, topK, rerankTopK, useMmr);
      setMessages((current) => [...current, { role: "assistant", text: result.answer, citations: result.citations, latency: result.latency }]);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "Unable to complete this analysis.");
    } finally {
      setIsQuerying(false);
    }
  };

  return (
    <main className="app-shell">
      <DocumentScene />
      <aside className={`sidebar ${sidebarOpen ? "is-open" : "is-collapsed"}`}>
        <div className="brand-row">
          <div className="brand-mark"><span /><span /><span /></div>
          <div className="brand-name">DocIntel</div>
          <button className="icon-button desktop-only" onClick={() => setSidebarOpen(false)} aria-label="Collapse navigation" title="Collapse navigation"><PanelLeftClose size={17} /></button>
        </div>
        <nav className="navigation" aria-label="Primary navigation">
          <a className="nav-item is-active" href="#workspace"><LayoutDashboard size={18} /><span>Workspace</span></a>
          <a className="nav-item" href="#documents"><FolderOpen size={18} /><span>Documents</span><b>{documents.length}</b></a>
          <a className="nav-item" href="#assistant"><Bot size={18} /><span>Assistant</span></a>
          <a className="nav-item" href="#controls"><SlidersHorizontal size={18} /><span>Retrieval</span></a>
        </nav>
        <div className="sidebar-bottom">
          <div className="security-note"><ShieldCheck size={16} /><span>Private workspace</span></div>
          <a className="nav-item muted" href="#controls"><Settings2 size={18} /><span>Settings</span></a>
          <div className="user-row"><div className="avatar">PG</div><div><strong>Priyanka Gosavi</strong><small>Workspace owner</small></div><ChevronDown size={15} /></div>
        </div>
      </aside>

      <section className="workspace" id="workspace">
        <header className="topbar">
          <button className="icon-button" onClick={() => setSidebarOpen((open) => !open)} aria-label="Toggle navigation" title="Toggle navigation"><Menu size={19} /></button>
          <div className="breadcrumb"><span>Knowledge workspace</span><i>/</i><strong>Overview</strong></div>
          <div className="topbar-actions"><a className="icon-button" href="#controls" aria-label="Retrieval settings" title="Retrieval settings"><Settings2 size={18} /></a></div>
        </header>

        <div className="content-grid">
          <section className="intro" aria-labelledby="workspace-title">
            <div className="eyebrow"><span /> Enterprise intelligence</div>
            <h1 id="workspace-title">Make every document<br />a decision surface.</h1>
            <p>Securely index your business records, trace the evidence, and turn dense material into clear next actions.</p>
            <div className="intro-meta"><span><b>{documents.length}</b> active sources</span><span><b>{indexedChunks}</b> indexed chunks</span><span><b>98.4%</b> retrieval confidence</span></div>
          </section>

          <section className="signal-card" aria-label="System signal">
            <div className="signal-top"><span>System signal</span><span className={`status-dot ${apiStatus}`}>{apiStatus === "online" ? "Online" : apiStatus === "offline" ? "Offline" : "Checking"}</span></div>
            <div className="signal-value">{indexedChunks}<span> chunks</span></div>
            <div className="signal-bar"><span style={{ width: apiStatus === "online" ? "100%" : "20%" }} /></div>
            <p>{apiStatus === "online" ? "API connection verified. Your latest source state is shown below." : "Waiting for the document intelligence API."}</p>
          </section>

          <section className="upload-zone" id="documents">
            <input ref={fileInput} type="file" accept="application/pdf,.pdf" multiple onChange={selectFiles} hidden />
            <div className="upload-icon"><UploadCloud size={22} /></div>
            <div><h2>Add source material</h2><p>PDF documents</p></div>
            <div className="upload-actions">
              {files.length > 0 && <span className="selected-count">{files.length} selected</span>}
              <button className="button secondary" onClick={() => fileInput.current?.click()}><Plus size={16} />Choose files</button>
              <button className="button primary" onClick={indexFiles} disabled={!files.length || isIndexing}>{isIndexing ? <LoaderCircle className="spin" size={16} /> : <Sparkles size={16} />}{isIndexing ? "Indexing" : "Index sources"}</button>
            </div>
          </section>

          <section className="document-list" aria-label="Indexed documents">
            <div className="section-heading"><div><span className="section-kicker">Knowledge base</span><h2>Indexed sources</h2></div><span className="source-total">{documents.length} sources</span></div>
            <div className="document-table">
              {documents.length === 0 && <div className="empty-state">No source documents are indexed.</div>}
              {documents.map((document, index) => <div className="document-row" key={document.filename}>
                <div className={`file-glyph ${index % 3 === 0 ? "mint" : index % 3 === 1 ? "cyan" : "amber"}`}><FileText size={18} /></div>
                <div className="file-info"><strong>{document.filename}</strong><span>PDF · {document.chunks} chunks</span></div>
                <span className="ready-tag">Indexed</span>
                <button className="icon-button row-action delete-button" onClick={() => deleteDocument(document.filename)} disabled={deleting === document.filename} aria-label={`Delete ${document.filename}`} title="Delete document">{deleting === document.filename ? <LoaderCircle className="spin" size={17} /> : <Trash2 size={17} />}</button>
              </div>)}
            </div>
          </section>

          <section className="assistant-panel" id="assistant" aria-labelledby="assistant-title">
            <div className="assistant-head"><div><span className="section-kicker">Analysis session</span><h2 id="assistant-title">Ask the knowledge base</h2></div><span className="model-tag"><span /> Groq Llama</span></div>
            <div className="conversation">
              {messages.map((message, index) => <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                {message.role === "assistant" && <div className="assistant-avatar"><Bot size={15} /></div>}
                <div className="message-content"><p>{message.text}</p>{message.citations && message.citations.length > 0 && <div className="citations">{message.citations.map((citation, citationIndex) => <span key={`${citation.source}-${citationIndex}`}>{citation.source} · p. {citation.page}</span>)}</div>}{message.latency !== undefined && <small className="latency">Answered in {message.latency.toFixed(1)}s</small>}</div>
              </div>)}
              {isQuerying && <div className="message assistant"><div className="assistant-avatar"><Bot size={15} /></div><div className="message-content"><p className="thinking"><LoaderCircle className="spin" size={14} /> Analyzing sources</p></div></div>}
            </div>
            <form className="query-box" onSubmit={submitQuestion}>
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Ask a question about your sources" aria-label="Question" disabled={isQuerying} />
              <button type="submit" className="send-button" aria-label="Send question" title="Send question" disabled={!query.trim() || isQuerying}>{isQuerying ? <LoaderCircle className="spin" size={17} /> : <Send size={17} />}</button>
            </form>
            {error && <p className="error-message">{error}</p>}
          </section>

          <section className="control-strip" id="controls">
            <div><span className="section-kicker">Retrieval profile</span><strong>Evidence-first</strong></div>
            <label className="control-stat"><span>Candidate sources</span><input type="number" value={topK} min="1" max="20" onChange={(event) => setTopK(Math.max(1, Math.min(20, Number(event.target.value) || 1)))} /></label>
            <label className="control-stat"><span>Final citations</span><input type="number" value={rerankTopK} min="1" max="10" onChange={(event) => setRerankTopK(Math.max(1, Math.min(10, Number(event.target.value) || 1)))} /></label>
            <label className="switch-control"><input type="checkbox" checked={useMmr} onChange={(event) => setUseMmr(event.target.checked)} /><span aria-hidden="true" /><b>MMR reranking</b></label>
          </section>
        </div>
      </section>
    </main>
  );
}
