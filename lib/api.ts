export type DocumentInfo = {
  filename: string;
  chunks: number;
};

export type Health = {
  status: string;
  model: string;
  total_chunks: number;
};

export type Citation = {
  source: string;
  page: number | string;
  relevance: number;
};

export type QueryResult = {
  answer: string;
  citations: Citation[];
  latency: number;
};

const baseUrl = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${baseUrl}${path}`, { ...init, cache: "no-store" });
  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new Error(payload?.detail || `Request failed with status ${response.status}.`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  health: () => request<Health>("/health"),
  documents: async () => (await request<{ documents: DocumentInfo[] }>("/documents/")).documents,
  index: (file: File) => {
    const data = new FormData();
    data.append("file", file);
    return request<{ filename: string; chunks: number }>("/documents/index", { method: "POST", body: data });
  },
  remove: (filename: string) => request<{ status: string }>(`/documents/${encodeURIComponent(filename)}`, { method: "DELETE" }),
  query: async (question: string, topK: number, rerankTopK: number, useMmr: boolean): Promise<QueryResult> => {
    const response = await request<{
      answer: string;
      latency_sec: number;
      retrieved_chunks: Array<{ metadata: { source?: string; page_num?: number }; relevance_score: number }>;
    }>("/query/", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, top_k: topK, rerank_top_k: rerankTopK, use_mmr: useMmr, prompt_version: "v2" }),
    });
    return {
      answer: response.answer,
      latency: response.latency_sec,
      citations: response.retrieved_chunks.map((chunk) => ({
        source: chunk.metadata.source ?? "Source document",
        page: chunk.metadata.page_num ?? "-",
        relevance: chunk.relevance_score,
      })),
    };
  },
};
