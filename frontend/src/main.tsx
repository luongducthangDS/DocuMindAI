import React, { useState, useRef, useEffect } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

// ── Types ──────────────────────────────────────────────────────────────────────
interface Source {
  index: number;
  title: string;
  dieu_header: string;
  source_url: string;
  score: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  latency_ms?: number;
}

interface Document {
  id: string;
  title: string;
  doc_type: string;
  chunk_count: number;
  so_hieu: string;
  ngay_ban_hanh: string;
}

// ── API ────────────────────────────────────────────────────────────────────────
const BASE = "/api/v1";
const api = {
  async query(query: string, session_id: string) {
    const r = await fetch(`${BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id }),
    });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Query failed");
    return r.json();
  },
  async upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
    if (!r.ok) throw new Error((await r.json()).detail ?? "Upload failed");
    return r.json();
  },
  async documents(): Promise<{ total: number; documents: Document[] }> {
    const r = await fetch(`${BASE}/documents`);
    if (!r.ok) throw new Error("Failed to load documents");
    return r.json();
  },
  async health() {
    const r = await fetch(`${BASE}/health`);
    return r.json();
  },
};

// ── Helpers ────────────────────────────────────────────────────────────────────
function genSessionId() {
  return "s-" + Math.random().toString(36).slice(2, 10);
}

const SUGGESTED = [
  "Điều kiện để được hoàn thuế GTGT là gì?",
  "Mức phạt vi phạm hành chính trong lĩnh vực thuế?",
  "Thủ tục đăng ký kinh doanh hộ cá thể?",
  "Quy định về hợp đồng lao động theo thời vụ?",
  "Điều kiện thành lập công ty TNHH một thành viên?",
];

// ── Markdown renderer (lightweight) ───────────────────────────────────────────
function MdText({ text }: { text: string }) {
  return (
    <div className="md">
      {text.split("\n").map((line, i) => {
        if (line.startsWith("### ")) return <h3 key={i}>{line.slice(4)}</h3>;
        if (line.startsWith("## ")) return <h2 key={i}>{line.slice(3)}</h2>;
        if (line.startsWith("# ")) return <h1 key={i}>{line.slice(2)}</h1>;
        if (/^[\*\-]\s/.test(line)) return <p key={i} className="li">• {line.slice(2)}</p>;
        if (/^\d+\.\s/.test(line)) return <p key={i} className="li">{line}</p>;
        if (line.trim() === "---") return <hr key={i} />;
        if (line.trim() === "") return <div key={i} className="br" />;
        // Bold inline
        const parts = line.split(/\*\*(.*?)\*\*/g);
        if (parts.length > 1) {
          return (
            <p key={i}>
              {parts.map((p, j) => (j % 2 === 1 ? <strong key={j}>{p}</strong> : p))}
            </p>
          );
        }
        return <p key={i}>{line}</p>;
      })}
    </div>
  );
}

// ── Source card ────────────────────────────────────────────────────────────────
function SourceCard({ src }: { src: Source }) {
  const score = Math.round(src.score * 100);
  return (
    <div className="source-card">
      <div className="source-header">
        <span className="source-index">[{src.index}]</span>
        <span className="source-score">{score}%</span>
      </div>
      {src.dieu_header && <div className="source-dieu">{src.dieu_header}</div>}
      <div className="source-title">{src.title || "Văn bản pháp luật"}</div>
      {src.source_url && (
        <a href={src.source_url} target="_blank" rel="noreferrer" className="source-link">
          Xem nguồn →
        </a>
      )}
    </div>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function App() {
  const [sessionId] = useState(genSessionId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"chat" | "docs" | "upload">("chat");
  const [docs, setDocs] = useState<Document[]>([]);
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [health, setHealth] = useState<"ok" | "degraded" | "error" | "unknown">("unknown");
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    api.health().then((d) => setHealth(d.status ?? "unknown")).catch(() => setHealth("error"));
  }, []);

  async function loadDocs() {
    if (docsLoaded) return;
    try {
      const d = await api.documents();
      setDocs(d.documents ?? []);
      setDocsLoaded(true);
    } catch {
      setDocs([]);
    }
  }

  async function send(q: string) {
    if (!q.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setBusy(true);
    try {
      const d = await api.query(q, sessionId);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: d.answer ?? "",
          sources: d.sources ?? [],
          latency_ms: d.latency_ms,
        },
      ]);
    } catch (e: unknown) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `❌ ${(e as Error).message}` },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setUploadStatus("❌ Chỉ hỗ trợ file PDF.");
      return;
    }
    setBusy(true);
    setUploadStatus("⏳ Đang xử lý...");
    try {
      const d = await api.upload(file);
      setUploadStatus(
        `✅ Đã lập chỉ mục "${d.document_title}" — ${d.indexed_chunks} đoạn văn bản.`
      );
      setDocsLoaded(false); // refresh docs list
    } catch (e: unknown) {
      setUploadStatus(`❌ ${(e as Error).message}`);
    } finally {
      setBusy(false);
    }
  }

  const healthDot = health === "ok" ? "dot-green" : health === "degraded" ? "dot-yellow" : "dot-red";

  return (
    <div className="layout">
      {/* ── Sidebar ── */}
      <aside className="sidebar">
        <div className="brand">
          <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
            <rect width="28" height="28" rx="8" fill="#0ea5e9" />
            <path d="M8 8h12M8 13h12M8 18h8" stroke="white" strokeWidth="2" strokeLinecap="round" />
            <circle cx="21" cy="19" r="4" fill="#22c55e" />
            <path d="M19.5 19l1 1 1.5-1.5" stroke="white" strokeWidth="1.2" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="brand-name">DocuMind AI</div>
            <div className="brand-sub">Pháp luật Việt Nam</div>
          </div>
        </div>

        <div className="health-row">
          <span className={`dot ${healthDot}`} />
          <span className="health-label">
            {health === "ok" ? "Hệ thống bình thường" : health === "degraded" ? "Suy giảm" : health === "unknown" ? "Đang kiểm tra…" : "Lỗi kết nối"}
          </span>
        </div>

        <nav>
          <button
            className={`nav-item${tab === "chat" ? " active" : ""}`}
            onClick={() => setTab("chat")}
          >
            💬 Hỏi đáp pháp luật
          </button>
          <button
            className={`nav-item${tab === "docs" ? " active" : ""}`}
            onClick={() => { setTab("docs"); loadDocs(); }}
          >
            📚 Văn bản đã lập chỉ mục
          </button>
          <button
            className={`nav-item${tab === "upload" ? " active" : ""}`}
            onClick={() => setTab("upload")}
          >
            📤 Tải lên văn bản
          </button>
        </nav>

        <div className="divider" />

        <div className="sugg-label">Câu hỏi gợi ý</div>
        <div className="suggestions">
          {SUGGESTED.map((s) => (
            <button
              key={s}
              className="chip"
              onClick={() => { setTab("chat"); send(s); }}
            >
              {s}
            </button>
          ))}
        </div>

        <div className="session-info">
          <span className="session-label">Phiên:</span>
          <span className="session-id">{sessionId}</span>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        {/* Chat */}
        {tab === "chat" && (
          <div className="panel chat-panel">
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty">
                  <div className="empty-icon">⚖️</div>
                  <div className="empty-title">Tra cứu văn bản pháp luật</div>
                  <div className="empty-sub">
                    Đặt câu hỏi bằng tiếng Việt — trả lời kèm trích dẫn điều khoản
                  </div>
                </div>
              )}

              {messages.map((msg, i) => (
                <div key={i} className={`msg-row ${msg.role}`}>
                  <div className="bubble">
                    {msg.role === "assistant" ? (
                      <>
                        <MdText text={msg.content} />
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="sources-section">
                            <div className="sources-label">Nguồn tham khảo</div>
                            <div className="sources-list">
                              {msg.sources.map((src) => (
                                <SourceCard key={src.index} src={src} />
                              ))}
                            </div>
                          </div>
                        )}
                        {msg.latency_ms && (
                          <div className="latency">{msg.latency_ms}ms</div>
                        )}
                      </>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
              ))}

              {busy && tab === "chat" && (
                <div className="msg-row assistant">
                  <div className="bubble typing">
                    <span /><span /><span />
                  </div>
                </div>
              )}
              <div ref={bottomRef} />
            </div>

            <div className="input-bar">
              <textarea
                rows={2}
                placeholder="Nhập câu hỏi pháp luật… (Enter để gửi)"
                value={question}
                disabled={busy}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    send(question);
                  }
                }}
              />
              <button
                className="btn-primary"
                disabled={busy || !question.trim()}
                onClick={() => send(question)}
              >
                Gửi câu hỏi ↵
              </button>
            </div>
          </div>
        )}

        {/* Docs */}
        {tab === "docs" && (
          <div className="panel">
            <div className="panel-head">
              Văn bản đã lập chỉ mục
              <span className="muted"> — {docs.length} văn bản</span>
            </div>
            {docs.length === 0 ? (
              <div className="empty">
                <div className="empty-icon">📭</div>
                <div className="empty-title">Chưa có văn bản nào</div>
                <div className="empty-sub">Tải lên PDF để bắt đầu</div>
              </div>
            ) : (
              <div className="doc-grid">
                {docs.map((doc) => (
                  <div key={doc.id} className="doc-card">
                    <div className="doc-title">{doc.title}</div>
                    <div className="doc-meta">
                      {doc.so_hieu && <span className="doc-tag">{doc.so_hieu}</span>}
                      {doc.doc_type && <span className="doc-tag">{doc.doc_type}</span>}
                    </div>
                    <div className="doc-chunks">{doc.chunk_count} đoạn</div>
                    {doc.ngay_ban_hanh && (
                      <div className="doc-date">Ban hành: {doc.ngay_ban_hanh}</div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Upload */}
        {tab === "upload" && (
          <div className="panel">
            <div className="panel-head">Tải lên văn bản PDF</div>
            <div className="upload-zone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault();
                const f = e.dataTransfer.files[0];
                if (f) handleUpload(f);
              }}
              onClick={() => fileRef.current?.click()}
            >
              <input
                ref={fileRef}
                type="file"
                accept=".pdf"
                style={{ display: "none" }}
                onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) handleUpload(f);
                }}
              />
              <div className="upload-icon">📄</div>
              <div className="upload-text">
                {busy ? "Đang xử lý…" : "Kéo thả PDF hoặc click để chọn"}
              </div>
              <div className="upload-sub">Hỗ trợ: Luật, Nghị định, Thông tư, Quyết định</div>
            </div>
            {uploadStatus && (
              <div className={`upload-status ${uploadStatus.startsWith("✅") ? "success" : uploadStatus.startsWith("❌") ? "error" : "info"}`}>
                {uploadStatus}
              </div>
            )}
            <div className="upload-note">
              <strong>Lưu ý:</strong> File PDF sẽ được phân tích theo từng điều khoản,
              tự động lập chỉ mục để tìm kiếm ngữ nghĩa. Quá trình mất 10–60 giây tùy kích thước.
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
