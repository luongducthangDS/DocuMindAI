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

interface ThinkingStep {
  label: string;
  detail: string;
  ms: number;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  latency_ms?: number;
  used_llm?: string;
  steps?: ThinkingStep[];
}

interface Bookmark {
  id: string;
  question: string;
  answer: string;
  sources: Source[];
  savedAt: number;
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
// VITE_API_URL: set khi frontend và backend deploy tách domain (vd. Vercel + Render).
// Không set -> mặc định "/api/v1" (dev local qua Vite proxy, hoặc same-origin).
const BASE = `${import.meta.env.VITE_API_URL ?? ""}/api/v1`;
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

const LS_SESSION_KEY = "documind_session_id";
const LS_MESSAGES_KEY = "documind_messages";

function loadStoredSessionId(): string {
  try {
    return localStorage.getItem(LS_SESSION_KEY) || genSessionId();
  } catch {
    return genSessionId();
  }
}

function loadStoredMessages(): Message[] {
  try {
    const raw = localStorage.getItem(LS_MESSAGES_KEY);
    return raw ? (JSON.parse(raw) as Message[]) : [];
  } catch {
    return [];
  }
}

const LS_BOOKMARKS_KEY = "documind_bookmarks";

function loadStoredBookmarks(): Bookmark[] {
  try {
    const raw = localStorage.getItem(LS_BOOKMARKS_KEY);
    return raw ? (JSON.parse(raw) as Bookmark[]) : [];
  } catch {
    return [];
  }
}

type Theme = "light" | "dark";
const LS_THEME_KEY = "documind_theme";

function loadStoredTheme(): Theme {
  try {
    return localStorage.getItem(LS_THEME_KEY) === "dark" ? "dark" : "light";
  } catch {
    return "light";
  }
}

function bookmarkId(question: string, answer: string): string {
  // Cheap non-cryptographic hash — only needs to be stable + unique enough
  // to dedupe identical Q&A pairs, not collision-proof.
  let h = 0;
  const s = question + "|" + answer;
  for (let i = 0; i < s.length; i++) {
    h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  }
  return "bm-" + h;
}

const SUGGESTED = [
  "Điều kiện để được vay tín chấp là gì?",
  "Cách tính lãi suất trả góp như thế nào?",
  "Hạn mức thẻ tín dụng tối đa là bao nhiêu?",
  "Phí thường niên áp dụng trong trường hợp nào?",
  "Quy trình mở tài khoản doanh nghiệp gồm những bước gì?",
];

// ── Citation helpers ──────────────────────────────────────────────────────────
function sourceDomId(msgIndex: number, citationN: number) {
  return `source-${msgIndex}-${citationN}`;
}

function scrollToSource(msgIndex: number, citationN: number) {
  const el = document.getElementById(sourceDomId(msgIndex, citationN));
  if (!el) return;
  el.scrollIntoView({ behavior: "smooth", block: "center" });
  el.classList.add("source-card-flash");
  setTimeout(() => el.classList.remove("source-card-flash"), 1200);
}

// Splits text on "[N]" citation markers, turning each into a clickable button
// that scrolls to (and briefly highlights) the matching source card below.
// The LLM isn't consistent about combined citations ("[1, 2]") vs separate
// ("[1], [2]") — handle both by splitting each bracket group's numbers into
// individual clickable buttons.
function renderWithCitations(text: string, msgIndex: number): React.ReactNode[] {
  const parts = text.split(/(\[[\d,\s]+\])/g);
  return parts.flatMap<React.ReactNode>((part, j) => {
    const m = part.match(/^\[([\d,\s]+)\]$/);
    if (!m) return [part];
    const numbers = m[1].split(",").map((s) => s.trim()).filter((s) => /^\d+$/.test(s));
    if (numbers.length === 0) return [part];
    return numbers.map((numStr, k) => {
      const n = Number(numStr);
      return (
        <React.Fragment key={`${j}-${k}`}>
          {k > 0 && ", "}
          <button
            className="citation-link"
            onClick={() => scrollToSource(msgIndex, n)}
            title={`Xem nguồn [${n}]`}
          >
            [{n}]
          </button>
        </React.Fragment>
      );
    });
  });
}

// ── Markdown renderer (lightweight) ───────────────────────────────────────────
// Renders one line's content: bold (**...**) plus citation markers. Shared
// between plain paragraphs and table cells so both get the same inline formatting.
function renderInline(text: string, msgIndex: number): React.ReactNode {
  const parts = text.split(/\*\*(.*?)\*\*/g);
  if (parts.length === 1) return renderWithCitations(text, msgIndex);
  return parts.map((p, j) => (j % 2 === 1 ? <strong key={j}>{p}</strong> : <React.Fragment key={j}>{renderWithCitations(p, msgIndex)}</React.Fragment>));
}

const TABLE_ROW_RE = /^\s*\|(.+)\|\s*$/;
const TABLE_SEPARATOR_RE = /^\s*\|?[\s:-]+\|[\s:|-]*\|?\s*$/;

function splitTableRow(line: string): string[] {
  return line.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
}

// Markdown tables span multiple consecutive lines (header + separator + rows),
// so unlike every other construct here they can't be handled one line at a time.
function renderTable(lines: string[], start: number, msgIndex: number): { node: React.ReactNode; next: number } | null {
  if (!TABLE_ROW_RE.test(lines[start]) || !lines[start + 1] || !TABLE_SEPARATOR_RE.test(lines[start + 1])) {
    return null;
  }
  const header = splitTableRow(lines[start]);
  const rows: string[][] = [];
  let i = start + 2;
  while (i < lines.length && TABLE_ROW_RE.test(lines[i])) {
    rows.push(splitTableRow(lines[i]));
    i++;
  }
  const node = (
    <div className="md-table-wrap" key={`table-${start}`}>
      <table className="md-table">
        <thead>
          <tr>{header.map((h, j) => <th key={j}>{renderInline(h, msgIndex)}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, ri) => (
            <tr key={ri}>{row.map((cell, ci) => <td key={ci}>{renderInline(cell, msgIndex)}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
  );
  return { node, next: i };
}

function MdText({ text, msgIndex }: { text: string; msgIndex: number }) {
  const lines = text.split("\n");
  const nodes: React.ReactNode[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const table = renderTable(lines, i, msgIndex);
    if (table) {
      nodes.push(table.node);
      i = table.next;
      continue;
    }
    if (line.startsWith("### ")) nodes.push(<h3 key={i}>{line.slice(4)}</h3>);
    else if (line.startsWith("## ")) nodes.push(<h2 key={i}>{line.slice(3)}</h2>);
    else if (line.startsWith("# ")) nodes.push(<h1 key={i}>{line.slice(2)}</h1>);
    else if (/^[\*\-]\s/.test(line)) nodes.push(<p key={i} className="li">• {renderInline(line.slice(2), msgIndex)}</p>);
    else if (/^\d+\.\s/.test(line)) nodes.push(<p key={i} className="li">{renderInline(line, msgIndex)}</p>);
    else if (line.trim() === "---") nodes.push(<hr key={i} />);
    else if (line.trim() === "") nodes.push(<div key={i} className="br" />);
    else nodes.push(<p key={i}>{renderInline(line, msgIndex)}</p>);
    i++;
  }
  return <div className="md">{nodes}</div>;
}

// ── Thinking panel ────────────────────────────────────────────────────────────
const STEP_ICONS: Record<string, string> = {
  "Phân loại câu hỏi": "🔍",
  "Tìm kiếm tài liệu": "📚",
  "Tổng hợp câu trả lời": "🤖",
};

function ThinkingPanel({ steps }: { steps: ThinkingStep[] }) {
  const [open, setOpen] = React.useState(false);
  if (!steps || steps.length === 0) return null;
  return (
    <div className="thinking-panel">
      <button className="thinking-toggle" onClick={() => setOpen(!open)}>
        <span className="thinking-icon">{open ? "▾" : "▸"}</span>
        <span>Quá trình xử lý</span>
        <span className="thinking-count">{steps.length} bước</span>
      </button>
      {open && (
        <div className="thinking-steps">
          {steps.map((s, i) => (
            <div key={i} className="thinking-step">
              <span className="step-icon">{STEP_ICONS[s.label] ?? "⚙️"}</span>
              <div className="step-body">
                <span className="step-label">{s.label}</span>
                {s.detail && <span className="step-detail">{s.detail}</span>}
              </div>
              <span className="step-ms">{s.ms}ms</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ── Answer actions (copy / bookmark) ──────────────────────────────────────────
function AnswerActions({
  bookmarked,
  onCopy,
  onToggleBookmark,
}: {
  bookmarked: boolean;
  onCopy: () => Promise<boolean>;
  onToggleBookmark: () => void;
}) {
  const [copied, setCopied] = React.useState(false);
  return (
    <div className="answer-actions">
      <button
        className="icon-btn"
        title="Sao chép câu trả lời"
        onClick={async () => {
          if (await onCopy()) {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
          }
        }}
      >
        {copied ? "✅ Đã sao chép" : "📋 Sao chép"}
      </button>
      <button
        className={`icon-btn${bookmarked ? " icon-btn-active" : ""}`}
        title={bookmarked ? "Bỏ lưu" : "Lưu lại"}
        onClick={onToggleBookmark}
      >
        {bookmarked ? "🔖 Đã lưu" : "🔖 Lưu"}
      </button>
    </div>
  );
}

// ── Source card ────────────────────────────────────────────────────────────────
function SourceCard({ src, msgIndex }: { src: Source; msgIndex: number }) {
  return (
    <div className="source-card" id={sourceDomId(msgIndex, src.index)}>
      <div className="source-header">
        <span className="source-index">[{src.index}]</span>
        {src.dieu_header && <span className="source-dieu">{src.dieu_header}</span>}
      </div>
      <div className="source-title">{src.title || "Văn bản ngân hàng"}</div>
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
  const [sessionId, setSessionId] = useState(loadStoredSessionId);
  const [messages, setMessages] = useState<Message[]>(loadStoredMessages);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<"chat" | "docs" | "upload" | "bookmarks">("chat");
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(loadStoredBookmarks);
  const [docs, setDocs] = useState<Document[]>([]);
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [health, setHealth] = useState<"ok" | "degraded" | "error" | "unknown">("unknown");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(loadStoredTheme);
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    document.body.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(LS_THEME_KEY, theme);
    } catch {
      // localStorage unavailable — theme just won't persist across reloads
    }
  }, [theme]);

  // Persist conversation so a page refresh doesn't wipe it — the backend's
  // ShortTermMemory is keyed by sessionId, but only the frontend can survive reload.
  useEffect(() => {
    try {
      localStorage.setItem(LS_SESSION_KEY, sessionId);
      localStorage.setItem(LS_MESSAGES_KEY, JSON.stringify(messages));
    } catch {
      // localStorage unavailable (private mode, quota) — conversation just won't persist
    }
  }, [sessionId, messages]);

  useEffect(() => {
    try {
      localStorage.setItem(LS_BOOKMARKS_KEY, JSON.stringify(bookmarks));
    } catch {
      // localStorage unavailable — bookmarks just won't persist
    }
  }, [bookmarks]);

  function isBookmarked(question: string, answer: string): boolean {
    const id = bookmarkId(question, answer);
    return bookmarks.some((b) => b.id === id);
  }

  function toggleBookmark(question: string, answer: string, sources: Source[]) {
    const id = bookmarkId(question, answer);
    setBookmarks((prev) =>
      prev.some((b) => b.id === id)
        ? prev.filter((b) => b.id !== id)
        : [{ id, question, answer, sources, savedAt: Date.now() }, ...prev]
    );
  }

  async function copyAnswer(text: string) {
    try {
      await navigator.clipboard.writeText(text);
      return true;
    } catch {
      return false;
    }
  }

  function newConversation() {
    const id = genSessionId();
    setSessionId(id);
    setMessages([]);
    setSidebarOpen(false);
  }

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
          used_llm: d.used_llm,
          steps: d.steps ?? [],
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
  const healthLabel = health === "ok" ? "Hệ thống bình thường" : health === "degraded" ? "Suy giảm" : health === "unknown" ? "Đang kiểm tra…" : "Lỗi kết nối";

  const NAV_DEFS: { id: typeof tab; icon: string; label: string; badge?: string }[] = [
    { id: "chat", icon: "💬", label: "Hỏi đáp tài liệu ngân hàng" },
    { id: "docs", icon: "📚", label: "Văn bản đã lập chỉ mục", badge: docsLoaded ? String(docs.length) : undefined },
    { id: "bookmarks", icon: "🔖", label: "Đã lưu", badge: bookmarks.length > 0 ? String(bookmarks.length) : undefined },
    { id: "upload", icon: "📤", label: "Tải lên văn bản" },
  ];

  const HEADERS: Record<typeof tab, [string, string]> = {
    chat: ["Hỏi đáp tài liệu ngân hàng", "Trả lời kèm trích dẫn nguồn cụ thể"],
    docs: ["Văn bản đã lập chỉ mục", `${docs.length} văn bản`],
    bookmarks: ["Câu trả lời đã lưu", `${bookmarks.length} mục`],
    upload: ["Tải lên văn bản", "PDF được tách theo Điều / Khoản"],
  };
  const [headerTitle, headerSub] = HEADERS[tab];

  return (
    <div className="layout">
      {/* ── Mobile topbar ── */}
      <div className="mobile-topbar">
        <button className="hamburger" onClick={() => setSidebarOpen(true)} aria-label="Mở menu">
          ☰
        </button>
        <span className="mobile-topbar-title">DocuMind AI</span>
      </div>
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* ── Sidebar ── */}
      <aside className={`sidebar${sidebarOpen ? " sidebar-open" : ""}`}>
        <div className="brand">
          <svg width="34" height="34" viewBox="0 0 34 34" fill="none">
            <rect width="34" height="34" rx="9" style={{ fill: "var(--accent)" }} />
            <path d="M10 10h14M10 15.5h14M10 21h9" stroke="#fdfaf2" strokeWidth="2.2" strokeLinecap="round" />
            <circle cx="24" cy="23.5" r="4.6" style={{ fill: "var(--highlight)" }} />
            <path d="M22.2 23.5l1.2 1.2 2-2.2" stroke="#fdfaf2" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <div>
            <div className="brand-name">DocuMind</div>
            <div className="brand-sub">Trợ lý tra cứu tài liệu ngân hàng</div>
          </div>
          <button className="sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Đóng menu">
            ✕
          </button>
        </div>

        <nav>
          {NAV_DEFS.map((nv) => (
            <button
              key={nv.id}
              className={`nav-item${tab === nv.id ? " active" : ""}`}
              onClick={() => { setTab(nv.id); if (nv.id === "docs") loadDocs(); setSidebarOpen(false); }}
            >
              <span className="nav-bar" />
              <span>{nv.icon}</span>
              <span className="nav-label">{nv.label}</span>
              {nv.badge && <span className="nav-badge">{nv.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="divider" />

        <div>
          <div className="sugg-label">Câu hỏi gợi ý</div>
          <div className="suggestions">
            {SUGGESTED.map((s) => (
              <button
                key={s}
                className="chip"
                onClick={() => { setTab("chat"); send(s); setSidebarOpen(false); }}
              >
                {s}
              </button>
            ))}
          </div>
        </div>

        <div className="sidebar-footer">
          <button
            className="new-chat-btn"
            onClick={newConversation}
            disabled={messages.length === 0}
          >
            + Cuộc trò chuyện mới
          </button>

          <div className="status-row">
            <span className="status-left">
              <span className={`dot ${healthDot}`} />
              <span className="health-label">{healthLabel}</span>
            </span>
            <button
              className="theme-toggle"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              title="Đổi giao diện sáng / tối"
            >
              {theme === "light" ? "◐ Tối" : "◑ Sáng"}
            </button>
          </div>

          <div className="session-info">
            <span className="session-label">Phiên:</span>
            <span className="session-id">{sessionId}</span>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        <div className="header-bar">
          <span className="header-title">{headerTitle}</span>
          <span className="header-sub">{headerSub}</span>
        </div>
        {/* Chat */}
        {tab === "chat" && (
          <div className="panel chat-panel">
            <div className="messages">
              {messages.length === 0 && (
                <div className="empty">
                  <div className="empty-icon">🏦</div>
                  <div className="empty-title">Hỏi đáp tài liệu ngân hàng</div>
                  <div className="empty-sub">
                    Đặt câu hỏi về lãi suất, hạn mức, phí dịch vụ, quy trình nghiệp vụ — trả lời kèm trích dẫn quy định
                  </div>
                </div>
              )}

              {messages.map((msg, i) => {
                const noAnswer =
                  msg.role === "assistant" && (!msg.sources || msg.sources.length === 0);
                return (
                <div key={i} className={`msg-row ${msg.role}`}>
                  <div className={`bubble ${noAnswer ? "bubble-noanswer" : ""}`}>
                    {msg.role === "assistant" ? (
                      <>
                        {noAnswer && (
                          <div className="noanswer-flag">
                            <span className="noanswer-icon">🔍</span>
                            <span>Không tìm thấy trong dữ liệu hiện có</span>
                          </div>
                        )}
                        {msg.steps && msg.steps.length > 0 && (
                          <ThinkingPanel steps={msg.steps} />
                        )}
                        <MdText text={msg.content} msgIndex={i} />
                        {msg.sources && msg.sources.length > 0 && (
                          <div className="sources-section">
                            <div className="sources-label">Nguồn trích dẫn</div>
                            <div className="sources-list">
                              {msg.sources.map((src) => (
                                <SourceCard key={src.index} src={src} msgIndex={i} />
                              ))}
                            </div>
                          </div>
                        )}
                        {!noAnswer && (
                          <AnswerActions
                            bookmarked={isBookmarked(messages[i - 1]?.content ?? "", msg.content)}
                            onCopy={() => copyAnswer(msg.content)}
                            onToggleBookmark={() =>
                              toggleBookmark(messages[i - 1]?.content ?? "", msg.content, msg.sources ?? [])
                            }
                          />
                        )}
                        <div className="msg-meta">
                          {msg.used_llm && msg.used_llm !== "none" && (
                            <span className="llm-badge">
                              {msg.used_llm === "groq" ? "Llama 3.3 70B" : msg.used_llm === "gemini" ? "Gemini" : msg.used_llm === "extractive_fallback" ? "Trích xuất trực tiếp" : msg.used_llm}
                            </span>
                          )}
                          {msg.latency_ms && (
                            <span className="latency">{msg.latency_ms}ms</span>
                          )}
                        </div>
                      </>
                    ) : (
                      msg.content
                    )}
                  </div>
                </div>
                );
              })}

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
              <div className="input-bar-inner">
                <textarea
                  rows={2}
                  placeholder="Đặt câu hỏi về tài liệu ngân hàng… (Enter để gửi)"
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
                  Gửi ↵
                </button>
              </div>
              <div className="input-disclaimer">
                Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn nghiệp vụ ngân hàng chính thức.
              </div>
            </div>
          </div>
        )}

        {/* Docs */}
        {tab === "docs" && (
          <div className="panel">
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
                      {doc.doc_type && <span className="doc-tag">{doc.doc_type}</span>}
                    </div>
                    {doc.so_hieu && <div className="muted">{doc.so_hieu}</div>}
                    <div className="doc-footer">
                      <span>{doc.chunk_count} đoạn</span>
                      {doc.ngay_ban_hanh && <span>Ban hành: {doc.ngay_ban_hanh}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Bookmarks */}
        {tab === "bookmarks" && (
          <div className="panel">
            {bookmarks.length === 0 ? (
              <div className="empty">
                <div className="empty-icon">🔖</div>
                <div className="empty-title">Chưa lưu câu hỏi nào</div>
                <div className="empty-sub">
                  Bấm "🔖 Lưu" dưới một câu trả lời để xem lại sau
                </div>
              </div>
            ) : (
              <div className="messages">
                {bookmarks.map((b, i) => (
                  <div key={b.id} className="bookmark-card">
                    <div className="bookmark-question">{b.question}</div>
                    <MdText text={b.answer} msgIndex={1000 + i} />
                    {b.sources.length > 0 && (
                      <div className="sources-section">
                        <div className="sources-label">Nguồn trích dẫn</div>
                        <div className="sources-list">
                          {b.sources.map((src) => (
                            <SourceCard key={src.index} src={src} msgIndex={1000 + i} />
                          ))}
                        </div>
                      </div>
                    )}
                    <button
                      className="icon-btn"
                      onClick={() => setBookmarks((prev) => prev.filter((x) => x.id !== b.id))}
                    >
                      🗑️ Bỏ lưu
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Upload */}
        {tab === "upload" && (
          <div className="panel">
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
              <div className="upload-sub">Hỗ trợ: Quy định, Biểu phí, Thông tư, Quyết định — tối đa 20MB</div>
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
