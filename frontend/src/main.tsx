import React, { useState, useRef, useEffect, useLayoutEffect } from "react";
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
  streaming?: boolean; // true trong lúc chờ/nhận token qua WebSocket
  error?: boolean; // true nếu đây là thông báo lỗi hệ thống, không phải câu trả lời
  stopped?: boolean; // người dùng bấm Dừng giữa chừng — phần chữ đã stream được giữ lại
  as_of?: string; // ngày áp dụng (YYYY-MM-DD) đã gửi kèm câu hỏi; không có = hôm nay
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
const API_URL = import.meta.env.VITE_API_URL ?? "";
const BASE = `${API_URL}/api/v1`;
// Cùng origin (API_URL rỗng, dev local qua Vite proxy) -> lấy origin hiện tại;
// khác domain (production, Vercel gọi Render) -> đổi scheme của chính API_URL.
// "https"→"wss", "http"→"ws" (regex chỉ khớp 4 ký tự "http" ở đầu, phần "s"
// còn lại của "https" giữ nguyên).
const WS_BASE = `${(API_URL || (typeof window !== "undefined" ? window.location.origin : "")).replace(/^http/, "ws")}/api/v1`;

// FastAPI trả `detail` string khi lỗi tự viết (HTTPException), nhưng mảng
// [{loc, msg, type}, ...] khi lỗi validate Pydantic tự động (422) — ném thẳng
// mảng/object vào `new Error(...)` từng ra "[object Object]" (String(array)
// nối .toString() của từng phần tử) thay vì lý do lỗi đọc được, vd. câu "hi"
// (2 ký tự) từng bị 422 trước khi kịp tới nhánh chào hỏi.
function extractErrorMessage(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => (d && typeof d === "object" && "msg" in d ? String((d as { msg: unknown }).msg) : String(d)))
      .join("; ") || fallback;
  }
  if (detail && typeof detail === "object") return JSON.stringify(detail);
  return fallback;
}

const api = {
  async query(query: string, session_id: string, as_of_date: string, signal?: AbortSignal) {
    const r = await fetch(`${BASE}/query`, {
      method: "POST",
      signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, session_id, as_of_date: as_of_date || null }),
    });
    if (!r.ok) throw new Error(extractErrorMessage((await r.json()).detail, "Query failed"));
    return r.json();
  },
  async upload(file: File) {
    const form = new FormData();
    form.append("file", file);
    const r = await fetch(`${BASE}/upload`, { method: "POST", body: form });
    if (!r.ok) throw new Error(extractErrorMessage((await r.json()).detail, "Upload failed"));
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
    const msgs = raw ? (JSON.parse(raw) as Message[]) : [];
    // Reload giữa lúc stream: placeholder rỗng bỏ đi (không thì "..." treo mãi),
    // phần chữ đã nhận được giữ lại như bấm Dừng.
    return msgs
      .filter((m) => !(m.streaming && !m.content))
      .map((m) => (m.streaming ? { ...m, streaming: false, stopped: true } : m));
  } catch {
    return [];
  }
}

// Đoán nhanh phía client để CHỌN ĐƯỜNG TRUYỀN (REST đủ tính năng nhưng chờ
// xong mới hiện, hay WS stream từng chữ nhưng chỉ trả lời hỏi-đáp thường) —
// không phải quyết định định tuyến thật, cái đó vẫn ở
// src/agent/graph.py::_keyword_classify (giữ đúng cùng bộ từ khoá, đừng sửa
// lệch 2 bên). Đoán sai không hỏng gì: WS vẫn trả lời được bằng RAG chung,
// chỉ là không dùng tool chuyên biệt so sánh/tóm tắt/báo cáo/tuân thủ.
const _COMPLIANCE_PHRASES = ["có được", "có đủ điều kiện", "có bị", "có đạt", "quá", "vượt quá"];

function needsRest(query: string): boolean {
  const q = query.toLowerCase();
  if (["so sánh", "khác nhau", "giống nhau", "phân biệt"].some((w) => q.includes(w))) return true;
  if (["tóm tắt", "tóm lược", "nội dung chính"].some((w) => q.includes(w))) return true;
  if (["báo cáo", "xuất pdf", "tổng hợp"].some((w) => q.includes(w))) return true;
  if (_COMPLIANCE_PHRASES.some((p) => q.includes(p)) && /\d/.test(q)) return true;
  return false;
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

// Phạm vi sản phẩm — đối ứng của DOMAIN_* trong src/config.py. Đổi corpus thì
// sửa cả hai chỗ; mọi nhãn hiển thị phải lấy từ đây, không viết lại rải rác.
const PRODUCT = {
  tagline: "Trợ lý tra cứu pháp luật lao động & BHXH",
  laws: "Bộ luật Lao động · Luật BHXH · Luật Việc làm",
  heroTitle: "Hỏi đáp luật lao động & BHXH, có dẫn chứng từng điều khoản",
  scope: "pháp luật lao động và bảo hiểm xã hội",
  chatTitle: "Hỏi đáp pháp luật lao động & BHXH",
  emptySub:
    "Đặt câu hỏi về hợp đồng lao động, tiền lương, thời giờ làm việc, BHXH, bảo hiểm thất nghiệp — trả lời kèm trích dẫn điều khoản",
  disclaimer:
    "Câu trả lời chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chính thức.",
  sourceFallback: "Văn bản pháp luật",
};

const SUGGESTED = [
  { topic: "Hợp đồng", q: "Thời gian thử việc tối đa là bao lâu?" },
  { topic: "Thời giờ làm việc", q: "Làm thêm giờ tối đa bao nhiêu giờ trong một năm?" },
  { topic: "Tiền lương", q: "Mức lương tối thiểu vùng I hiện nay là bao nhiêu?" },
  { topic: "Bảo hiểm thất nghiệp", q: "Điều kiện hưởng trợ cấp thất nghiệp là gì?" },
  { topic: "Chấm dứt hợp đồng", q: "Nghỉ việc đúng luật cần báo trước bao nhiêu ngày?" },
  { topic: "Hưu trí", q: "Tuổi nghỉ hưu của người lao động năm 2026 là bao nhiêu?" },
];

const HIGHLIGHTS: { icon: IconName; text: string }[] = [
  { icon: "quote", text: "Trích dẫn đúng điều, khoản" },
  { icon: "clock", text: "Phân biệt văn bản còn / hết hiệu lực" },
  { icon: "shield", text: "Từ chối câu hỏi ngoài phạm vi" },
];

// ── Icons ─────────────────────────────────────────────────────────────────────
// ponytail: bộ path nét 24×24 tự viết thay vì thêm thư viện icon cho ~20 hình.
const ICON_PATHS = {
  chat: "M21 12a8 8 0 0 1-11.6 7.1L4 20l1-4.6A8 8 0 1 1 21 12z",
  book: "M4 19.5A2.5 2.5 0 0 1 6.5 17H20V2H6.5A2.5 2.5 0 0 0 4 4.5v15zM4 19.5A2.5 2.5 0 0 0 6.5 22H20v-5",
  bookmark: "M19 21l-7-5-7 5V5a2 2 0 0 1 2-2h10a2 2 0 0 1 2 2z",
  upload: "M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4M17 8l-5-5-5 5M12 3v12",
  plus: "M12 5v14M5 12h14",
  arrowUp: "M12 19V5M5 12l7-7 7 7",
  stop: "M6 6h12v12H6z",
  copy: "M9 9h11v11H9zM5 15H4V4h11v1",
  check: "M20 6L9 17l-5-5",
  moon: "M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z",
  sun: "M12 17a5 5 0 1 0 0-10 5 5 0 0 0 0 10zM12 1v2M12 21v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M1 12h2M21 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4",
  menu: "M3 6h18M3 12h18M3 18h18",
  x: "M18 6L6 18M6 6l12 12",
  trash: "M3 6h18M8 6V4h8v2M19 6l-1 14H6L5 6",
  file: "M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8zM14 2v6h6",
  search: "M11 19a8 8 0 1 0 0-16 8 8 0 0 0 0 16zM21 21l-4.3-4.3",
  edit: "M12 20h9M16.5 3.5a2.1 2.1 0 0 1 3 3L7 19l-4 1 1-4z",
  quote: "M4 7h6v6H4zM4 13c0 3 1 5 4 6M14 7h6v6h-6zM14 13c0 3 1 5 4 6",
  clock: "M12 22a10 10 0 1 0 0-20 10 10 0 0 0 0 20zM12 6v6l4 2",
  shield: "M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z",
  calendar: "M3 5h18v16H3zM16 3v4M8 3v4M3 10h18",
  chevronDown: "M6 9l6 6 6-6",
  chevronRight: "M9 6l6 6-6 6",
};
type IconName = keyof typeof ICON_PATHS;

function Icon({ name, size = 16 }: { name: IconName; size?: number }) {
  const filled = name === "stop";
  return (
    <svg
      className="icon"
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? "currentColor" : "none"}
      stroke="currentColor"
      strokeWidth={filled ? 0 : 1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d={ICON_PATHS[name]} />
    </svg>
  );
}

function BrandMark({ size = 34 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 34 34" fill="none" aria-hidden="true" className="brand-mark">
      <rect width="34" height="34" rx="9" style={{ fill: "var(--accent)" }} />
      <path d="M10 10h14M10 15.5h14M10 21h9" stroke="#fdfaf2" strokeWidth="2.2" strokeLinecap="round" />
      <circle cx="24" cy="23.5" r="4.6" style={{ fill: "var(--highlight)" }} />
      <path d="M22.2 23.5l1.2 1.2 2-2.2" stroke="#fdfaf2" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ── Citation helpers ──────────────────────────────────────────────────────────
function sourceDomId(msgIndex: number, citationN: number) {
  return `source-${msgIndex}-${citationN}`;
}

// Màn rộng: nguồn nằm ở cột phải (id có tiền tố "panel-"), chỉ hiện nguồn của
// MỘT câu trả lời — báo App đổi sang câu trả lời chứa trích dẫn vừa bấm rồi mới
// tìm thẻ. Màn hẹp: cột phải ẩn, rơi về thẻ nguồn nằm ngay dưới câu trả lời.
// ponytail: setTimeout 0 chờ React vẽ lại cột phải (setState trong onClick được
// commit ngay khi handler kết thúc) thay vì nối callback qua MdText →
// renderInline → renderWithCitations. Không dùng rAF: tab ẩn thì rAF không chạy.
const CITE_EVENT = "documind:cite";

function scrollToSource(msgIndex: number, citationN: number) {
  window.dispatchEvent(new CustomEvent(CITE_EVENT, { detail: msgIndex }));
  setTimeout(() => {
    const id = sourceDomId(msgIndex, citationN);
    const el = [document.getElementById(`panel-${id}`), document.getElementById(id)]
      .find((e) => e && e.offsetParent !== null);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("source-card-flash");
    setTimeout(() => el.classList.remove("source-card-flash"), 1200);
  }, 0);
}

function fmtDate(iso: string) {
  return iso.split("-").reverse().join("/");
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

// ── Agent trace (tiến trình xử lý) ───────────────────────────────────────────
// Nhãn bước khớp đúng label backend phát: REST trả cả mảng steps lúc xong
// (src/agent/graph.py), WS phát từng {"step": ...} ngay khi xong bước
// (src/api/routes/query.py::websocket_stream). Nhãn lạ vẫn hiện trong nhật ký.
const STAGES: { id: string; label: string; short: string; steps: string[] }[] = [
  { id: "understand", label: "Hiểu câu hỏi", short: "Hiểu", steps: ["Phân tích câu hỏi", "Khôi phục dấu tiếng Việt", "Diễn giải câu hỏi theo ngữ cảnh", "Phân loại câu hỏi"] },
  { id: "retrieve", label: "Truy hồi điều khoản", short: "Truy hồi", steps: ["Tìm kiếm tài liệu", "Đánh giá độ liên quan", "Tìm lại với truy vấn khác"] },
  { id: "filter", label: "Lọc hiệu lực", short: "Hiệu lực", steps: ["Lọc theo hiệu lực"] },
  { id: "answer", label: "Soạn trả lời", short: "Trả lời", steps: ["Tổng hợp câu trả lời", "Kiểm định tuân thủ"] },
];

type StageState = "done" | "running" | "pending" | "skipped";

function fmtMs(ms: number) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms}ms`;
}

function AgentTrace({ steps, running, elapsedMs, totalMs }: {
  steps: ThinkingStep[];
  running: boolean;
  elapsedMs: number;
  totalMs?: number;
}) {
  const [open, setOpen] = useState(false);
  const hit = STAGES.map((st) => steps.filter((s) => st.steps.includes(s.label)));
  const lastHit = hit.reduce((acc, h, i) => (h.length ? i : acc), -1);
  const states: StageState[] = STAGES.map((_, i) => {
    if (hit[i].length) return "done";
    if (i < lastHit) return "skipped"; // vd. chào hỏi: bỏ qua truy hồi
    if (!running) return lastHit >= 0 ? "skipped" : "pending";
    return i === lastHit + 1 ? "running" : "pending";
  });
  const doneCount = states.filter((x) => x === "done").length;
  const stepsMs = steps.reduce((a, s) => a + s.ms, 0);
  const total = totalMs ?? stepsMs;
  const showLog = open || running;

  return (
    <div className={`trace${running ? " is-running" : ""}`}>
      <ol className="trace-stepper" aria-label="Tiến trình xử lý">
        {STAGES.map((st, i) => {
          const ms = hit[i].reduce((a, s) => a + s.ms, 0);
          return (
            <li key={st.id} className={`trace-stage ${states[i]}`}>
              <span className="trace-node" aria-hidden="true">
                {states[i] === "done" ? <Icon name="check" size={11} /> : i + 1}
              </span>
              <span className="trace-text">
                <span className="trace-no">Bước {i + 1}</span>
                <span className="trace-label">{st.label}</span>
                <span className="trace-label trace-label-short">{st.short}</span>
                <span className="trace-ms">
                  {states[i] === "done" ? fmtMs(ms) : states[i] === "running" ? fmtMs(Math.max(0, elapsedMs - stepsMs)) : states[i] === "skipped" ? "bỏ qua" : "chờ"}
                </span>
              </span>
            </li>
          );
        })}
      </ol>
      {steps.length > 0 && (
        <div className="trace-log-wrap">
          {!running && (
            <button className="trace-toggle" onClick={() => setOpen(!open)} aria-expanded={open}>
              <Icon name={open ? "chevronDown" : "chevronRight"} size={12} />
              {doneCount}/{STAGES.length} giai đoạn · {steps.length} bước · {fmtMs(total)}
            </button>
          )}
          {showLog && (
            <ol className="trace-log">
              {steps.map((s, i) => (
                <li key={i} className="trace-row">
                  <span className="trace-row-dot" aria-hidden="true" />
                  <span className="trace-row-label">{s.label}</span>
                  {s.detail && <span className="trace-row-detail">{s.detail}</span>}
                  <span className="trace-row-ms">{fmtMs(s.ms)}</span>
                </li>
              ))}
            </ol>
          )}
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
        <Icon name={copied ? "check" : "copy"} size={14} />
        {copied ? "Đã sao chép" : "Sao chép"}
      </button>
      <button
        className={`icon-btn${bookmarked ? " icon-btn-active" : ""}`}
        title={bookmarked ? "Bỏ lưu" : "Lưu lại"}
        onClick={onToggleBookmark}
      >
        <Icon name="bookmark" size={14} />
        {bookmarked ? "Đã lưu" : "Lưu"}
      </button>
    </div>
  );
}

// ── Source card ────────────────────────────────────────────────────────────────
function SourceCard({ src, msgIndex, idPrefix = "" }: { src: Source; msgIndex: number; idPrefix?: string }) {
  return (
    <div className="source-card" id={idPrefix + sourceDomId(msgIndex, src.index)}>
      <div className="source-header">
        <span className="source-index">[{src.index}]</span>
        {src.dieu_header && <span className="source-dieu">{src.dieu_header}</span>}
      </div>
      <div className="source-title">{src.title || PRODUCT.sourceFallback}</div>
      {src.source_url && (
        <a href={src.source_url} target="_blank" rel="noreferrer" className="source-link">
          Xem nguồn →
        </a>
      )}
    </div>
  );
}

// ── Ngày áp dụng (as_of_date) ─────────────────────────────────────────────────
// Rỗng = quy định hiện hành hôm nay. Dùng lịch có sẵn của trình duyệt
// (<input type="date">, ẩn, mở bằng showPicker) thay vì tự dựng date picker.
function AsOfChip({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const ref = useRef<HTMLInputElement>(null);
  function open() {
    const el = ref.current;
    if (!el) return;
    try {
      el.showPicker();
    } catch {
      el.focus(); // trình duyệt cũ không có showPicker
    }
  }
  return (
    <span className="asof-chip">
      <button
        type="button"
        className="asof-open"
        onClick={open}
        title="Tra cứu theo quy định có hiệu lực tại một ngày cụ thể"
      >
        <Icon name="calendar" size={14} />
        <span>Áp dụng tại: <strong>{value ? fmtDate(value) : "hôm nay"}</strong></span>
      </button>
      {value && (
        <button type="button" className="asof-clear" onClick={() => onChange("")} aria-label="Về quy định hiện hành hôm nay">
          <Icon name="x" size={12} />
        </button>
      )}
      <input
        ref={ref}
        type="date"
        className="asof-input"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        aria-label="Ngày áp dụng"
        tabIndex={-1}
      />
    </span>
  );
}

// ── App ────────────────────────────────────────────────────────────────────────
function App() {
  const [sessionId, setSessionId] = useState(loadStoredSessionId);
  const [messages, setMessages] = useState<Message[]>(loadStoredMessages);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [elapsedMs, setElapsedMs] = useState(0); // đếm giờ sống trong lúc chờ/stream câu trả lời
  const [tab, setTab] = useState<"chat" | "docs" | "upload" | "bookmarks">("chat");
  const [bookmarks, setBookmarks] = useState<Bookmark[]>(loadStoredBookmarks);
  const [docs, setDocs] = useState<Document[]>([]);
  const [docsLoaded, setDocsLoaded] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string>("");
  const [health, setHealth] = useState<"ok" | "degraded" | "error" | "unknown">("unknown");
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [theme, setTheme] = useState<Theme>(loadStoredTheme);
  const [asOf, setAsOf] = useState(""); // YYYY-MM-DD, rỗng = hôm nay
  const [panelMsg, setPanelMsg] = useState<number | null>(null); // null = câu trả lời mới nhất có nguồn
  const fileRef = useRef<HTMLInputElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  // Mỗi lượt gửi/huỷ tăng reqRef; callback của lượt cũ (WS/REST về muộn) so id
  // và tự bỏ qua. abortRef đóng WS / abort fetch của lượt đang chạy.
  const reqRef = useRef(0);
  const abortRef = useRef<(() => void) | null>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    const onCite = (e: Event) => {
      const i = (e as CustomEvent<number>).detail;
      if (i < 1000) setPanelMsg(i); // 1000+ = thẻ đã lưu, không thuộc hội thoại
    };
    window.addEventListener(CITE_EVENT, onCite);
    return () => window.removeEventListener(CITE_EVENT, onCite);
  }, []);

  // Ô nhập tự giãn theo nội dung, kể cả khi đổi từ code (huỷ, gợi ý, sửa).
  useLayoutEffect(() => {
    const el = inputRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }, [question, tab]);

  // Đếm giờ sống trong lúc chờ (REST) hoặc đang stream (WS) — vd. "12.3s"
  // cạnh chấm "..." thay vì im lặng không biết còn đang chạy hay đứng hình.
  useEffect(() => {
    if (!busy) {
      setElapsedMs(0);
      return;
    }
    const startedAt = Date.now();
    setElapsedMs(0);
    const id = window.setInterval(() => setElapsedMs(Date.now() - startedAt), 100);
    return () => window.clearInterval(id);
  }, [busy]);

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

  function abortInFlight() {
    reqRef.current++;
    abortRef.current?.();
    abortRef.current = null;
    setBusy(false);
  }

  function newConversation() {
    if (busy) abortInFlight();
    const id = genSessionId();
    setSessionId(id);
    setMessages([]);
    setPanelMsg(null);
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

  // Đường cũ: chờ toàn bộ pipeline xong mới hiện 1 cục câu trả lời. Vẫn dùng
  // cho so sánh/tóm tắt/báo cáo/tuân thủ (chỉ REST làm được), và làm fallback
  // khi WS lỗi/không mở được.
  async function sendViaRest(q: string, reqId: number, asOfDate: string) {
    const ctrl = new AbortController();
    abortRef.current = () => ctrl.abort();
    try {
      const d = await api.query(q, sessionId, asOfDate, ctrl.signal);
      if (reqId !== reqRef.current) return;
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: d.answer ?? "",
          sources: d.sources ?? [],
          latency_ms: d.latency_ms,
          used_llm: d.used_llm,
          steps: d.steps ?? [],
          as_of: asOfDate || undefined,
        },
      ]);
    } catch (e: unknown) {
      if (reqId !== reqRef.current) return;
      setMessages((m) => [
        ...m,
        { role: "assistant", content: `❌ ${(e as Error).message}`, error: true },
      ]);
    } finally {
      if (reqId === reqRef.current) setBusy(false);
    }
  }

  // Đường mới: mở 1 WebSocket riêng cho câu hỏi này (không giữ kết nối sống
  // xuyên suốt phiên — lịch sử hội thoại đã nằm ở server theo session_id, mở
  // mới mỗi câu tránh hẳn việc phải tự dựng cơ chế reconnect khi Render free
  // tier ngủ/rớt kết nối). Server gửi token thô (text frame) xen giữa 2 JSON
  // control message ({"error":...} hoặc {"done":true,"sources":[...]}) — xem
  // src/api/routes/query.py::websocket_stream.
  function sendViaStream(q: string, reqId: number, asOfDate: string) {
    setMessages((m) => [...m, { role: "assistant", content: "", streaming: true, as_of: asOfDate || undefined }]);

    const startedAt = Date.now();
    let settled = false; // true khi đã có done/error, hoặc đã fallback sang REST
    let gotToken = false;

    const updateLast = (patch: Partial<Message>) => {
      setMessages((m) => {
        const next = [...m];
        next[next.length - 1] = { ...next[next.length - 1], ...patch };
        return next;
      });
    };

    const fallbackToRest = () => {
      if (settled) return;
      settled = true;
      setMessages((m) => m.slice(0, -1)); // bỏ placeholder rỗng
      sendViaRest(q, reqId, asOfDate);
    };

    let ws: WebSocket;
    try {
      ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
    } catch {
      fallbackToRest();
      return;
    }

    // Render free tier ngủ sau 15 phút không dùng -> request đầu chờ ~50s;
    // cho đủ thời gian trước khi bỏ cuộc và chuyển sang REST.
    const openTimeout = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        ws.close();
        fallbackToRest();
      }
    }, 60000);

    abortRef.current = () => {
      settled = true;
      clearTimeout(openTimeout);
      ws.close();
    };

    ws.onopen = () => {
      clearTimeout(openTimeout);
      ws.send(JSON.stringify({ query: q, as_of_date: asOfDate || null, progress: true }));
    };

    ws.onmessage = (ev) => {
      if (reqId !== reqRef.current) return;
      const data = ev.data as string;
      let control: { done?: boolean; error?: string; sources?: Source[]; step?: ThinkingStep } | null = null;
      try {
        const parsed = JSON.parse(data);
        if (parsed && typeof parsed === "object" && ("done" in parsed || "error" in parsed || "step" in parsed)) {
          control = parsed;
        }
      } catch {
        // Không parse được JSON => token trả lời thô, không phải control message.
      }

      if (control?.step) {
        const step = control.step;
        setMessages((m) => {
          const next = [...m];
          const last = next[next.length - 1];
          next[next.length - 1] = { ...last, steps: [...(last.steps ?? []), step] };
          return next;
        });
        return;
      }

      if (control) {
        settled = true;
        if (control.error) {
          updateLast({ content: `❌ ${control.error}`, streaming: false, error: true });
        } else {
          updateLast({
            sources: control.sources ?? [],
            streaming: false,
            latency_ms: Date.now() - startedAt,
          });
        }
        setBusy(false);
        ws.close();
        return;
      }

      gotToken = true;
      setMessages((m) => {
        const next = [...m];
        const last = next[next.length - 1];
        next[next.length - 1] = { ...last, content: (last.content ?? "") + data };
        return next;
      });
    };

    ws.onclose = () => {
      clearTimeout(openTimeout);
      if (settled) return;
      // Rớt kết nối trước khi có done/error. Đã có chữ hiện ra thì giữ lại
      // (chuyển sang REST sẽ hỏi lại từ đầu, mất ngữ cảnh phần đã trả lời);
      // chưa có chữ nào thì coi như thử WS thất bại, chuyển REST.
      if (gotToken) {
        settled = true;
        updateLast({ streaming: false, latency_ms: Date.now() - startedAt });
        setBusy(false);
      } else {
        fallbackToRest();
      }
    };
  }

  async function send(q: string) {
    if (!q.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", content: q }]);
    setQuestion("");
    setPanelMsg(null); // cột nguồn theo câu trả lời mới
    setBusy(true);
    const reqId = ++reqRef.current;
    if (needsRest(q)) {
      await sendViaRest(q, reqId, asOf);
    } else {
      sendViaStream(q, reqId, asOf);
    }
  }

  // Dừng lượt đang chờ. Chưa có chữ nào thì gỡ luôn câu hỏi khỏi khung chat;
  // đã stream được một phần thì giữ phần đó, đánh dấu "Đã dừng". Cả hai trường
  // hợp đều trả câu hỏi về ô nhập để sửa rồi gửi lại.
  // ponytail: REST chỉ abort phía client — server vẫn chạy nốt và ghi lượt đó
  // vào memory phiên; WS thì không (server chỉ lưu khi stream xong).
  function cancel() {
    if (!busy) return;
    abortInFlight();

    const last = messages[messages.length - 1];
    const hasPlaceholder = last?.role === "assistant" && !!last.streaming;
    const keepPartial = hasPlaceholder && !!last.content;
    const userIdx = messages.length - (hasPlaceholder ? 2 : 1);
    const userMsg = messages[userIdx];
    if (keepPartial) {
      setMessages([...messages.slice(0, -1), { ...last, streaming: false, stopped: true }]);
    } else if (userMsg?.role === "user") {
      setMessages(messages.slice(0, userIdx));
    }
    if (userMsg?.role === "user" && !question.trim()) setQuestion(userMsg.content);
    inputRef.current?.focus();
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

  const NAV_DEFS: { id: typeof tab; icon: IconName; label: string; badge?: string }[] = [
    { id: "chat", icon: "chat", label: "Hỏi đáp" },
    { id: "docs", icon: "book", label: "Văn bản đã lập chỉ mục", badge: docsLoaded ? String(docs.length) : undefined },
    { id: "bookmarks", icon: "bookmark", label: "Đã lưu", badge: bookmarks.length > 0 ? String(bookmarks.length) : undefined },
    { id: "upload", icon: "upload", label: "Tải lên văn bản" },
  ];

  const HEADERS: Record<typeof tab, [string, string]> = {
    chat: [PRODUCT.chatTitle, "Trả lời kèm trích dẫn điều khoản cụ thể"],
    docs: ["Văn bản đã lập chỉ mục", `${docs.length} văn bản`],
    bookmarks: ["Câu trả lời đã lưu", `${bookmarks.length} mục`],
    upload: ["Tải lên văn bản", "PDF được tách theo Điều / Khoản"],
  };
  const [headerTitle, headerSub] = HEADERS[tab];
  const isLanding = tab === "chat" && messages.length === 0;

  // Cột nguồn bên phải (màn rộng): câu trả lời đang chọn, mặc định cái mới nhất có nguồn.
  const latestSourced = messages.reduce((acc, m, i) => (m.role === "assistant" && m.sources?.length ? i : acc), -1);
  const panelIdx = panelMsg !== null && messages[panelMsg]?.sources?.length ? panelMsg : latestSourced;
  const panelSources = panelIdx >= 0 ? messages[panelIdx].sources ?? [] : [];

  // Điền vào ô nhập thay vì gửi ngay — sửa được trước khi Enter.
  function fillComposer(q: string) {
    setQuestion(q);
    inputRef.current?.focus();
  }

  return (
    <div className="layout">
      {/* ── Mobile topbar ── */}
      <div className="mobile-topbar">
        <button className="icon-only" onClick={() => setSidebarOpen(true)} aria-label="Mở menu">
          <Icon name="menu" size={20} />
        </button>
        <span className="mobile-topbar-title">
          {tab === "chat" && <BrandMark size={24} />}
          {tab === "chat" ? "DocuMind" : headerTitle}
        </span>
        <button
          className="icon-only"
          onClick={() => { newConversation(); setTab("chat"); }}
          disabled={isLanding}
          aria-label="Cuộc trò chuyện mới"
        >
          <Icon name="plus" size={20} />
        </button>
      </div>
      {sidebarOpen && <div className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} />}

      {/* ── Sidebar ── */}
      <aside className={`sidebar${sidebarOpen ? " sidebar-open" : ""}`}>
        <div className="brand">
          <BrandMark size={34} />
          <div className="brand-text">
            <div className="brand-name">DocuMind</div>
            <div className="brand-sub">{PRODUCT.tagline}</div>
          </div>
          <button className="icon-only sidebar-close" onClick={() => setSidebarOpen(false)} aria-label="Đóng menu">
            <Icon name="x" size={18} />
          </button>
        </div>

        <button
          className="new-chat-btn"
          onClick={() => { newConversation(); setTab("chat"); }}
          disabled={isLanding}
        >
          <Icon name="plus" size={16} />
          Cuộc trò chuyện mới
        </button>

        <nav>
          {NAV_DEFS.map((nv) => (
            <button
              key={nv.id}
              className={`nav-item${tab === nv.id ? " active" : ""}`}
              onClick={() => { setTab(nv.id); if (nv.id === "docs") loadDocs(); setSidebarOpen(false); }}
            >
              <Icon name={nv.icon} size={17} />
              <span className="nav-label">{nv.label}</span>
              {nv.badge && <span className="nav-badge">{nv.badge}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className="status-row" title={`Phiên: ${sessionId}`}>
            <span className="status-left">
              <span className={`dot ${healthDot}`} />
              <span className="health-label">{healthLabel}</span>
            </span>
            <button
              className="theme-toggle"
              onClick={() => setTheme((t) => (t === "light" ? "dark" : "light"))}
              title="Đổi giao diện sáng / tối"
            >
              <Icon name={theme === "light" ? "moon" : "sun"} size={13} />
              {theme === "light" ? "Tối" : "Sáng"}
            </button>
          </div>
        </div>
      </aside>

      {/* ── Main ── */}
      <main className="main">
        <div className="header-bar">
          <span className="header-title">{headerTitle}</span>
          <span className="header-sub">{headerSub}</span>
        </div>

        {/* Chat — ô nhập luôn ở cùng vị trí trong cây DOM (giữa hero và gợi ý
            khi trang trống, dưới đáy khi đã có hội thoại) để textarea không bị
            mount lại, không mất focus/nội dung khi chuyển qua lại. */}
        {tab === "chat" && (
          <div className="chat-row">
          <div className={`panel chat-panel${isLanding ? " is-landing" : ""}`}>
            {isLanding ? (
              <div className="hero">
                <BrandMark size={48} />
                <div className="hero-eyebrow">{PRODUCT.laws}</div>
                <h1 className="hero-title">{PRODUCT.heroTitle}</h1>
                <p className="hero-sub">{PRODUCT.emptySub}</p>
              </div>
            ) : (
              <div className="messages">
                {messages.map((msg, i) => {
                  const noAnswer =
                    msg.role === "assistant" && !msg.streaming && !msg.error && !msg.stopped &&
                    (!msg.sources || msg.sources.length === 0);
                  return (
                  <div key={i} className={`msg-row ${msg.role}`}>
                    {msg.role === "user" && (
                      <button
                        className="edit-btn"
                        onClick={() => fillComposer(msg.content)}
                        title="Sửa rồi hỏi lại"
                        aria-label="Sửa câu hỏi này"
                      >
                        <Icon name="edit" size={14} />
                      </button>
                    )}
                    <div className={`bubble ${noAnswer ? "bubble-noanswer" : ""}`}>
                      {msg.role === "assistant" ? (
                        <>
                          {(msg.streaming || (msg.steps && msg.steps.length > 0)) && (
                            <AgentTrace
                              steps={msg.steps ?? []}
                              running={!!msg.streaming}
                              elapsedMs={elapsedMs}
                              totalMs={msg.latency_ms}
                            />
                          )}
                          {noAnswer && (
                            <div className="noanswer-flag">
                              <Icon name="search" size={14} />
                              <span>Không tìm thấy trong dữ liệu hiện có</span>
                            </div>
                          )}
                          {msg.content ? (
                            <MdText text={msg.content} msgIndex={i} />
                          ) : msg.streaming ? (
                            <div className="skeleton" aria-hidden="true"><span /><span /><span /></div>
                          ) : null}
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
                          {!msg.streaming && (
                          <div className="msg-footer">
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
                              {msg.sources && msg.sources.length > 0 && (
                                <button
                                  className={`icon-btn show-sources${i === panelIdx ? " icon-btn-active" : ""}`}
                                  onClick={() => setPanelMsg(i)}
                                >
                                  {msg.sources.length} nguồn
                                </button>
                              )}
                              {msg.as_of && <span className="latency">Áp dụng tại {fmtDate(msg.as_of)}</span>}
                              {msg.stopped && <span className="latency">Đã dừng</span>}
                              {msg.used_llm && msg.used_llm !== "none" && (
                                <span className="llm-badge">
                                  {msg.used_llm === "gemini" ? "Gemini" : msg.used_llm === "extractive_fallback" ? "Trích xuất trực tiếp" : msg.used_llm}
                                </span>
                              )}
                              {msg.latency_ms && (
                                <span className="latency">{(msg.latency_ms / 1000).toFixed(1)}s</span>
                              )}
                            </div>
                          </div>
                          )}
                        </>
                      ) : (
                        msg.content
                      )}
                    </div>
                  </div>
                  );
                })}

                {/* Đường REST: backend chỉ trả các bước lúc xong hẳn — trong lúc chờ
                    chỉ biết đang ở giai đoạn đầu, không bịa tiến độ các bước sau. */}
                {busy && !messages[messages.length - 1]?.streaming && (
                  <div className="msg-row assistant">
                    <div className="bubble">
                      <AgentTrace steps={[]} running elapsedMs={elapsedMs} />
                      <div className="skeleton" aria-hidden="true"><span /><span /><span /></div>
                    </div>
                  </div>
                )}
                <div ref={bottomRef} />
              </div>
            )}

            <div className="input-bar">
              <div className="composer">
                <textarea
                  ref={inputRef}
                  rows={1}
                  aria-label="Câu hỏi"
                  placeholder={`Hỏi về ${PRODUCT.scope}…`}
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape" && busy) {
                      e.preventDefault();
                      cancel();
                    } else if (e.key === "Enter" && !e.shiftKey) {
                      e.preventDefault();
                      send(question);
                    }
                  }}
                />
                <div className="composer-bar">
                <AsOfChip value={asOf} onChange={setAsOf} />
                {busy ? (
                  <button className="send-btn stop" onClick={cancel} title="Dừng (Esc)">
                    <Icon name="stop" size={12} />
                    Dừng
                  </button>
                ) : (
                  <button
                    className="send-btn"
                    disabled={!question.trim()}
                    onClick={() => send(question)}
                    aria-label="Gửi câu hỏi"
                    title="Gửi (Enter)"
                  >
                    <Icon name="arrowUp" size={18} />
                  </button>
                )}
                </div>
              </div>
              <div className="input-hint">
                <span className="kbd-hint">
                  {busy ? "Esc để dừng — câu hỏi quay lại ô nhập để sửa" : "Enter để gửi · Shift + Enter để xuống dòng"}
                </span>
                <span>{PRODUCT.disclaimer}</span>
              </div>
            </div>

            {isLanding && (
              <div className="hero-extra">
                <div className="sugg-label">Thử hỏi</div>
                <div className="suggest-grid">
                  {SUGGESTED.map((s) => (
                    <button key={s.q} className="suggest-card" onClick={() => fillComposer(s.q)}>
                      <span className="suggest-topic">{s.topic}</span>
                      <span className="suggest-q">{s.q}</span>
                    </button>
                  ))}
                </div>
                <ul className="highlights">
                  {HIGHLIGHTS.map((h) => (
                    <li key={h.text}>
                      <Icon name={h.icon} size={14} />
                      {h.text}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          {panelSources.length > 0 && (
            <aside className="sources-aside" aria-label="Nguồn trích dẫn">
              <div className="aside-head">
                <h2 className="aside-title">Nguồn trích dẫn</h2>
                <span className="aside-count">{panelSources.length} nguồn</span>
              </div>
              {messages[panelIdx - 1]?.role === "user" && (
                <div className="aside-q">{messages[panelIdx - 1].content}</div>
              )}
              <div className="sources-list">
                {panelSources.map((src) => (
                  <SourceCard key={src.index} src={src} msgIndex={panelIdx} idPrefix="panel-" />
                ))}
              </div>
            </aside>
          )}
          </div>
        )}

        {/* Docs */}
        {tab === "docs" && (
          <div className="panel">
            {docs.length === 0 ? (
              <div className="empty">
                <div className="empty-icon"><Icon name="book" size={22} /></div>
                <div className="empty-title">Chưa có văn bản nào</div>
                <div className="empty-sub">Tải lên PDF để bắt đầu</div>
              </div>
            ) : (
              <div className="doc-table" role="table" aria-label="Văn bản đã lập chỉ mục">
                <div className="doc-row doc-head" role="row">
                  <span role="columnheader">Số hiệu</span>
                  <span role="columnheader">Tên văn bản</span>
                  <span role="columnheader">Loại</span>
                  <span role="columnheader">Số đoạn</span>
                  <span role="columnheader">Ban hành</span>
                </div>
                {docs.map((doc) => (
                  <div key={doc.id} className="doc-row" role="row">
                    <span role="cell" className="doc-so">{doc.so_hieu || "—"}</span>
                    <span role="cell" className="doc-title">{doc.title}</span>
                    <span role="cell">{doc.doc_type && <span className="doc-tag">{doc.doc_type}</span>}</span>
                    <span role="cell" className="doc-num">{doc.chunk_count}</span>
                    <span role="cell" className="doc-num">{doc.ngay_ban_hanh || "—"}</span>
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
                <div className="empty-icon"><Icon name="bookmark" size={22} /></div>
                <div className="empty-title">Chưa lưu câu hỏi nào</div>
                <div className="empty-sub">
                  Bấm "Lưu" dưới một câu trả lời để xem lại sau
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
                    <div className="answer-actions">
                      <button
                        className="icon-btn"
                        onClick={() => setBookmarks((prev) => prev.filter((x) => x.id !== b.id))}
                      >
                        <Icon name="trash" size={14} />
                        Bỏ lưu
                      </button>
                    </div>
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
              <div className="empty-icon"><Icon name="file" size={22} /></div>
              <div className="upload-text">
                {busy ? "Đang xử lý…" : "Kéo thả PDF hoặc bấm để chọn"}
              </div>
              <div className="upload-sub">Hỗ trợ: Luật, Nghị định, Thông tư, Quyết định — tối đa 20MB</div>
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
