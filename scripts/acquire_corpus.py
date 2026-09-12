"""
scripts/acquire_corpus.py — tái lập / kiểm tra corpus toàn văn trong
`data/raw/lao_dong/` (T5 của pivot lao động – BHXH).

Corpus đã được thu thập ở commit T5; script này là công cụ tái lập để sau này
còn kiểm tra lại được nguồn, hoặc bổ sung văn bản mới vào `locked_list_v1`
mà không phải làm tay.

Nguồn cho mỗi văn bản lấy theo thứ tự:
  1. frontmatter của chính file đang có trong `data/raw/lao_dong/` (trường
     `nguon` / `nguon_file`) — đây là nguồn đã dùng thật, đã đối chiếu;
  2. trường `nguon` trong `docs/corpus/corpus_manifest.yaml`.

Quan hệ với `src/ingestion/congbao_crawler.py`: crawler đó duyệt Công báo theo
số phát hành để mở rộng corpus (Playwright, DOCX). Script này đi hướng ngược
lại — danh sách văn bản đã CHỐT trong manifest, cần tải đúng từng văn bản đó và
kiểm tra lại nguồn — nên chỉ dùng HTTP thuần, không cần Playwright.

Phụ thuộc: pdfplumber + beautifulsoup4 (đã khai báo trong requirements.txt),
PyYAML (đã có sẵn trong môi trường, chưa khai báo). PyMuPDF là tuỳ chọn, chỉ để
chạy nhanh hơn.

Cách trích text:
  - PDF Công báo (congbao.chinhphu.vn / congbaocdn) có text layer -> PyMuPDF.
  - Trang toàn văn của Cổng TTĐT Chính phủ (xaydungchinhsach.chinhphu.vn) ->
    lấy `div.detail-content`.
  - Bản .signed.pdf trên datafiles.chinhphu.vn là bản SCAN, không có text
    layer: script từ chối thay vì đoán bằng OCR.
  - File `cach_lay` ghi "mirror thứ cấp" là những văn bản không có nguồn chính
    thức trích được text (đã đối chiếu tay). Script KHÔNG tự ghi đè nhóm này.

Chạy:
    python scripts/acquire_corpus.py                 # kiểm tra, không ghi gì
    python scripts/acquire_corpus.py --write         # tải các file còn thiếu
    python scripts/acquire_corpus.py --force --only 293-2025-ND-CP
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
import unicodedata
import urllib.request
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs" / "corpus" / "corpus_manifest.yaml"
OUT_DIR = ROOT / "data" / "raw" / "lao_dong"

UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept-Encoding": "gzip"}
TIMEOUT = 60

PDF_LINK_RE = re.compile(r"https://congbaocdn\.chinhphu\.vn/[^\"'\s<>]+")
VBPQ_LINK_RE = re.compile(r"https://datafiles\.chinhphu\.vn/cpp/files/vbpq/[^\"'\s<>]+?\.pdf")
DIEU_RE = re.compile(r"^\s*Điều\s+(\d+)\s*[.．]", re.M)

# Thứ tự trường theo đúng corpus đã commit, để file sinh lại không lệch định dạng.
FM_ORDER = (
    "doc_id", "so_hieu", "ten", "loai", "ngay_ban_hanh", "ngay_hieu_luc",
    "trang_thai", "nguon", "nguon_file", "cach_lay", "ngay_lay", "doi_chieu",
)


# ---------------------------------------------------------------- fetching
def http_get(url: str, retries: int = 4) -> bytes:
    """Tải một URL. CDN Công báo hay đứt giữa chừng (IncompleteRead) và trả 502
    lẻ tẻ, nên đọc theo chunk và thử lại vài lần."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                chunks = []
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    chunks.append(chunk)
            raw = b"".join(chunks)
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            return raw
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2 * (attempt + 1))
    raise last


def resolve_source(url: str):
    """Trả (url_tai, kind, cách_resolve); kind = 'pdf' | 'html'."""
    if not url:
        return None, None, "no-source"

    if "xaydungchinhsach.chinhphu.vn" in url or "baochinhphu.vn" in url:
        return url, "html", "chinhphu-html"

    if url.lower().endswith(".pdf") or "format=pdf" in url.lower():
        return url, "pdf", "direct-pdf"

    if "congbao.chinhphu.vn" in url:
        try:
            html = http_get(url).decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            return None, None, "congbao-fetch-error: {}".format(exc)
        pdfs = [l for l in PDF_LINK_RE.findall(html) if "pdf" in l.lower()]
        if pdfs:
            return pdfs[0], "pdf", "congbao-html"
        return None, None, "congbao-no-pdf-link"

    if "vanban.chinhphu.vn" in url or "chinhphu.vn/?pageid=" in url:
        try:
            html = http_get(url).decode("utf-8", "ignore")
        except Exception as exc:  # noqa: BLE001
            return None, None, "vanban-fetch-error: {}".format(exc)
        pdfs = VBPQ_LINK_RE.findall(html)
        if pdfs:
            return pdfs[0], "pdf", "vanban-html"
        return None, None, "vanban-no-pdf-link"

    return None, None, "unsupported-source"


# ---------------------------------------------------------------- text
def pdf_to_text(data: bytes):
    """Trích text PDF. Mặc định dùng pdfplumber (đã có trong requirements, cùng
    thư viện `src/ingestion/congbao_crawler.py` đang dùng); nếu máy có PyMuPDF
    thì dùng nó vì nhanh hơn nhiều trên văn bản 90+ trang."""
    try:
        import fitz  # PyMuPDF — tuỳ chọn, không bắt buộc

        doc = fitz.open(stream=data, filetype="pdf")
        pages = [doc[i].get_text("text") for i in range(doc.page_count)]
        n = doc.page_count
        doc.close()
        return "\n".join(pages), n
    except ImportError:
        import io as _io

        import pdfplumber

        with pdfplumber.open(_io.BytesIO(data)) as pdf:
            pages = [p.extract_text(x_tolerance=1, y_tolerance=3) or "" for p in pdf.pages]
        return "\n".join(pages), len(pages)


def html_to_text(data: bytes) -> str:
    """Trích thân bài toàn văn trên Cổng TTĐT Chính phủ."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(data.decode("utf-8", "ignore"), "html.parser")
    node = soup.select_one("div.detail-content") or soup.select_one("div.afcbc-body")
    return node.get_text("\n", strip=True) if node else ""


def clean_text(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.replace("\xa0", " ").rstrip()
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):          # số trang đứng một mình
            continue
        if re.match(r"^CÔNG BÁO/Số \d+", stripped):     # header lặp mỗi trang
            continue
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip() + "\n"


def has_text_layer(text: str, pages: int) -> bool:
    return len(text.strip()) >= max(400, 80 * pages)


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", s)).upper()


def so_hieu_matches(text: str, so_hieu: str, head_chars: int = 20000) -> bool:
    """Chống lấy nhầm văn bản trùng số: số hiệu phải có ở phần đầu văn bản.

    Cần thiết vì congbao.chinhphu.vn định tuyến theo id và bỏ qua phần slug,
    nên một URL sai id vẫn trả về 200 với một văn bản hoàn toàn khác.
    """
    if not so_hieu:
        return True
    return _squash(so_hieu) in _squash(text[:head_chars])


def count_dieu(text: str):
    nums = [int(n) for n in DIEU_RE.findall(text)]
    return len(set(nums)), (max(nums) if nums else 0)


# ---------------------------------------------------------------- file i/o
def read_frontmatter(path: Path) -> dict:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    try:
        return yaml.safe_load(text[3:end]) or {}
    except yaml.YAMLError:
        return {}


def read_body(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].lstrip("\n")
    return text


def dump_frontmatter(fm: dict) -> str:
    out = []
    for key in FM_ORDER:
        if key in fm and fm[key] != "":
            out.append('{}: "{}"'.format(key, str(fm[key]).replace('"', "'")))
        elif key in ("nguon_file",) and key in fm:
            out.append('{}: ""'.format(key))
    return "\n".join(out)


def write_markdown(doc: dict, body: str, page_url: str, file_url: str, kind: str, old_fm: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if kind == "pdf":
        cach_lay = "pdf_text_layer (PyMuPDF)"
    else:
        host = re.sub(r"^https?://([^/]+).*", r"\1", page_url)
        cach_lay = "html_fulltext ({} — toàn văn chính thức)".format(host)

    fm = {
        "doc_id": doc["doc_id"],
        "so_hieu": doc.get("so_hieu", ""),
        "ten": doc.get("ten", ""),
        "loai": doc.get("loai", ""),
        "ngay_ban_hanh": str(doc.get("ngay_ban_hanh", "")),
        "ngay_hieu_luc": str(doc.get("ngay_hieu_luc", "")),
        "trang_thai": doc.get("trang_thai", ""),
        "nguon": old_fm.get("nguon") or page_url,
        "nguon_file": old_fm.get("nguon_file", file_url if file_url != page_url else ""),
        "cach_lay": old_fm.get("cach_lay") or cach_lay,
        "ngay_lay": date.today().isoformat(),
    }
    # Ghi chú đối chiếu thủ công là công sức xác minh — không để mất khi sinh lại.
    if old_fm.get("doi_chieu"):
        fm["doi_chieu"] = old_fm["doi_chieu"]

    path = OUT_DIR / "{}.md".format(doc["doc_id"])
    path.write_text("---\n{}\n---\n\n{}".format(dump_frontmatter(fm), body), encoding="utf-8")
    return path


# ---------------------------------------------------------------- main
def load_targets():
    man = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    locked = man["locked_list_v1"]
    ids = list(locked["con_hieu_luc"]) + list(locked["het_hieu_luc_giu_cho_point_in_time"])
    docs = {d["doc_id"]: d for d in man["documents"]}
    missing = [i for i in ids if i not in docs]
    if missing:
        raise SystemExit("locked_list_v1 có doc_id không có trong documents: {}".format(missing))
    return [docs[i] for i in ids]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="chỉ xử lý các doc_id này")
    ap.add_argument("--write", action="store_true", help="tải các file còn thiếu")
    ap.add_argument("--force", action="store_true", help="tải lại và ghi đè file đã có")
    args = ap.parse_args()

    targets = load_targets()
    if args.only:
        targets = [d for d in targets if d["doc_id"] in args.only]

    ok = diff = skipped = 0
    failed = []
    for doc in targets:
        did = doc["doc_id"]
        path = OUT_DIR / "{}.md".format(did)
        old_fm = read_frontmatter(path)

        if path.exists() and "mirror" in str(old_fm.get("cach_lay", "")) and not args.force:
            print("[skip] {}: nguồn mirror thứ cấp, đã đối chiếu tay — script không tái lập".format(did))
            skipped += 1
            continue
        if path.exists() and not (args.force or args.write) and not old_fm:
            print("[skip] {}: file không có frontmatter".format(did))
            skipped += 1
            continue

        page_url = old_fm.get("nguon") or doc.get("nguon")
        # File đã lấy từ trang toàn văn HTML nhưng frontmatter chỉ ghi trang
        # tra cứu (bản .signed.pdf là scan): ưu tiên nguồn còn trích được text.
        if "html_fulltext" in str(old_fm.get("cach_lay", "")) and "chinhphu.vn/?pageid" in str(page_url):
            page_url = str(old_fm.get("nguon_file") or page_url)
        url, kind, how = resolve_source(page_url)
        if not url:
            print("[FAIL] {}: {} ({})".format(did, how, page_url))
            failed.append((did, how))
            continue

        try:
            data = http_get(url)
        except Exception as exc:  # noqa: BLE001
            print("[FAIL] {}: download error {}".format(did, exc))
            failed.append((did, "download: {}".format(exc)))
            continue

        # Endpoint "…?format=pdf" của Công báo đôi khi trả trang xem trước thay
        # vì file: lấy link PDF thật trong đó rồi tải lại.
        if kind == "pdf" and not data[:5].startswith(b"%PDF"):
            pdfs = [l for l in PDF_LINK_RE.findall(data.decode("utf-8", "ignore")) if "pdf" in l.lower()]
            if pdfs:
                try:
                    data = http_get(pdfs[0])
                    url, how = pdfs[0], how + "+viewer-fallback"
                except Exception as exc:  # noqa: BLE001
                    print("[FAIL] {}: viewer fallback lỗi {}".format(did, exc))
                    failed.append((did, "viewer-fallback"))
                    continue

        if kind == "html":
            text, pages = html_to_text(data), 0
            if len(text) < 2000:
                print("[FAIL] {}: trang HTML không có phần toàn văn ({})".format(did, url))
                failed.append((did, "html-no-fulltext"))
                continue
        else:
            if not data[:5].startswith(b"%PDF"):
                print("[FAIL] {}: nguồn không trả PDF ({})".format(did, url))
                failed.append((did, "not-pdf"))
                continue
            text, pages = pdf_to_text(data)
            if not has_text_layer(text, pages):
                print("[FAIL] {}: PDF là bản scan (không có text layer) — cần nguồn khác".format(did))
                failed.append((did, "scanned-pdf"))
                continue

        if not so_hieu_matches(text, doc.get("so_hieu", "")):
            print("[FAIL] {}: nội dung không chứa số hiệu {} — nguồn sai văn bản".format(did, doc.get("so_hieu")))
            failed.append((did, "so-hieu-mismatch"))
            continue

        body = clean_text(text)
        n_dieu, max_dieu = count_dieu(body)

        if path.exists() and not args.force:
            cur = read_body(path)
            c_dieu, c_max = count_dieu(cur)
            if (c_dieu, c_max) == (n_dieu, max_dieu):
                print("[ok  ] {}: khớp file đang có ({} điều, tới Điều {})".format(did, n_dieu, max_dieu))
                ok += 1
            else:
                print("[DIFF] {}: file có {} điều (tới {}), nguồn cho {} điều (tới {})".format(
                    did, c_dieu, c_max, n_dieu, max_dieu))
                diff += 1
            continue

        if not (args.write or args.force):
            print("[miss] {}: chưa có file (chạy --write để tải)".format(did))
            skipped += 1
            continue

        out = write_markdown(doc, body, page_url, url, kind, old_fm)
        print("[ok  ] {}: {} điều, {:,} ký tự -> {}".format(did, n_dieu, len(body), out.relative_to(ROOT)))
        ok += 1

    print("\nTổng: ok={} diff={} skip={} fail={}".format(ok, diff, skipped, len(failed)))
    for did, why in failed:
        print("  - {}: {}".format(did, why))
    return 1 if (failed or diff) else 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.exit(main())
