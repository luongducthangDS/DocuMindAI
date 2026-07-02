"""
scripts/expand_corpus.py — Crawl thêm văn bản pháp luật từ vbpl.vn

Mục đích: mở rộng corpus từ 356 chunks / 18 docs hiện tại lên ~2000+ chunks
để cải thiện context_recall (hiện 0.7683).

QUAN TRỌNG: Script này KHÔNG chạy crawl thực tế — chỉ in hướng dẫn và validate
config. Để chạy thật, truyền --execute flag. Mọi request đều có rate limiting
và retry logic để tránh bị ban IP và tôn trọng server vbpl.vn.

Cách dùng:
  # Dry-run: xem sẽ crawl gì (không gửi request)
  python scripts/expand_corpus.py --dry-run

  # Crawl thực tế (cần có ChromaDB running)
  python scripts/expand_corpus.py --execute --categories company_law labor_law tax_law

  # Crawl một category với limit
  python scripts/expand_corpus.py --execute --categories labor_law --limit 20

Requirements:
  pip install httpx beautifulsoup4 tenacity tqdm
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ── Cấu hình crawl ──────────────────────────────────────────────────────────

BASE_URL = "https://vbpl.vn"

# Endpoint tìm kiếm văn bản theo lĩnh vực
SEARCH_ENDPOINT = "/tw/pages/vbpq-timkiem.aspx"

# Lĩnh vực pháp luật → tham số tìm kiếm vbpl.vn
CATEGORY_PARAMS: dict[str, dict] = {
    "company_law": {
        "display_name": "Luật Doanh nghiệp",
        "keywords": ["luật doanh nghiệp", "công ty cổ phần", "công ty TNHH", "doanh nghiệp tư nhân"],
        "doc_type": "1",   # Luật
    },
    "labor_law": {
        "display_name": "Luật Lao động",
        "keywords": ["bộ luật lao động", "hợp đồng lao động", "tiền lương", "bảo hiểm xã hội"],
        "doc_type": "1",
    },
    "tax_law": {
        "display_name": "Luật Thuế",
        "keywords": ["thuế thu nhập doanh nghiệp", "thuế giá trị gia tăng", "thuế thu nhập cá nhân"],
        "doc_type": "1",
    },
    "civil_law": {
        "display_name": "Bộ luật Dân sự",
        "keywords": ["bộ luật dân sự", "hợp đồng dân sự", "tài sản", "quyền sở hữu"],
        "doc_type": "1",
    },
    "investment_law": {
        "display_name": "Luật Đầu tư",
        "keywords": ["luật đầu tư", "vốn đầu tư nước ngoài", "ưu đãi đầu tư"],
        "doc_type": "1",
    },
}

# Rate limiting — tôn trọng server
REQUEST_DELAY_SECONDS = (2.0, 4.0)   # random delay giữa các request
MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0              # exponential backoff: 2s, 4s, 8s
REQUEST_TIMEOUT_SECONDS = 30
MAX_DOCS_PER_CATEGORY = 50           # hard cap để tránh overload

# User-Agent rotation — tránh bị phát hiện là bot
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36",
]


# ── Data models ──────────────────────────────────────────────────────────────

@dataclass
class LegalDocument:
    """Văn bản pháp luật đã crawl."""
    url: str
    title: str
    so_hieu: str          # số hiệu văn bản (VD: "59/2020/QH14")
    doc_type: str         # loại văn bản (Luật, Nghị định, Thông tư...)
    category: str         # lĩnh vực (company_law, labor_law...)
    content: str          # nội dung full text đã làm sạch
    effective_date: str = ""
    issuing_body: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class CrawlProgress:
    """Track tiến độ crawl để resume nếu bị gián đoạn."""
    total_discovered: int = 0
    total_crawled: int = 0
    total_chunks_added: int = 0
    total_errors: int = 0
    skipped_duplicates: int = 0
    categories_done: list[str] = field(default_factory=list)


# ── HTTP Client với retry ──────────────────────────────────────────────────

def _get_http_client():
    """Tạo httpx client với headers hợp lý."""
    try:
        import httpx
    except ImportError:
        raise ImportError("httpx not installed. Run: pip install httpx")

    return httpx.Client(
        headers={
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
    )


def _fetch_with_retry(client, url: str, attempt: int = 1) -> str | None:
    """
    Fetch URL với retry logic và exponential backoff.
    Trả về HTML string hoặc None nếu fail sau MAX_RETRIES lần.
    """
    try:
        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )
        import httpx
    except ImportError:
        raise ImportError("tenacity not installed. Run: pip install tenacity")

    # Random delay để không bị rate limit
    delay = random.uniform(*REQUEST_DELAY_SECONDS)
    time.sleep(delay)

    # Rotate User-Agent
    client.headers["User-Agent"] = random.choice(USER_AGENTS)

    for attempt_num in range(1, MAX_RETRIES + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as exc:
            wait = RETRY_BACKOFF_BASE ** attempt_num
            print(f"  [Retry {attempt_num}/{MAX_RETRIES}] {url}: {exc} — waiting {wait:.0f}s")
            if attempt_num < MAX_RETRIES:
                time.sleep(wait)
            else:
                print(f"  [FAIL] Exhausted retries for {url}")
                return None


# ── Parser ────────────────────────────────────────────────────────────────────

def _parse_search_results(html: str) -> list[dict]:
    """
    Parse trang kết quả tìm kiếm vbpl.vn.
    Trả về list of {url, title, so_hieu, doc_type}.
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed. Run: pip install beautifulsoup4")

    soup = BeautifulSoup(html, "html.parser")
    results = []

    # vbpl.vn listing format: <div class="vbTitle"> hoặc <td class="title">
    for link in soup.select(".vbTitle a, .title a, .van-ban-title a"):
        href = link.get("href", "")
        if not href:
            continue
        full_url = href if href.startswith("http") else BASE_URL + href
        results.append({
            "url": full_url,
            "title": link.get_text(strip=True),
            "so_hieu": _extract_so_hieu(link.get_text(strip=True)),
            "doc_type": "Luật",
        })

    return results


def _extract_so_hieu(title: str) -> str:
    """Trích số hiệu văn bản từ title. VD: '59/2020/QH14'."""
    import re
    match = re.search(r"\d+/\d{4}/[A-Z0-9-]+", title)
    return match.group(0) if match else ""


def _parse_document_content(html: str, url: str) -> str:
    """
    Parse nội dung full text của văn bản pháp luật.
    Làm sạch HTML, giữ lại text có cấu trúc (điều, khoản, điểm).
    """
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("beautifulsoup4 not installed.")

    soup = BeautifulSoup(html, "html.parser")

    # Tìm container nội dung chính — vbpl.vn dùng nhiều class khác nhau
    content_div = (
        soup.select_one("#toanvan")         # nội dung toàn văn
        or soup.select_one(".toanVanDiv")
        or soup.select_one(".content-detail")
        or soup.select_one("article")
        or soup.select_one(".main-content")
    )

    if not content_div:
        # Fallback: lấy body text và lọc navigation/footer
        for tag in soup(["nav", "header", "footer", "script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    # Remove noise tags
    for tag in content_div(["script", "style", "nav", "button", "img"]):
        tag.decompose()

    text = content_div.get_text(separator="\n", strip=True)

    # Chuẩn hoá whitespace
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)

    return text.strip()


def _parse_metadata(html: str) -> dict:
    """Parse metadata từ trang văn bản: ngày có hiệu lực, cơ quan ban hành."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return {}

    soup = BeautifulSoup(html, "html.parser")
    metadata = {}

    # vbpl.vn thường có bảng metadata ở sidebar
    for row in soup.select(".vbInfo tr, .van-ban-info tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            key = cells[0].get_text(strip=True).lower()
            val = cells[1].get_text(strip=True)
            if "hiệu lực" in key:
                metadata["effective_date"] = val
            elif "ban hành" in key or "cơ quan" in key:
                metadata["issuing_body"] = val

    return metadata


# ── Chunker ──────────────────────────────────────────────────────────────────

def _chunk_document(doc: LegalDocument, chunk_size: int = 512, overlap: int = 64) -> list[dict]:
    """
    Chia văn bản thành chunks theo ranh giới "Điều".
    Ưu tiên split tại header "Điều X." để giữ nguyên cấu trúc pháp luật.
    Fallback: split theo token count.
    """
    import re

    # Split tại ranh giới "Điều" — mỗi điều luật là một unit ngữ nghĩa độc lập
    dieu_pattern = re.compile(r"(?=Điều\s+\d+[\.:])", re.IGNORECASE)
    parts = dieu_pattern.split(doc.content)
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        # Fallback: word-based chunking
        words = doc.content.split()
        parts = [
            " ".join(words[i:i + chunk_size])
            for i in range(0, len(words), chunk_size - overlap)
        ]

    chunks = []
    for i, part in enumerate(parts):
        if len(part) < 30:   # skip quá ngắn (headers rỗng)
            continue

        # Lấy dieu_header từ dòng đầu của part
        first_line = part.split("\n")[0][:120]

        chunk = {
            "text": part,
            "metadata": {
                "source_url": doc.url,
                "title": doc.title,
                "so_hieu": doc.so_hieu,
                "doc_type": doc.doc_type,
                "category": doc.category,
                "effective_date": doc.effective_date,
                "issuing_body": doc.issuing_body,
                "dieu_header": first_line,
                "chunk_index": i,
                **doc.metadata,
            },
        }
        chunks.append(chunk)

    return chunks


# ── ChromaDB ingestion ────────────────────────────────────────────────────────

def _ingest_chunks(chunks: list[dict], dry_run: bool = False) -> int:
    """
    Embed chunks và upsert vào ChromaDB.
    Dùng cùng embedder + collection như production để đảm bảo compatibility.
    Trả về số chunks đã thêm.
    """
    if dry_run:
        print(f"  [DRY-RUN] Would ingest {len(chunks)} chunks")
        return len(chunks)

    if not chunks:
        return 0

    import src.logger  # noqa: F401
    from src.rag.embedder import get_chroma_collection, get_embedder

    embedder = get_embedder()
    _, collection = get_chroma_collection()

    added = 0
    batch_size = 50  # ChromaDB upsert batch

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start:batch_start + batch_size]
        texts = [c["text"] for c in batch]
        metas = [c["metadata"] for c in batch]

        # Generate embeddings
        embeddings = [embedder.get_text_embedding(t) for t in texts]

        # Build stable IDs từ URL + chunk index để tránh duplicate
        ids = [
            f"{c['metadata']['source_url']}__chunk_{c['metadata']['chunk_index']}"
            for c in batch
        ]

        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metas,
        )
        added += len(batch)
        print(f"  Upserted batch of {len(batch)} chunks (total: {added})")

    return added


# ── Discovery: tìm URLs văn bản ──────────────────────────────────────────────

def discover_document_urls(
    client,
    category: str,
    limit: int = MAX_DOCS_PER_CATEGORY,
    dry_run: bool = False,
) -> list[dict]:
    """
    Tìm danh sách URL văn bản pháp luật từ vbpl.vn cho một category.
    Phân trang qua nhiều trang kết quả tìm kiếm.
    """
    if category not in CATEGORY_PARAMS:
        print(f"Unknown category: {category}. Available: {list(CATEGORY_PARAMS.keys())}")
        return []

    params = CATEGORY_PARAMS[category]
    print(f"\nDiscovering {params['display_name']} documents (limit={limit})...")

    all_docs = []

    for keyword in params["keywords"]:
        if len(all_docs) >= limit:
            break

        search_url = (
            f"{BASE_URL}{SEARCH_ENDPOINT}"
            f"?type={params['doc_type']}&s={keyword.replace(' ', '+')}"
        )

        if dry_run:
            print(f"  [DRY-RUN] Would search: {search_url}")
            # Return dummy data for dry-run validation
            all_docs.append({
                "url": f"https://vbpl.vn/example/{category}/1.aspx",
                "title": f"[DRY-RUN] {keyword}",
                "so_hieu": "XX/2024/QH15",
                "doc_type": "Luật",
            })
            continue

        html = _fetch_with_retry(client, search_url)
        if not html:
            print(f"  Failed to fetch search results for: {keyword}")
            continue

        results = _parse_search_results(html)
        print(f"  Found {len(results)} docs for keyword: '{keyword}'")
        all_docs.extend(results)

        if len(all_docs) >= limit:
            all_docs = all_docs[:limit]
            break

    # Deduplicate by URL
    seen = set()
    unique_docs = []
    for doc in all_docs:
        if doc["url"] not in seen:
            seen.add(doc["url"])
            unique_docs.append(doc)

    print(f"  Discovered {len(unique_docs)} unique documents for {category}")
    return unique_docs


# ── Main crawl loop ──────────────────────────────────────────────────────────

def crawl_category(
    client,
    category: str,
    limit: int,
    progress: CrawlProgress,
    dry_run: bool = False,
    progress_file: Path | None = None,
) -> int:
    """Crawl một lĩnh vực pháp luật. Trả về số chunks đã thêm vào ChromaDB."""

    doc_infos = discover_document_urls(client, category, limit=limit, dry_run=dry_run)
    progress.total_discovered += len(doc_infos)

    chunks_added = 0

    for i, doc_info in enumerate(doc_infos, 1):
        url = doc_info["url"]
        print(f"\n  [{i}/{len(doc_infos)}] Crawling: {url}")

        if dry_run:
            print(f"  [DRY-RUN] Would crawl and chunk: {doc_info['title']}")
            progress.total_crawled += 1
            chunks_added += 5  # estimate
            continue

        html = _fetch_with_retry(client, url)
        if not html:
            progress.total_errors += 1
            continue

        content = _parse_document_content(html, url)
        if len(content) < 100:
            print(f"  Skipping: content too short ({len(content)} chars)")
            progress.total_errors += 1
            continue

        meta = _parse_metadata(html)
        doc = LegalDocument(
            url=url,
            title=doc_info["title"],
            so_hieu=doc_info.get("so_hieu", ""),
            doc_type=doc_info.get("doc_type", "Luật"),
            category=category,
            content=content,
            effective_date=meta.get("effective_date", ""),
            issuing_body=meta.get("issuing_body", ""),
            metadata=meta,
        )

        chunks = _chunk_document(doc)
        added = _ingest_chunks(chunks, dry_run=dry_run)

        progress.total_crawled += 1
        progress.total_chunks_added += added
        chunks_added += added

        print(f"  OK: {len(chunks)} chunks → ChromaDB (title: {doc.title[:60]})")

        # Save progress periodically (every 5 docs) for resume capability
        if progress_file and i % 5 == 0:
            _save_progress(progress, progress_file)

    progress.categories_done.append(category)
    return chunks_added


def _save_progress(progress: CrawlProgress, path: Path) -> None:
    """Ghi tiến độ ra file JSON để resume nếu bị gián đoạn."""
    path.write_text(
        json.dumps(progress.__dict__, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_progress(path: Path) -> CrawlProgress:
    """Load tiến độ từ file JSON."""
    if not path.exists():
        return CrawlProgress()
    data = json.loads(path.read_text(encoding="utf-8"))
    p = CrawlProgress()
    for k, v in data.items():
        setattr(p, k, v)
    return p


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl văn bản pháp luật từ vbpl.vn và ingest vào ChromaDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ:
  # Xem sẽ crawl gì (không gửi request, không thay đổi DB)
  python scripts/expand_corpus.py --dry-run

  # Crawl thực tế (cần ChromaDB running)
  python scripts/expand_corpus.py --execute

  # Chỉ crawl một số lĩnh vực
  python scripts/expand_corpus.py --execute --categories labor_law tax_law

  # Giới hạn số văn bản mỗi lĩnh vực
  python scripts/expand_corpus.py --execute --categories company_law --limit 10

  # Resume từ lần chạy trước
  python scripts/expand_corpus.py --execute --resume
        """,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In kế hoạch crawl mà không gửi request hoặc thay đổi DB",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Thực sự crawl và ingest vào ChromaDB",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(CATEGORY_PARAMS.keys()),
        default=list(CATEGORY_PARAMS.keys()),
        help=f"Lĩnh vực cần crawl (default: tất cả)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=30,
        help=f"Số văn bản tối đa mỗi lĩnh vực (default: 30, max: {MAX_DOCS_PER_CATEGORY})",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume từ tiến độ đã lưu ở lần chạy trước",
    )
    parser.add_argument(
        "--progress-file",
        type=Path,
        default=Path("data/crawl_progress.json"),
        help="File lưu tiến độ crawl (default: data/crawl_progress.json)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        parser.error("Phải truyền --dry-run hoặc --execute")

    if args.limit > MAX_DOCS_PER_CATEGORY:
        print(f"Warning: limit {args.limit} > max {MAX_DOCS_PER_CATEGORY}, capping.")
        args.limit = MAX_DOCS_PER_CATEGORY

    dry_run = not args.execute

    # Load or init progress
    progress = _load_progress(args.progress_file) if args.resume else CrawlProgress()
    if args.resume and progress.categories_done:
        print(f"Resuming — already done: {progress.categories_done}")
        args.categories = [c for c in args.categories if c not in progress.categories_done]
        if not args.categories:
            print("All categories already completed.")
            return

    print("=" * 60)
    print("DocuMind AI — Corpus Expansion")
    print(f"Mode: {'DRY-RUN' if dry_run else 'EXECUTE (LIVE)'}")
    print(f"Categories: {args.categories}")
    print(f"Limit per category: {args.limit}")
    print("=" * 60)

    if dry_run:
        print("\nDRY-RUN MODE: No HTTP requests will be made. No DB writes.")
        print("Run with --execute to actually crawl.\n")

    client = _get_http_client()
    total_chunks = 0

    try:
        for category in args.categories:
            added = crawl_category(
                client=client,
                category=category,
                limit=args.limit,
                progress=progress,
                dry_run=dry_run,
                progress_file=args.progress_file if not dry_run else None,
            )
            total_chunks += added
            print(f"\n[{category}] Done: {added} chunks added")

    except KeyboardInterrupt:
        print("\nInterrupted by user. Progress saved.")
        if not dry_run:
            _save_progress(progress, args.progress_file)

    finally:
        client.close()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print(f"  Documents discovered : {progress.total_discovered}")
    print(f"  Documents crawled    : {progress.total_crawled}")
    print(f"  Chunks added to DB   : {progress.total_chunks_added}")
    print(f"  Errors               : {progress.total_errors}")
    print(f"  Duplicate skips      : {progress.skipped_duplicates}")
    print("=" * 60)

    if dry_run:
        print("\nRun with --execute to perform the actual crawl.")
    else:
        print("\nDone. Re-run benchmark to measure impact:")
        print("  python eval/rag_comparison.py --limit 20 --output reports/post_expansion.json")


if __name__ == "__main__":
    main()
