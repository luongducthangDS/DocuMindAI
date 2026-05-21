"""
Crawler for vbpl.vn — Cơ sở dữ liệu quốc gia văn bản pháp luật (Bộ Tư pháp).
Data is 100% public. Crawl rate is deliberately limited to 2s/request.
"""

import json
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.ingestion.cleaner import clean_legal_text

BASE_URL = "https://vbpl.vn"
_HEADERS = {
    "User-Agent": "DocuMind-Research-Bot/1.0 (academic, non-commercial; contact: luongducthang289@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "vi,en;q=0.9",
}
_REQUEST_DELAY = 2.0  # seconds — polite crawling

_DOC_TYPES = {
    "luat": "Luật",
    "nghi-dinh": "Nghị định",
    "thong-tu": "Thông tư",
    "quyet-dinh": "Quyết định",
}


class VbplCrawler:
    def __init__(self, output_dir: Path | None = None):
        settings = get_settings()
        self._output = output_dir or settings.data_dir / "raw"
        self._output.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update(_HEADERS)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
    def _get(self, url: str, timeout: int = 15) -> requests.Response:
        resp = self._session.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp

    def _safe_filename(self, title: str) -> str:
        """Sanitize title to safe filename — prevent path traversal."""
        safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in title)
        return safe[:80].strip()

    def get_doc_links(self, doc_type: str = "luat", page: int = 1) -> list[str]:
        url = f"{BASE_URL}/TW/Pages/vbpq-toanvan.aspx?type={doc_type}&page={page}"
        try:
            resp = self._get(url)
        except Exception as exc:
            logger.error("Failed to fetch listing page {}: {}", url, exc)
            return []

        soup = BeautifulSoup(resp.text, "lxml")
        links = []
        for a in soup.select("div.title a, h3.title a, div.vbTitle a"):
            href = a.get("href", "")
            if not href:
                continue
            # Validate it's a relative path to vbpl.vn — prevent SSRF
            parsed = urlparse(href)
            if parsed.scheme and parsed.netloc and parsed.netloc not in ("vbpl.vn", "www.vbpl.vn"):
                logger.warning("Skipping external link: {}", href)
                continue
            full = urljoin(BASE_URL, href)
            links.append(full)

        return links

    def fetch_document(self, url: str) -> dict | None:
        # Validate URL is on vbpl.vn before fetching
        parsed = urlparse(url)
        if parsed.netloc not in ("vbpl.vn", "www.vbpl.vn"):
            logger.error("Refused to fetch non-vbpl.vn URL: {}", url)
            return None

        try:
            resp = self._get(url)
        except Exception as exc:
            logger.error("Failed to fetch {}: {}", url, exc)
            return None

        soup = BeautifulSoup(resp.text, "lxml")

        title_el = (
            soup.select_one("h1.doc-title")
            or soup.select_one("h1.title")
            or soup.select_one("div.toanvan h1")
        )
        content_el = (
            soup.select_one("div.toanvan")
            or soup.select_one("div.content-full")
        )

        if not content_el:
            logger.warning("No content found at {}", url)
            return None

        title = title_el.get_text(strip=True) if title_el else "unknown"
        raw_content = content_el.get_text(separator="\n", strip=True)
        cleaned = clean_legal_text(raw_content, doc_title=title)

        if len(cleaned) < 200:
            logger.debug("Skipping thin document: {}", title[:50])
            return None

        # Extract metadata from page
        so_hieu = ""
        ngay_ban_hanh = ""
        for meta_row in soup.select("table.vbInfo tr, div.docInfo tr"):
            cells = meta_row.find_all(["td", "th"])
            if len(cells) >= 2:
                label = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                if "số hiệu" in label:
                    so_hieu = value
                elif "ngày ban hành" in label or "ngày ký" in label:
                    ngay_ban_hanh = value

        return {
            "title": title,
            "content": cleaned,
            "url": url,
            "source": "vbpl.vn",
            "doc_type": "legal",
            "so_hieu": so_hieu,
            "ngay_ban_hanh": ngay_ban_hanh,
        }

    def crawl(
        self,
        doc_types: list[str] | None = None,
        max_pages: int = 5,
        max_docs: int = 100,
    ) -> int:
        """
        Crawl vbpl.vn. Returns count of saved documents.
        max_docs is a hard safety limit to prevent runaway crawls.
        """
        if doc_types is None:
            doc_types = list(_DOC_TYPES.keys())

        saved = 0
        for dtype in doc_types:
            if saved >= max_docs:
                break
            for page in range(1, max_pages + 1):
                if saved >= max_docs:
                    break

                links = self.get_doc_links(doc_type=dtype, page=page)
                if not links:
                    logger.info("No links on page {} for type {}, stopping", page, dtype)
                    break

                for url in links:
                    if saved >= max_docs:
                        break

                    doc = self.fetch_document(url)
                    if doc:
                        self._save(doc)
                        saved += 1

                    time.sleep(_REQUEST_DELAY)

        logger.info("Crawl complete: {} documents saved to {}", saved, self._output)
        return saved

    def _save(self, doc: dict) -> None:
        fname = self._safe_filename(doc["title"]) + ".json"
        out_path = self._output / fname
        # Don't overwrite existing (idempotent)
        if out_path.exists():
            return
        out_path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.debug("Saved: {}", fname)
