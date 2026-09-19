from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Phạm vi sản phẩm ──────────────────────────────────────────────────────────
# NGUỒN SỰ THẬT DUY NHẤT cho "DocuMind AI trả lời về cái gì". Mọi prompt, message
# từ chối và nhãn UI phải lấy từ đây — trước đây 5 nơi tự khai báo scope riêng và
# cả 5 đều nói "tài liệu ngân hàng" sau khi corpus đã pivot sang lao động/BHXH.
# Đổi phạm vi corpus ⇒ sửa 3 hằng số này + frontend/src/main.tsx (PRODUCT).
DOMAIN_NAME = "pháp luật lao động và bảo hiểm xã hội Việt Nam"
DOMAIN_SCOPE = (
    "Bộ luật Lao động, Luật Bảo hiểm xã hội, Luật Việc làm cùng các nghị định, "
    "thông tư hướng dẫn"
)
DOMAIN_TOPICS = (
    "hợp đồng lao động, tiền lương, thời giờ làm việc, kỷ luật lao động, "
    "bảo hiểm xã hội, bảo hiểm thất nghiệp, hưu trí"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM — Gemini là nhà cung cấp DUY NHẤT (2026-09-19, quyết định của Ted).
    # Không còn Groq / OpenAI-compatible / model local: mọi lời gọi LLM đi qua
    # src.rag.generator.gemini_generate, xoay vòng (key x model).
    google_api_key: str = ""
    google_api_key_2: str = ""
    google_api_key_3: str = ""
    gemini_judge_models: str = "gemini-3.1-flash-lite,gemini-3.5-flash-lite"
    # gemini_generation_models: danh sach model Gemini cho generation, phan cach dau phay.
    # Generation se xoay vong 3 key x cac model nay. Uu tien model RPD cao.
    # Free tier RPD/RPM (Ted 2026-09-10): 3.1-flash-lite=500/15, 3.5-flash-lite=500/15,
    # 2.5-flash-lite=20/10. Cac "flash" thuong (2.5/3/3.5/3.6/3.7/3.8) chi RPD 20.
    # gemini-2.0-* / 1.5-* / 2.5-pro / 3.1-pro = KHONG co quota tren key nay.
    gemini_generation_models: str = "gemini-3.1-flash-lite,gemini-3.5-flash-lite,gemini-2.5-flash-lite"
    # Embedding chạy qua Gemini Embedding API — không nạp model nào vào RAM.
    # Collection mang nhãn model đã index; lệch nhãn ⇒ EmbeddingModelMismatch lúc mở
    # store, nên đổi giá trị này BẮT BUỘC chạy: python scripts/reembed_corpus.py --yes
    embedding_model: str = "gemini-embedding-001"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "documind-ai"

    # HuggingFace
    hf_token: str = ""

    # Vector store — "chroma" (mặc định, local/self-hosted) hoặc "qdrant" (Qdrant Cloud,
    # xem default-tech-stack: production nên dùng Qdrant Cloud). Đổi provider chỉ cần
    # đổi biến này + set QDRANT_URL/QDRANT_API_KEY, không cần sửa code retrieval.
    vector_store_provider: str = "chroma"

    # ChromaDB
    chroma_host: str = ""
    chroma_port: int = 8000
    chroma_collection: str = "documind_legal"

    # Qdrant Cloud
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    qdrant_collection: str = "documind_legal"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8081
    environment: str = "development"
    allowed_origins: str = "http://localhost:8501"

    # Rate limiting
    rate_limit_per_minute: int = 10

    # Runtime resource controls
    # initialize_rag_on_startup=True: pre-warms embedder + ChromaDB during lifespan startup,
    # avoiding cold-start on first user query (~8-12s penalty). Set False for dev/local.
    initialize_rag_on_startup: bool = True
    # enable_reranker=True: cross-encoder reranker runs locally (no API), adds ~150ms,
    # improves context_precision measurably. Set False to reduce memory on constrained hosts.
    enable_reranker: bool = True
    # reranker_model: cross-encoder for reranking. Default is the multilingual
    # BGE reranker (handles Vietnamese natively). For low-RAM hosts you can fall
    # back to the smaller English "cross-encoder/ms-marco-MiniLM-L-6-v2" via .env,
    # but it scores Vietnamese pairs poorly. bge-reranker-v2-m3 ≈ 2.2GB on first download.
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Storage
    data_dir: Path = Path("./data")
    reports_dir: Path = Path("./reports")
    logs_dir: Path = Path("./logs")
    sqlite_db: Path = Path("./data/documind.db")

    # Ingestion
    max_upload_size_mb: int = 50
    chunk_size: int = 512
    chunk_overlap: int = 64

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.reports_dir, self.logs_dir, self.data_dir / "raw",
                  self.data_dir / "processed", self.data_dir / "eval"):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
