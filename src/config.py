from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # LLM
    groq_api_key: str = ""
    google_api_key: str = ""
    google_api_key_2: str = ""
    google_api_key_3: str = ""
    gemini_judge_models: str = "gemini-3.1-flash-lite"
    primary_llm: str = "groq/llama-3.3-70b-versatile"
    fallback_llm: str = "gemini/gemini-2.5-flash-lite"
    # generator_provider: nhà cung cấp sinh câu trả lời. "groq" (mặc định, llama-3.3-70b)
    # hoặc "gemini" (bỏ qua Groq, dùng thẳng Gemini — hữu ích khi Groq cạn TPD/ngày).
    generator_provider: str = "groq"
    # gemini_generation_models: danh sách model Gemini cho generation, phân cách dấu phẩy.
    # Generation sẽ xoay vòng 3 key × các model này. Ưu tiên model RPD cao (3.1-flash-lite=500/ngày).
    gemini_generation_models: str = "gemini-2.5-flash-lite,gemini-3.1-flash-lite,gemini-2.5-flash"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    # embedding_provider: "local" (mặc định, load model vào RAM qua sentence-transformers/
    # torch, ~700MB) hoặc "hf_api" (gọi HuggingFace Inference API, không load model —
    # dùng cho host RAM thấp như Render free 512MB). Cùng model, cùng vector 384-dim,
    # corpus ChromaDB không cần re-index. Cần HF_TOKEN khi dùng "hf_api".
    embedding_provider: str = "local"

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

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8081
    api_secret_key: str = "change-me"
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

    # Email
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""

    @field_validator("allowed_origins")
    @classmethod
    def parse_origins(cls, v: str) -> str:
        return v

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
