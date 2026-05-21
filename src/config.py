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
    primary_llm: str = "groq/llama-3.3-70b-versatile"
    fallback_llm: str = "gemini/gemini-1.5-flash"
    embedding_model: str = "BAAI/bge-m3"

    # LangSmith
    langchain_tracing_v2: bool = False
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str = ""
    langchain_project: str = "documind-ai"

    # HuggingFace
    hf_token: str = ""

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "documind_legal"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # FastAPI
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_secret_key: str = "change-me"
    environment: str = "development"
    allowed_origins: str = "http://localhost:8501"

    # Rate limiting
    rate_limit_per_minute: int = 10

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
