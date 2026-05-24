# ── Build stage ────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build deps (needed for some C extensions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install to isolated prefix for clean copy
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DocuMind AI"
LABEL org.opencontainers.image.description="RAG + Agentic AI for Vietnamese Legal Documents"
LABEL org.opencontainers.image.version="1.0.0"

# Security: non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY src/ ./src/
COPY frontend/ ./frontend/
COPY scripts/ ./scripts/
COPY ingest.py ./

# Pre-loaded corpus (ChromaDB local persistent data)
COPY data/chroma_db/ ./data/chroma_db/

# appuser is a system user with no home dir — redirect everything to /app
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Create runtime directories
RUN mkdir -p data/raw data/processed data/eval logs reports .cache/huggingface

# Pre-download embedding model at build time.
# IMPORTANT: embedding model MUST match what was used to index data/chroma_db/.
# The corpus was indexed with paraphrase-multilingual-MiniLM-L12-v2.
# Do NOT change without re-running: python ingest.py --source json --dir data/raw
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading embedding model (~120MB)...'); \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print('Model cached to /app/.cache/huggingface')"

# Hand ownership to appuser
RUN chown -R appuser:appgroup /app

# Drop to non-root
USER appuser

# Health check (Railway uses $PORT, default 8080)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import httpx,os; httpx.get(f'http://localhost:{os.getenv(\"PORT\",\"8080\")}/_stcore/health', timeout=5).raise_for_status()"

EXPOSE 8080

# Railway injects $PORT; Streamlit is public, FastAPI runs on 127.0.0.1:8081.
CMD ["sh", "scripts/start_railway.sh"]
