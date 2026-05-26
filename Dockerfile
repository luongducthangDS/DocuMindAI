# ── Stage 1: Build React frontend ─────────────────────────────────────────────
FROM node:20-slim AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build


# ── Stage 2: Python build (install deps) ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


# ── Stage 3: Runtime ───────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL org.opencontainers.image.title="DocuMind AI"
LABEL org.opencontainers.image.description="RAG + Agentic AI for Vietnamese Legal Documents"
LABEL org.opencontainers.image.version="1.0.0"

# Non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup -s /sbin/nologin appuser

WORKDIR /app

# Python packages from builder
COPY --from=builder /install /usr/local

# React build output
COPY --from=frontend-builder /app/dist ./dist

# Application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY ingest.py ./

# Pre-loaded corpus
COPY data/chroma_db/ ./data/chroma_db/

# Cache dirs
ENV HOME=/app
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

RUN mkdir -p data/raw data/processed data/eval logs reports .cache/huggingface

# Fix Windows line endings
RUN sed -i 's/\r$//' scripts/start_railway.sh

# Pre-download embedding model at build time
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
print('Downloading embedding model (~120MB)...'); \
SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'); \
print('Model cached.')"

RUN chown -R appuser:appgroup /app
USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import httpx,os; httpx.get(f'http://localhost:{os.getenv(\"PORT\",\"8080\")}/api/v1/health', timeout=5).raise_for_status()"

EXPOSE 8080

CMD ["sh", "scripts/start_railway.sh"]
