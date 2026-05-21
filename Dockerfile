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
COPY ingest.py ./

# Pre-loaded corpus (ChromaDB local persistent data)
COPY data/chroma_db/ ./data/chroma_db/

# Create runtime directories with correct ownership
RUN mkdir -p data/raw data/processed data/eval logs reports && \
    chown -R appuser:appgroup /app

# Drop to non-root
USER appuser

# Health check (Railway uses $PORT, default 8080)
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD python -c "import httpx,os; httpx.get(f'http://localhost:{os.getenv(\"PORT\",\"8080\")}/api/v1/health', timeout=5).raise_for_status()"

EXPOSE 8080

# Railway injects $PORT; fall back to 8080 for local docker run
CMD ["sh", "-c", "uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --log-level info"]
