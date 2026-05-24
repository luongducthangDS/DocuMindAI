# Railway Production Checklist

This project runs one Railway service with two processes:

- Streamlit UI: public, listens on Railway `$PORT`
- FastAPI backend: internal only, listens on `127.0.0.1:9000`

## Required Railway Settings

In `Settings -> Networking`, the public domain must target the Streamlit port:

```text
Port 8080
```

If the public domain points to `8081` or `9000`, users will hit the FastAPI backend instead of the UI.

In `Settings -> Deploy`, leave Start Command empty so Dockerfile `CMD` is used. If a command is required, use:

```text
sh scripts/start_railway.sh
```

## Required Variables

Use `railway.production.env.example` as the source of truth. The most important values are:

```text
ENVIRONMENT=production
API_HOST=127.0.0.1
API_PORT=9000
API_BASE_URL=http://127.0.0.1:9000
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
CHROMA_COLLECTION=documind_legal
CHROMA_HOST=
INITIALIZE_RAG_ON_STARTUP=false
ENABLE_RERANKER=false
LANGCHAIN_TRACING_V2=false
REDIS_URL=
```

Do not use the local development values below in Railway:

```text
ENVIRONMENT=development
API_HOST=0.0.0.0
API_PORT=8081
EMBEDDING_MODEL=BAAI/bge-m3
CHROMA_HOST=localhost
REDIS_URL=redis://localhost:6379/0
LANGCHAIN_TRACING_V2=true
```

## Expected Logs

After deploy, logs should include:

```text
Starting DocuMind AI - environment: production
Uvicorn running on http://127.0.0.1:9000
You can now view your Streamlit app
URL: http://0.0.0.0:8080
```

After the first user query, logs should include:

```text
ChromaDB collection 'documind_legal' has 356 chunks
```

If Railway still has an old `EMBEDDING_MODEL` value, the app logs a warning and forces the indexed MiniLM model.

## Deployment

```powershell
git add .
git commit -m "fix: production Railway deployment"
git push
```

If Railway is not connected to GitHub auto-deploy, redeploy manually from the dashboard.

## Security

If API keys were pasted into chat, logs, screenshots, or commits, rotate them immediately in the provider dashboards and update Railway Variables with the new values.
