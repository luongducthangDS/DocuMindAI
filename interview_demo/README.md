# DocuMind AI — Interview Demo

Thư mục này chứa phiên bản **đơn giản hóa** của hệ thống, dùng để giải thích
kiến trúc trong interview. Code production thực sự nằm ở `src/`.

Mỗi file ở đây là một "version giải thích được trong 5 phút" của component tương ứng.

| File demo | File production | Nội dung |
|-----------|----------------|----------|
| `01_hybrid_retriever.py` | `src/rag/retriever.py` | Dense + BM25 + RRF + Reranker |
| `02_grader.py` | `src/rag/grader.py` | LLM-as-judge self-correction |
| `03_agentic_loop.py` | `src/agent/research.py` | Enough/Fetch/Search decision |
| `04_langgraph_flow.py` | `src/agent/graph.py` | Full LangGraph agent workflow |

## Cách chạy demo

```bash
# Cài dependencies tối thiểu
pip install langchain-google-genai langgraph chromadb llama-index

# Set API key
export GOOGLE_API_KEY=your_key

# Chạy từng file
python interview_demo/01_hybrid_retriever.py
python interview_demo/04_langgraph_flow.py
```
