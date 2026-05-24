# DocuMind AI — Portfolio Reference

> File này dùng để chuẩn bị CV và phỏng vấn.  
> Không cần commit nếu không muốn, nhưng giữ lại để reference.

---

## CV Bullets — Design-Focused

### Phiên bản ngắn (1 dòng mỗi điểm — cho CV 1 trang)

```
• Thiết kế pipeline RAG 3 tầng (intent routing → hybrid retrieval → reranking) đạt RAGAS faithfulness 0.871, +25% context recall so với baseline dense-only
• Chọn LangGraph thay AgentExecutor: explicit state machine giúp mỗi node (router/retrieve/generate) có thể unit test độc lập, thêm intent mới không sửa code cũ
• Implement Redis cache trước retrieval (không phải sau LLM): tiết kiệm toàn bộ pipeline 1.2s thay vì chỉ generation 0.8s; estimated 40% cache hit rate
• Triển khai so sánh 3 chiến lược RAG (naive / hybrid / agentic) bằng RAGAS; viết benchmark harness tái sử dụng được với bất kỳ test set nào
• Tích hợp LangSmith @traceable phát hiện Groq timeout 8% → implement Gemini fallback < 200ms overhead; mọi query có trace đầy đủ
• Deploy Docker multi-stage (build/runtime), non-root user, pre-download ML models 120MB tại build time để tránh Railway healthcheck timeout
```

---

### Phiên bản dài (2-3 dòng — cho LinkedIn / portfolio page)

**[RAG Architecture]**
> Thiết kế và đánh giá 3 chiến lược RAG cho hệ thống hỏi đáp pháp luật Việt Nam: (1) naive vector search, (2) hybrid BM25 + dense với RRF fusion, (3) agentic với LangGraph intent routing. Đo bằng RAGAS trên 20 câu hỏi: hybrid đạt +25.5% context recall, agentic đạt faithfulness 0.871 — vượt target 0.80.

**[System Design Decision]**
> Chọn BM25 + dense hybrid thay vì dense-only sau khi xác định 30%+ query chứa tên điều khoản cụ thể ("Điều 48 Luật DN 2020") mà embedding model xử lý kém hơn BM25 2x. Dùng Reciprocal Rank Fusion để fuse kết quả mà không cần tune hyperparameter khi corpus thay đổi.

**[Agent Orchestration]**
> Implement LangGraph state machine với 7 nodes (router, retrieve, answer, compare, summarize, report, persist) thay vì LangChain AgentExecutor. Lý do: mỗi node là pure function với typed input/output — có thể mock và test độc lập; thêm intent handler mới không ảnh hưởng code cũ.

**[Observability & Reliability]**
> Tích hợp LangSmith với @traceable decorator trên run_agent() và generate_answer(); trace phát hiện Groq timeout rate 8% → implement Gemini 1.5 Flash fallback tự động. Deploy trên Railway với Docker multi-stage, healthcheck 300s, pre-download embedding model (120MB) tại build time.

---

## Interview Talking Points

### "Tại sao dùng LangGraph?"

**Câu trả lời ngắn (30 giây):**
> "Tôi cần orchestrate 4 intent khác nhau với shared state. LangGraph cho phép mỗi node là independent function, dễ test và dễ extend. Nếu dùng LangChain AgentExecutor, tôi sẽ mất khả năng test từng bước và trace sẽ là 1 blob khó debug."

**Câu trả lời sâu (nếu interviewer follow up):**
> "Cụ thể, khi phát hiện bug ở bước retrieve, tôi có thể gọi `retrieve_node(state)` trực tiếp với mock state, không cần chạy toàn bộ graph. Với AgentExecutor, không làm được điều này. LangGraph cũng cho phép tôi thêm intent 'report' mà không sửa các node khác — chỉ thêm 1 node và 1 conditional edge."

---

### "Tại sao cache trước retrieval?"

**Câu trả lời:**
> "Có 2 option: cache sau LLM (cache full response) hoặc cache trước retrieval (cache early). Tôi chọn cache trước retrieval vì:
> 1. Tiết kiệm được cùng lượng latency (~1.2s) nhưng key đơn giản hơn — chỉ là query string, không phải serialized LLM output
> 2. Nếu đổi response format, cache cũ vẫn valid — không bị stale cache vì schema change
> Trade-off: câu trả lời có thể stale nếu corpus update trong 1h TTL. Acceptable vì văn bản pháp luật không thay đổi hàng giờ."

---

### "Tại sao hybrid retrieval thay vì chỉ vector search?"

**Câu trả lời với số liệu:**
> "Tôi đo context recall trên 20 câu hỏi: dense-only đạt 0.591, hybrid đạt 0.742 — tức là +25.5%. Lý do là corpus pháp luật có nhiều query chứa exact terms như 'Điều 48 Luật Doanh nghiệp 2020'. Embedding model nén câu này thành vector nhưng mất thông tin exact match — BM25 xử lý tốt hơn trong trường hợp này. Với semantic query như 'quyền của người lao động', dense lại tốt hơn. RRF fuse cả 2 mà không cần tune weight."

---

### "RAGAS là gì và bạn dùng nó như thế nào?"

**Câu trả lời:**
> "RAGAS là framework đánh giá RAG pipeline với 4 metrics chính:
> - Faithfulness: câu trả lời có groundied trong context được retrieve không?
> - Answer Relevancy: câu trả lời có trả lời đúng câu hỏi không?
> - Context Recall: context retrieve có cover được ground truth không?
> - Context Precision: context retrieve có relevant không hay có nhiều noise?
>
> Tôi dùng RAGAS để so sánh 3 chiến lược RAG trên cùng 20 câu hỏi với ground truth viết tay. Judge LLM là Groq llama-3.3-70b. Kết quả dùng để justify quyết định hybrid > naive (+25% recall) và document trong README."

---

### "Làm sao bạn xử lý hallucination?"

**Câu trả lời:**
> "3 tầng phòng vệ:
> 1. **Retrieval grounding**: LLM chỉ nhận top-5 chunks relevant nhất — không có information sẽ không generate
> 2. **System prompt enforcement**: 'Mỗi câu PHẢI trích dẫn điều khoản cụ thể [N]. Nếu không có trong văn bản được cung cấp, phải nói rõ.' LLM tự chọn [N] phù hợp, không post-process
> 3. **RAGAS faithfulness**: đo lường định kỳ. Hiện tại 0.871 — nghĩa là 87% câu trong answer có thể verify từ retrieved context."

---

## Key Numbers to Remember

| Metric | Value | Context |
|---|---|---|
| RAGAS Faithfulness | **0.871** | Agentic RAG, 20 test questions |
| Context Recall | **+25.5%** | Hybrid vs Naive RAG delta |
| Corpus size | **356 chunks**, 6 doc types | 18 Vietnamese legal documents |
| LLM provider uptime | **~92%** | Groq; 8% timeout → Gemini fallback |
| Cache hit rate | **~40%** | Estimated on repeated legal queries |
| Latency P95 | **3.4s** | Full agentic pipeline |
| Embedding model | **120MB** | MiniLM-L12-v2, CPU-only |
| Docker image build | **8-15 min** | Railway (model download at build) |

---

## Honest Limitations (câu trả lời thật khi bị hỏi)

| Limitation | Explanation |
|---|---|
| Corpus nhỏ (356 chunks) | Đủ để demo và eval pipeline; production cần 10K+ chunks từ toàn bộ vbpl.vn |
| Embedding model nhỏ | MiniLM thay vì bge-m3 vì Railway memory limit (512MB free); bge-m3 cho recall tốt hơn |
| Cache chưa có invalidation | TTL 1h là workaround; production cần event-driven invalidation khi corpus update |
| Redis sessions in-memory | Chỉ persist 1 session; production cần Redis để scale horizontally |
| RAGAS judge = same LLM | Groq judge Groq answer → potential bias; production dùng LLM khác làm judge |
