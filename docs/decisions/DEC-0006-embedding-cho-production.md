# DEC-0006: Lấy vector embedding cho production khi host không đủ RAM

- **Trạng thái:** Proposed — **chờ Ted duyệt phương án trước khi code**
- **Ngày:** 2026-09-17
- **Spec liên quan:** chưa có (sẽ viết sau khi chốt phương án)
- **Ảnh hưởng tới:** `render.yaml`, `src/rag/embedder.py`, `src/config.py`, có thể cả corpus (nếu chọn C)

## 1. Bối cảnh — vấn đề gì buộc phải quyết

`render.yaml` đang đặt `EMBEDDING_PROVIDER=hf_api` cho production. Gọi thử API đó bằng chính
cấu hình production:

```
ValueError: Model 'AITeamVN/Vietnamese_Embedding' doesn't support task 'feature-extraction'.
            Supported tasks: 'sentence-similarity'
```

HF **Serverless** Inference API chỉ mở task mà model khai báo. Model này khai `sentence-similarity`
— trả về **điểm tương đồng giữa các câu**, không trả **vector**. Không có vector thì không query
được vector store. Nghĩa là **production hiện không phục vụ được một truy vấn nào**, và lỗi này im
lặng cho tới lúc có người hỏi.

Đường còn lại, `EMBEDDING_PROVIDER=local`, bị chặn bởi RAM:

| Thông số model | Giá trị |
|---|---|
| Base | `BAAI/bge-m3` |
| Tham số | 0.6B |
| Số chiều | 1024 |
| Max sequence | 2048 token |
| Trọng số FP32 | ~2.2GB |

Ràng buộc bất biến: **collection `documind_legal` chứa 1146 vector 1024 chiều sinh từ chính model
này ở FP32**. Bất kỳ phương án nào đổi model — hoặc đổi vector đủ nhiều — đều buộc phải re-index, và
`get_chroma_collection(verify=True)` sẽ ném `EmbeddingModelMismatch` ngay lúc khởi động nếu nhãn lệch.

## 2. RAM thực tế cần bao nhiêu

Chỉ tính trọng số model là thiếu. Toàn bộ những thứ cùng nằm trong một process trên Render:

| Thành phần | FP32 | FP16 | ONNX INT8 |
|---|---|---|---|
| Trọng số model | ~2.30 GB | ~1.15 GB | ~0.60 GB |
| torch runtime (CPU-only) | ~0.40 GB | ~0.40 GB | — (onnxruntime ~0.10 GB) |
| transformers + sentence-transformers + tokenizer | ~0.15 GB | ~0.15 GB | ~0.10 GB |
| FastAPI + uvicorn + llama_index core | ~0.25 GB | ~0.25 GB | ~0.25 GB |
| chromadb + HNSW index (1146 × 1024 float32 ≈ 4.7 MB vector, phần lớn là thư viện) | ~0.15 GB | ~0.15 GB | ~0.15 GB |
| BM25 corpus 1146 node trong RAM | ~0.05 GB | ~0.05 GB | ~0.05 GB |
| **Tổng ổn định** | **~3.3 GB** | **~2.15 GB** | **~1.15 GB** |
| **Đỉnh lúc load** (có `low_cpu_mem_usage=True`) | **~3.6 GB** | **~2.4 GB** | **~1.3 GB** |

> Đây là **ước lượng, chưa đo**. Phải thay bằng RSS đo thật (`psutil` đỉnh trong 60s đầu) trước khi
> DEC chuyển sang Accepted. Bảng này **chưa tính** cross-encoder reranker `bge-reranker-v2-m3`
> (~2.2GB nữa) — nó đang tắt trên Render, và bật lại là một quyết định riêng.

## 3. Các phương án

> ⚠️ Mọi cột **latency p95** là **ước lượng, chưa đo** — không đo được vì chính đường lấy vector
> đang hỏng. Phải thay bằng số đo thật trước khi Accept.

### A — Self-host trên VM FPT Cloud

Dựng FastAPI `/embed` trên VM, load model một lần, trả vector 1024 chiều. Render gọi HTTP kèm API key.

- **Chi phí:** trong credit FPT (tiền mặt ~0 khi còn credit). **Ted điền số VM cụ thể.**
- **RAM:** VM ≥ 4GB (chỉ chạy model, không kèm app) — FP16 hạ xuống ~2GB.
- **p95:** ~300–400ms *(ước lượng: CPU inference 1 câu ~80–200ms + mạng Render↔FPT ~30–60ms)*.
- **Re-index:** không.
- **Rủi ro credit:** credit FPT **có hạn sử dụng**. Khi hết, hoặc trả tiền thật theo giá niêm yết,
  hoặc migrate gấp — và migrate gấp là lúc dễ làm hỏng nhất.
  **Kế hoạch chuyển đi phải viết trước, không viết sau:**
  1. Ghi ngày hết credit vào lịch, đặt nhắc **trước 30 ngày**.
  2. Giữ interface `/embed` **giống hệt** phía client (cùng request/response shape) để đổi
     `EMBEDDING_ENDPOINT_URL` là xong, không sửa code.
  3. Viết sẵn `Dockerfile` cho service embed — bất kỳ nơi nào chạy container đều nhận được
     (Fly.io, Hetzner, HF Endpoint, hoặc chính Render với plan RAM cao).
  4. Chạy `reports/embedding_parity.json` lại sau mỗi lần đổi nhà — cùng tiêu chí ở §5.
- **Rủi ro khác:** thêm một dịch vụ phải tự vận hành, tự giám sát, tự vá. VM chết = hệ thống chết.

### B — HF Inference Endpoint riêng (Dedicated, container TEI)

Khác serverless: Dedicated cho chọn container. **TEI** (Text Embeddings Inference) hỗ trợ bge-m3
và **trả vector**, đúng thứ serverless không trả.

- **Chi phí:** CPU từ **$0.033/giờ** (AWS, tính theo phút) → **~$24/tháng** nếu chạy 24/7,
  **~$288/12 tháng**. Instance đủ RAM cho 0.6B sẽ cao hơn mức sàn này.
- **RAM:** HF cấp theo instance đã chọn.
- **p95:** ~200–350ms khi warm; **cold start hàng chục giây** nếu bật scale-to-zero.
- **Re-index:** không.
- **Rủi ro:** phụ thuộc HF cho một thành phần cốt lõi. Scale-to-zero làm p95 vô nghĩa; tắt nó thì
  trả tiền 24/7 cho một demo.

### C — Đổi sang embedding model có API hosted

OpenAI `text-embedding-3-large`, Cohere `embed-multilingual-v3`, Voyage…

- **Chi phí:** theo token, rẻ ở quy mô demo.
- **RAM:** 0 phía mình.
- **p95:** ~150–300ms *(ước lượng)*.
- **Re-index:** **bắt buộc** — 1146 chunk. Ted đã yêu cầu duyệt riêng cho phương án này.
- **Rủi ro:** `reports/embedding_ab.json` đo được model hiện tại đạt final@8 = **1.000** còn
  MiniLM chỉ 0.821. Đổi model là **đặt cược lại con số đó** cho tiếng Việt, cộng một lần A/B nữa.

### E — Nâng plan Render cho đủ RAM

Giá và cấu hình kiểm tra ngày **2026-09-17**:

| Plan | CPU | RAM | Giá/tháng | Đủ cho FP32 (~3.6GB đỉnh)? | FP16 (~2.4GB)? | INT8 (~1.3GB)? |
|---|---|---|---|---|---|---|
| `free` | 0.1 | 512 MB | $0 | ✗ | ✗ | ✗ |
| `0.5c-512mb` (Starter) | 0.5 | 512 MB | **$7** | ✗ | ✗ | ✗ |
| `1c-2g` (Standard) | 1 | 2 GB | **$25** | ✗ | ✗ | ⚠️ sát nút |
| `2c-4g` (Pro) | 2 | 4 GB | **chưa lấy được** | ⚠️ sát nút | ✓ | ✓ |
| `2c-8g` | 2 | 8 GB | **chưa lấy được** | ✓ | ✓ | ✓ |

- **Nguồn cấu hình plan:** [Render Compute Plans](https://render.com/docs/compute-plans) (chính thức),
  kiểm tra 2026-09-17.
- **Nguồn giá:** trang [render.com/pricing](https://render.com/pricing) render bằng JS nên không đọc
  được bằng công cụ fetch; hai con số $7 và $25 lấy từ nguồn **bên thứ ba**
  ([srvrlss.io](https://www.srvrlss.io/provider/render/), kiểm tra 2026-09-17). Giá `2c-4g` / `2c-8g`
  **chưa xác minh được** — **Ted xác nhận trên dashboard Render** rồi em điền.
- **Lưu ý:** Render tính **workspace fee riêng** với compute, nên tổng hoá đơn cao hơn con số compute.
- **Re-index:** không. **Số dịch vụ thêm vào đường phục vụ: 0.**

### F — Chạy model nhẹ hơn, giữ nguyên corpus (mới)

Ý tưởng: không đổi *model*, chỉ đổi *độ chính xác số học*, để nó vừa RAM mà vector vẫn đủ gần FP32.

**F1 — FP16** (~1.15GB trọng số). Hạ một nửa bộ nhớ, giữ nguyên kiến trúc. Vector **không**
bit-identical với FP32 — sai khác ở chữ số thập phân cuối.

**F2 — ONNX INT8** (~0.60GB). Quantize động, chạy bằng `onnxruntime` thay torch (bỏ luôn ~0.4GB
torch runtime). Sai khác so với FP32 lớn hơn F1 đáng kể.

Script `scripts/build_query_embedding_cache.py` đã nhận `--dtype fp32|fp16` để sinh vector câu hỏi
cho phép đo F1 (chạy trên Colab, xem docstring của script).

**Với mỗi cách, đo trên máy đủ RAM và ghi vào `reports/embedding_precision.json`:**

| Phép đo | Cách đo |
|---|---|
| RAM đỉnh | RSS lớn nhất trong 60s đầu (`psutil`), tách riêng lúc load và lúc phục vụ |
| p95 latency 1 query | 100 lần embed một câu, lấy phân vị 95 |
| cosine so với FP32 | 20 câu cố định (20 câu đầu `temporal_questions.json`), vector FP32 làm chuẩn |

**Nhánh quyết định theo kết quả cosine:**

- **cosine ≥ 0.9999** → vector coi như tương thích với corpus FP32 đã index. **Không re-index.**
- **cosine < 0.9999** → **bắt buộc đo lại `recall@8` trên 30 câu temporal**, với **query dùng vector
  mới** và **corpus giữ nguyên FP32**, rồi so với recall@8 của cấu hình FP32/FP32. Đây mới là câu hỏi
  thật: sai khác số học có làm hỏng thứ hạng không. Nếu recall@8 không tụt → **vẫn không cần
  re-index**, vì thứ quan trọng là thứ hạng chứ không phải chữ số thập phân. Nếu recall@8 tụt →
  **phải re-index toàn corpus bằng đúng dtype đó**, và khi ấy F mất phần lớn cái lợi.

Chạy được phép đo này ngay bây giờ nhờ `eval/scoring_ab.py --query-embeddings <file>`: corpus đã có
vector, chỉ cần thay vector **câu hỏi**.

### D — Không làm gì

Production tiếp tục hỏng. Loại.

## 4. Bảng tiêu chí chọn

| Tiêu chí | A (VM FPT) | B (HF TEI) | C (đổi model) | E (nâng Render) | F1 (FP16) | F2 (INT8) |
|---|---|---|---|---|---|---|
| **Chi phí 12 tháng** | ~0 trong credit, **rủi ro vách đá khi hết** | **~$288** | theo token, thấp ở quy mô demo | = chênh lệch plan × 12 (**chưa có giá `2c-4g`**) | = như E nhưng plan thấp hơn | = như E, plan thấp nhất |
| **RAM cần** | VM ≥ 4GB | HF lo | 0 | 4–8GB | ~2.4GB | ~1.3GB |
| **p95 (ước lượng)** | 300–400ms | 200–350ms warm | 150–300ms | **không đổi (in-process)** | ~in-process, có thể nhanh hơn FP32 | ~in-process, nhanh nhất |
| **Re-index?** | không | không | **có** | không | **chỉ khi recall@8 tụt** | **chỉ khi recall@8 tụt** |
| **Số dịch vụ trên đường phục vụ** | **+1** | **+1** | **+1** | **0** | **0** | **0** |
| **Rủi ro phụ thuộc bên ngoài** | tự vận hành + credit hết hạn | phụ thuộc HF | phụ thuộc nhà cung cấp + đổi chất lượng | chỉ Render (đã phụ thuộc sẵn) | không thêm | không thêm + đổi runtime sang onnxruntime |

Đọc bảng này theo hàng "số dịch vụ trên đường phục vụ": A, B, C đều thêm một điểm hỏng qua mạng vào
đường nóng của retrieval; E và F thì không thêm gì.

## 5. Kiểm chứng bắt buộc trước khi Accept

Phương án được chọn phải chứng minh vector mới **khớp** vector local:

- **Tiêu chí:** cosine ≥ **0.9999** trên **20 câu mẫu**, đo trên model **1024 chiều hiện tại**.
- **Câu mẫu:** 20 câu đầu của `data/eval/temporal_questions.json` (cố định, không chọn lại).
- **Vector đối chứng:** FP32, sinh bằng `sentence-transformers` ở nơi đủ RAM (Colab / Kaggle / VM),
  **không** phải trên máy dev đang crash.
- **Ghi kèm:** tên model, số chiều, dtype, hash trọng số, ngày chạy, commit hash.
- **Kết quả lưu tại:** `reports/embedding_parity.json` — lệnh tái tạo ghi trong chính file đó.

Với A, B, C: cosine < 0.9999 ⇒ **loại thẳng**, bất kể chi phí — vector không tương thích với corpus
đã index, và dùng tiếp là dạng hỏng tệ nhất (vẫn xếp hạng, vẫn trả lời, sai không ai thấy).
Với F1/F2: cosine < 0.9999 ⇒ **chưa loại**, chuyển sang nhánh đo `recall@8` ở §3-F, vì ở đây ta biết
chính xác nguồn sai khác là số học chứ không phải khác không gian vector.

## 6. Tạm thời (đã áp dụng, Ted xác nhận 2026-09-17)

`render.yaml` đổi `EMBEDDING_PROVIDER` từ `hf_api` sang **`local`**.

Cả hai đều chưa chạy được trên free tier, nhưng chúng hỏng **khác nhau**:

| | hỏng lúc nào | ai thấy |
|---|---|---|
| `hf_api` | ở **từng query** | chỉ người dùng gặp lỗi — service vẫn báo "healthy" |
| `local` | lúc **khởi động** | deploy đỏ ngay, không ai kịp tin là nó đang chạy |

Fail-fast tốt hơn hỏng im lặng. Đây là trạng thái **tạm**, không phải quyết định — nó chỉ chọn cách
hỏng cho đúng, cho tới khi mục 3 được chốt.

## 7. Việc đi kèm đã làm ngay (không chờ duyệt)

Ba chỗ trong repo đang khẳng định sai sự thật về đường `hf_api` — sửa ngay vì để lại thì người đọc
sau (kể cả chính mình) sẽ tin. (`CLAUDE.md` thì không nhắc tới `hf_api` ở đâu cả — đã grep, không
có gì để sửa ở đó.)

| Chỗ | Nói sai gì |
|---|---|
| `src/rag/embedder.py` docstring `_HFInferenceAPIEmbedding` | "same 384-dim pooled vectors (verified to match the local SentenceTransformer output byte-for-byte)" — 384 chiều là MiniLM **cũ**; kiểm chứng đó không áp dụng cho model 1024 chiều hiện tại |
| `render.yaml` comment | "(đã verify: cùng model, vector giống hệt local, không cần re-index corpus)" — cùng một kiểm chứng cũ, đang được dẫn cho cấu hình hiện tại |
| `src/config.py` | comment cạnh `embedding_provider` ghi "Cùng model, cùng số chiều, corpus không cần re-index khi đổi provider" — đúng về lý thuyết, nhưng đọc như thể đường này dùng được |

## 8. Hệ quả

- **Được:** production lấy được vector, tức chạy được. Hết một lỗi chặn đường.
- **Trả giá:** A, B, C thêm một điểm hỏng qua mạng vào đường phục vụ nóng — retrieval không còn là
  lời gọi in-process. Cần timeout, retry, và một chỉ số riêng cho nó ở Phase 3. E trả bằng tiền
  hàng tháng. F trả bằng rủi ro sai khác số học, phải đo mới biết.
- **Sai khi nào:** nếu về sau có một model 1024 chiều nhỏ hơn đạt cùng final@8, hoặc corpus chuyển
  sang vector store có sẵn dịch vụ embed (Qdrant Cloud có), thì cả sáu phương án đều nên xét lại.

## 9. Câu hỏi phỏng vấn tự đặt

1. *Sao không phát hiện sớm hơn?* — Vì không có smoke test nào gọi thật đường `hf_api`. Nó chỉ bật
   ở production, và production không có eval. Bài học thuộc về Phase 1 (CI) hơn là về embedding.
2. *Serverless API khác Dedicated Endpoint chỗ nào?* — Serverless chỉ mở task model khai báo
   (`sentence-similarity` → trả điểm, không trả vector). Dedicated cho chọn container; TEI trả vector.
3. *FP16 rẻ thế sao không làm luôn?* — Vì chưa biết vector FP16 còn xếp hạng giống FP32 không. Câu
   trả lời là một phép đo (§3-F), không phải một niềm tin. Nếu `recall@8` tụt thì F1 kéo theo một
   lần re-index và hết rẻ.

---

**Nguồn giá HF:** [Inference Endpoints pricing](https://huggingface.co/docs/inference-endpoints/en/pricing) ·
[Hugging Face Inference Endpoints Pricing 2026 (Spheron)](https://www.spheron.network/blog/hugging-face-inference-endpoints-pricing-2026/)
**Nguồn plan Render:** [Render Compute Plans](https://render.com/docs/compute-plans) ·
giá tham chiếu [srvrlss.io](https://www.srvrlss.io/provider/render/) — cả hai kiểm tra 2026-09-17
**Nguồn thông số model:** [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
