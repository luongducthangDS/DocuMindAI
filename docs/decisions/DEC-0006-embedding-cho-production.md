# DEC-0006: Lấy vector embedding cho production khi host không đủ RAM

- **Trạng thái:** Proposed — **chờ Ted duyệt phương án trước khi code**
- **Ngày:** 2026-09-17
- **Spec liên quan:** chưa có (sẽ viết sau khi chốt phương án)
- **Ảnh hưởng tới:** `render.yaml`, `src/rag/embedder.py`, `src/config.py`, có thể cả corpus (nếu chọn C)

## 1. Bối cảnh — vấn đề gì buộc phải quyết

`render.yaml:57` đang đặt `EMBEDDING_PROVIDER=hf_api` cho production. Gọi thử API đó bằng chính
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
| Trọng số trên đĩa | ~2.2GB (FP32) |

Render free tier có **512MB RAM**. Riêng trọng số đã gấp hơn 4 lần, chưa kể torch + transformers.
Ngay cả máy dev 15.2GB cũng crash (exit 5 / segfault) khi chỉ còn 1.9–2.7GB trống.

Ràng buộc bất biến: **collection `documind_legal` chứa 1146 vector 1024 chiều sinh từ chính model
này**. Bất kỳ phương án nào đổi model đều buộc phải re-index — và `get_chroma_collection(verify=True)`
sẽ ném `EmbeddingModelMismatch` ngay lúc khởi động nếu nhãn lệch, đúng như thiết kế.

## 2. Các phương án đã cân nhắc

> ⚠️ Cột **latency p95** dưới đây là **ước lượng, chưa đo** — không đo được vì chính cái đang hỏng là
> đường lấy vector. Mọi số latency phải được thay bằng số đo thật trước khi DEC này chuyển sang
> Accepted. Cột chi phí HF lấy từ bảng giá công khai (xem §7); chi phí FPT Cloud cần Ted điền từ
> bảng giá thực tế và credit đang có.

| | A — Self-host trên VM FPT Cloud | B — HF Inference Endpoint riêng | C — Đổi sang model có API hosted |
|---|---|---|---|
| **Cách làm** | Dựng FastAPI `/embed` trên VM, load model một lần, trả vector 1024 chiều. Render gọi HTTP kèm API key | Tạo Dedicated Endpoint, chọn container **TEI** (Text Embeddings Inference) — TEI hỗ trợ bge-m3 và trả vector, khác hẳn serverless API | Chọn model khác có API trả vector sẵn (OpenAI `text-embedding-3-large`, Cohere `embed-multilingual-v3`, Voyage…) |
| **Chi phí/tháng** | Trong credit FPT (tiền mặt ~0 khi còn credit). **Ted điền số VM cụ thể** | CPU từ **$0.033/giờ** (AWS, tính theo phút). Chạy 24/7 ≈ **$24/tháng**; instance đủ RAM cho 0.6B sẽ cao hơn mức sàn. Scale-to-zero giảm được nhưng thêm cold start | Theo lượng token. Rẻ ở quy mô demo, nhưng phụ thuộc nhà cung cấp và thêm một khoá API nữa |
| **Latency p95 (ước lượng)** | ~300–400ms (CPU inference 1 câu ~80–200ms + mạng Render↔FPT ~30–60ms) | ~200–350ms khi warm; **cold start hàng chục giây** nếu bật scale-to-zero | ~150–300ms |
| **RAM cần** | VM ≥ 4GB (model FP32 ~2.4GB + overhead). FP16/ONNX hạ xuống ~2GB | Do HF cấp theo instance đã chọn | 0 phía mình |
| **Re-index?** | **Không** — cùng model, cùng vector | **Không** — cùng model, cùng vector | **Bắt buộc** re-index 1146 chunk. Ted đã nói phải duyệt riêng |
| **Rủi ro** | Thêm một dịch vụ phải tự vận hành, tự giám sát, tự vá. VM chết = toàn hệ thống chết. Hết credit = phải trả tiền thật hoặc migrate gấp | Phụ thuộc HF cho thành phần cốt lõi. Cold start làm p95 vô nghĩa nếu bật scale-to-zero; tắt scale-to-zero thì trả tiền 24/7 cho một demo | Đổi model = đổi chất lượng retrieval. `reports/embedding_ab.json` cho thấy model hiện tại đạt final@8 = 1.000 còn MiniLM chỉ 0.821 — **không có gì đảm bảo** model mới giữ được mức đó cho tiếng Việt. Phải chạy lại A/B trước khi chốt |

Phương án D — **không làm gì** — nghĩa là production tiếp tục hỏng. Loại.

Phương án E — **nâng plan Render** cho đủ RAM: chưa khảo sát, nhưng nó giải quyết vấn đề mà không
thêm dịch vụ nào. Em đề xuất Ted cân nhắc song song với A và B; nếu chi phí xấp xỉ $24/tháng của B
thì nó đơn giản hơn hẳn cả ba.

## 3. Quyết định

**Chưa quyết.** DEC này dừng ở mục 2 chờ Ted chọn, đúng theo chỉ đạo "0e-1 — CHƯA CODE, gửi DEC trước".

Nghiêng của em: **B** nếu chấp nhận ~$24/tháng và tắt scale-to-zero (ít việc vận hành nhất, không
re-index, không thêm dịch vụ phải tự nuôi); **A** nếu ưu tiên dùng credit FPT và chấp nhận tự vận
hành. **C** chỉ nên xét khi cả A và B đều không khả thi, vì nó đặt cược lại chất lượng retrieval
tiếng Việt đã đo được.

## 4. Kiểm chứng bắt buộc trước khi Accept

Theo yêu cầu của Ted, phương án được chọn phải chứng minh vector mới **khớp** vector local:

- **Tiêu chí:** cosine ≥ **0.9999** trên **20 câu mẫu**, đo trên model **1024 chiều hiện tại**.
- **Câu mẫu:** 20 câu đầu của `data/eval/temporal_questions.json` (cố định, không chọn lại).
- **Vector đối chứng:** sinh bằng `sentence-transformers` chạy ở nơi đủ RAM (Colab / Kaggle / VM),
  **không** phải trên máy dev đang crash.
- **Ghi kèm:** tên model, số chiều, hash file trọng số, ngày chạy, commit hash.
- **Kết quả lưu tại:** `reports/embedding_parity.json` — lệnh tái tạo ghi trong chính file đó.

Nếu cosine < 0.9999 thì vector không tương thích với corpus đã index, và phương án đó **bị loại**
bất kể chi phí — dùng tiếp sẽ là dạng hỏng tệ nhất: hệ thống vẫn xếp hạng, vẫn trả lời, và sai
không ai thấy (cùng loại lỗi mà `EmbeddingModelMismatch` sinh ra để chặn).

## 5. Hệ quả

- **Được:** production lấy được vector, tức chạy được. Hết một lỗi chặn đường.
- **Trả giá:** A và B đều thêm một điểm hỏng qua mạng vào đường phục vụ nóng — retrieval không còn
  là lời gọi in-process. Cần timeout, retry, và một chỉ số riêng cho nó ở Phase 3.
- **Sai khi nào:** nếu về sau corpus đủ nhỏ hoặc model đủ nhẹ (bge-m3 bản ONNX quantized, hoặc một
  model 1024 chiều nhỏ hơn đạt cùng final@8) để chạy vừa trong RAM của host, thì cả A lẫn B đều
  thừa và nên quay về `local`.

## 6. Việc đi kèm đã làm ngay (không chờ duyệt)

Ba chỗ trong repo đang khẳng định sai sự thật về đường `hf_api` — sửa ngay vì để lại thì người đọc
sau (kể cả chính mình) sẽ tin. (`CLAUDE.md` thì không nhắc tới `hf_api` ở đâu cả — đã grep, không
có gì để sửa ở đó.)

| Chỗ | Nói sai gì |
|---|---|
| `src/rag/embedder.py` docstring `_HFInferenceAPIEmbedding` | "same 384-dim pooled vectors (verified to match the local SentenceTransformer output byte-for-byte)" — 384 chiều là MiniLM **cũ**; kiểm chứng đó không áp dụng cho model 1024 chiều hiện tại |
| `render.yaml` comment | "(đã verify: cùng model, vector giống hệt local, không cần re-index corpus)" — cùng một kiểm chứng cũ, đang được dẫn cho cấu hình hiện tại |
| `src/config.py` | comment cạnh `embedding_provider` ghi "Cùng model, cùng số chiều, corpus không cần re-index khi đổi provider" — đúng về lý thuyết, nhưng đọc như thể đường này dùng được |

## 7. Câu hỏi phỏng vấn tự đặt

1. *Sao không phát hiện sớm hơn?* — Vì không có smoke test nào gọi thật đường `hf_api`. Nó chỉ bật
   ở production, và production không có eval. Bài học thuộc về Phase 1 (CI) hơn là về embedding.
2. *Serverless API khác Dedicated Endpoint chỗ nào?* — Serverless chỉ mở task model khai báo
   (`sentence-similarity` → trả điểm, không trả vector). Dedicated cho chọn container; TEI trả vector.
3. *Sao không đổi luôn sang model nhẹ hơn?* — Vì `reports/embedding_ab.json` đo được model hiện tại
   đạt final@8 = 1.000 so với 0.821 của MiniLM. Đổi model là đặt cược lại con số đó, và phải trả giá
   bằng một lần re-index cộng một lần A/B nữa.

---

**Nguồn giá HF:** [Inference Endpoints pricing](https://huggingface.co/docs/inference-endpoints/en/pricing) ·
[Hugging Face Inference Endpoints Pricing 2026 (Spheron)](https://www.spheron.network/blog/hugging-face-inference-endpoints-pricing-2026/)
**Nguồn thông số model:** [AITeamVN/Vietnamese_Embedding](https://huggingface.co/AITeamVN/Vietnamese_Embedding)
