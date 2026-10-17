# Gemini API Guidelines for this Project
1. Default Model: Luôn sử dụng `gemini-3.8-flash` cho các tác vụ sinh nội dung, suy luận, code và multimodal (15 RPM / 1,500 RPD Free).
2. High-speed / Lightweight: Dùng `gemini-3.5-flash-lite` cho tác vụ phân loại, tóm tắt ngắn, trích xuất JSON để tiết kiệm 50% chi phí và đạt độ trễ ~1s.
3. RAG & Vector Search: Luôn dùng `gemini-embedding-2` (Hạn mức Free lên tới 1,500 RPM).
4. Deprecated Notice: Tuyệt đối không dùng `gemini-2.5-flash` hay `gemini-2.5-pro` (đã bị Google ngừng hỗ trợ HTTP 404).
5. Error Handling: Áp dụng Exponential Backoff khi gặp mã 429 và fallback sang `3.8-flash` nếu `3.5-flash` gặp mã 503.
