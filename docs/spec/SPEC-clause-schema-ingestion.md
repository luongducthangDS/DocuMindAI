# Spec: `clause-schema-ingestion`

> Module 2/9. Phụ thuộc: `corpus-acquisition`. Xem `SPEC.md`.
> Định nghĩa schema chi tiết: `docs/corpus/CORPUS_SPEC.md` §2.

## Objective

Parse toàn văn `data/raw/lao_dong/` thành chunk cấp **điều/khoản**, gắn metadata **version cấp khoản**, và build vector index. Đây là module sở hữu **contract schema metadata** mà `temporal-retrieval`, `scope-gate`, `question-library` đều đọc.

## Contract: schema metadata chunk (mỗi bản của một khoản = 1 record)

Bắt buộc (tối thiểu Ted yêu cầu): `doc_id, dieu, khoan, effective_from, effective_to, superseded_by`.

Đầy đủ (đề xuất — xem `CORPUS_SPEC.md` §2.2 để có bảng kiểu đầy đủ):

| Trường | Ý nghĩa |
|---|---|
| `clause_uid` | `<doc_id>__d<dieu>[_k<khoan>][_p<diem>]`, vd `45-2019-QH14__d139_k1` |
| `version_id` | `<clause_uid>__v<effective_from>` |
| `doc_id`, `doc_so_hieu` | trỏ về manifest + hiển thị trích dẫn |
| `dieu` (int), `dieu_tieu_de` (str) | số điều + tên điều (dùng cho question-library) |
| `khoan` (int\|null), `diem` (str\|null) | null = phần mở đầu điều |
| `text` | nội dung bản này |
| `effective_from` (date), `effective_to` (date\|null) | null = còn hiệu lực |
| `status` | `in_force` \| `superseded` \| `not_yet_in_force` \| `repealed` |
| `superseded_by`, `amended_by_doc`, `supersedes_version` | quan hệ version |
| `source_url`, `consolidated_from` | nguồn text + số VBHN nếu lấy từ hợp nhất |
| `verify_status` | `VERIFIED` bắt buộc để index; `UNVERIFIED` **không index** |

**Đa số khoản chỉ có 1 version** (`effective_from` = ngày VB có hiệu lực, `effective_to` = null hoặc ngày VB hết hiệu lực). Chỉ khoản bị **sửa in-place** (từ bảng `sua_doi_boi` của `corpus-acquisition`) mới có ≥2 version — hiện gần như chỉ Điều 139 k1 BLLĐ.

## Boundaries (delta)

- **Ask first:** đổi tên/kiểu trường trong schema trên (3 module đọc nó).
- Mở rộng `chunk_by_dieu` hiện có (`src/ingestion/chunker.py`), **không viết chunker mới song song** — thêm khả năng tách tới `khoan` + gắn version metadata.
- `LegalChunk.metadata` hiện có `source_url, title, doc_type, so_hieu, ngay_ban_hanh, dieu_header, khoan_count, char_count, source` — giữ các trường này (backward-compat với retriever/generator), **thêm** trường version, không xoá.
- Chunk-by-Điều là chiến lược chính; khi điều > 4000 ký tự vẫn tách theo khoản như hiện tại.

## Công việc

1. **Loader manifest:** đọc `corpus_manifest.yaml` + bảng `sua_doi_boi`. Xây bản đồ `doc_id → {ngay_hieu_luc, het_hieu_luc_tu, danh sách khoản bị sửa in-place}`.
2. **Mở rộng `chunker.py`:** với mỗi Điều, sinh `clause_uid` cấp điều (và cấp khoản khi Điều bị sửa in-place ở cấp khoản). Gắn `dieu`, `dieu_tieu_de`, `khoan`, `effective_from/to`, `status`, `verify_status`.
3. **Dựng version cho khoản bị sửa in-place:** đọc 2 bản text (Điều 139 k1: bản 2021 + bản 2026-07-01 từ `corpus-acquisition`). Sinh 2 record: bản cũ `effective_to=2026-07-01, status=superseded, superseded_by=<version mới>`; bản mới `effective_from=2026-07-01, status=not_yet_in_force` (as_of hôm nay 2026-09-10 → thực ra `in_force`; tính theo ngày build).
4. **Ingest:** mở rộng `scripts/ingest_documents.py` để đọc `data/raw/lao_dong/` + manifest YAML (hiện đọc manifest JSON). Đẩy chunk + full metadata vào ChromaDB. Bỏ record `verify_status != VERIFIED`.
5. **Validation script** (`scripts/validate_corpus.py`): frontmatter đủ trường; mọi `clause_uid` unique trong 1 version; mọi `superseded_by` trỏ tới `version_id` tồn tại; không record `UNVERIFIED` trong index.

## Success Criteria

- [ ] `python scripts/ingest_documents.py --source-dir data/raw/lao_dong --manifest docs/corpus/corpus_manifest.yaml --reset` chạy sạch, in số chunk/văn bản.
- [ ] ChromaDB collection chứa chunk từ tất cả VB VERIFIED; `count > 0`; mỗi chunk có `clause_uid`, `effective_from`, `status`.
- [ ] Điều 139 BLLĐ có đúng 2 record khoản 1 với `effective_from` khác nhau; các điều khác 1 record.
- [ ] `scripts/validate_corpus.py` pass; 0 record `UNVERIFIED`.
- [ ] `tests/test_ingestion.py` mở rộng: test parse điều→khoản, test sinh `clause_uid`, test 2-version cho khoản sửa in-place, test loại `UNVERIFIED`.
- [ ] Retriever + generator hiện tại vẫn chạy (metadata cũ còn nguyên) — `pytest tests/test_rag.py` xanh.

## Verify

`pytest tests/test_ingestion.py tests/test_rag.py -v` + chạy ingest thật + `scripts/validate_corpus.py`.

## Open Questions

1. Tách tới cấp `khoan` cho **mọi** điều (chunk nhỏ hơn, nhiều hơn — ảnh hưởng retrieval) hay chỉ cho điều bị sửa in-place, còn lại giữ chunk cấp điều? (đề xuất: chunk cấp điều mặc định, cấp khoản chỉ khi cần version — giữ recall).
2. ChromaDB metadata không nhận `None`/`date` object — lưu `effective_to` rỗng là `""` hay `"9999-12-31"`? (ảnh hưởng filter ở `temporal-retrieval`).
3. Có cần giữ song song chunk "cấp điều gộp" + "cấp khoản" cho điều bị sửa để không mất ngữ cảnh khi retrieve?
