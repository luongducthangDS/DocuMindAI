"""Test cho lõi retrieval tự viết.

Chạy: pytest exercises/retrieval_core -q --no-cov -p no:cacheprovider

Các số trong file này được tính tay từ công thức ghi ở README — nếu test đỏ thì
implementation lệch công thức, ĐỪNG sửa con số trong test.
"""

import math

import pytest

from exercises.retrieval_core.core import (
    BM25Index,
    Candidate,
    chunk_by_dieu,
    rerank,
    rrf_fuse,
    split_dieu,
    tokenize,
)

# ---------------------------------------------------------------------------
# Phần 1 — đã cho sẵn, test ở đây để chốt cách đếm token cho các phần sau
# ---------------------------------------------------------------------------


def test_tokenize_giu_dau_va_chuan_hoa_nfc():
    assert tokenize("Mức lương tối thiểu") == ["mức", "lương", "tối", "thiểu"]
    # "nghỉ" dựng sẵn (U+1EC9) và tổ hợp (i + U+0309) phải ra cùng một token
    assert tokenize("nghỉ") == tokenize("nghỉ")
    assert tokenize("lãi") != tokenize("lai")


# ---------------------------------------------------------------------------
# Phần 2 — Chunking
# ---------------------------------------------------------------------------

_VAN_BAN = (
    "CHÍNH PHỦ\nSố: 01/2024/NĐ-CP\n\n"
    "Điều 1. Phạm vi điều chỉnh\n"
    "Nghị định này quy định mức lương tối thiểu tháng và mức lương tối thiểu giờ "
    "đối với người lao động làm việc theo hợp đồng lao động.\n\n"
    "Điều 2. Mức lương tối thiểu\n"
    + " ".join(f"khoản{i}" for i in range(1, 71))
    + "\n"
)


def test_split_dieu_bo_phan_dau_van_ban():
    parts = split_dieu(_VAN_BAN)
    assert len(parts) == 2
    assert parts[0][0].strip() == "Điều 1."
    assert parts[1][0].strip() == "Điều 2."
    assert "CHÍNH PHỦ" not in parts[0][1]
    assert "Phạm vi điều chỉnh" in parts[0][1]


def test_bat_bien_2_moi_chunk_tu_dung_duoc():
    for c in chunk_by_dieu(_VAN_BAN, max_tokens=30, overlap_tokens=5):
        assert c.text.lstrip().startswith(c.dieu_header.strip())


def test_bat_bien_1_khong_chunk_nao_tron_hai_dieu():
    for c in chunk_by_dieu(_VAN_BAN, max_tokens=30, overlap_tokens=5):
        headers = [h for h in ("Điều 1.", "Điều 2.") if h in c.text]
        assert headers == [c.dieu_header.strip()]


def test_bat_bien_3_khong_chunk_nao_vuot_max_tokens():
    for c in chunk_by_dieu(_VAN_BAN, max_tokens=30, overlap_tokens=5):
        assert len(tokenize(c.text)) <= 30


def test_bat_bien_4_va_5_overlap_dung_va_khong_mat_token():
    chunks = [c for c in chunk_by_dieu(_VAN_BAN, max_tokens=30, overlap_tokens=5)
              if c.dieu_header.strip() == "Điều 2."]
    assert len(chunks) > 1, "Điều 2 dài 70 token, phải bị cắt thành nhiều chunk"

    ghep = list(chunks[0].body_tokens)
    for truoc, sau in zip(chunks, chunks[1:]):
        assert truoc.body_tokens[-5:] == sau.body_tokens[:5]   # bất biến 4
        ghep.extend(sau.body_tokens[5:])                        # bất biến 5

    goc = tokenize(split_dieu(_VAN_BAN)[1][1])
    assert ghep == goc


# ---------------------------------------------------------------------------
# Phần 3 — BM25. Số neo: xem README §Phần 3, corpus 3 doc, avgdl = 7.0
# ---------------------------------------------------------------------------

_DOCS = [
    "mức lương tối thiểu vùng một",                 # 6 token
    "mức lương tối thiểu vùng hai và vùng ba",      # 9 token, "vùng" xuất hiện 2 lần
    "thời giờ làm việc bình thường",                # 6 token
]


@pytest.fixture
def idx():
    return BM25Index(_DOCS)


def test_avgdl(idx):
    assert idx.avgdl == pytest.approx(7.0)


def test_idf_tinh_tay(idx):
    # df("hai") = 1 -> ln(1 + 2.5/1.5);  df("vùng") = 2 -> ln(1 + 1.5/2.5)
    assert idx.idf_of("hai") == pytest.approx(0.980829, abs=1e-6)
    assert idx.idf_of("vùng") == pytest.approx(0.470004, abs=1e-6)
    assert idx.idf_of("hai") == pytest.approx(math.log(1 + 2.5 / 1.5), abs=1e-9)


def test_score_tinh_tay(idx):
    assert idx.score("vùng hai", 0) == pytest.approx(0.502294, abs=1e-6)
    assert idx.score("vùng hai", 1) == pytest.approx(1.484047, abs=1e-6)
    assert idx.score("vùng hai", 2) == pytest.approx(0.0, abs=1e-9)


def test_term_ngoai_corpus_khong_dong_gop(idx):
    assert idx.score("blockchain", 0) == pytest.approx(0.0, abs=1e-9)


def test_search_xep_hang(idx):
    ket_qua = idx.search("vùng hai", top_k=3)
    assert [doc_id for doc_id, _ in ket_qua][:2] == [1, 0]
    assert all(s > 0 for _, s in ket_qua)


def test_doc_dai_hon_bi_phat_do_chuan_hoa_do_dai(idx):
    """Cùng tf=1 cho "vùng", doc dài hơn phải ăn điểm thấp hơn — đó là việc của b."""
    chi_vung = [idx.score("vùng", i) for i in range(3)]
    assert chi_vung[1] > chi_vung[0]  # d1 có tf=2 nên vẫn cao hơn dù dài hơn

    khong_chuan_hoa = BM25Index(_DOCS, b=0.0)
    # b=0 -> độ dài doc không còn ảnh hưởng; điểm của d0 phải đổi
    assert khong_chuan_hoa.score("vùng", 0) != pytest.approx(idx.score("vùng", 0))


# ---------------------------------------------------------------------------
# Phần 4 — RRF. Số neo tính tay với k = 60, rank từ 1.
# ---------------------------------------------------------------------------

_DENSE = ["c3", "c1", "c7"]
_BM25 = ["c1", "c9", "c3"]


def test_rrf_diem_tinh_tay():
    diem = dict(rrf_fuse([_DENSE, _BM25]))
    assert diem["c1"] == pytest.approx(1 / 62 + 1 / 61, abs=1e-9)   # 0.032522
    assert diem["c3"] == pytest.approx(1 / 61 + 1 / 63, abs=1e-9)   # 0.032266
    assert diem["c9"] == pytest.approx(1 / 62, abs=1e-9)            # 0.016129
    assert diem["c7"] == pytest.approx(1 / 63, abs=1e-9)            # 0.015873


def test_rrf_doc_hang_nhat_cua_mot_list_co_the_thua():
    """c3 đứng nhất ở dense nhưng vẫn thua c1 — đây là toàn bộ ý nghĩa của RRF."""
    assert [d for d, _ in rrf_fuse([_DENSE, _BM25])] == ["c1", "c3", "c9", "c7"]


def test_rrf_vang_mat_khong_dong_gop():
    diem = dict(rrf_fuse([_DENSE, _BM25]))
    assert diem["c7"] == pytest.approx(1 / 63, abs=1e-9)  # chỉ 1 list đóng góp


def test_rrf_top_k():
    assert len(rrf_fuse([_DENSE, _BM25], top_k=2)) == 2


def test_rrf_thang_diem_khong_bao_gio_cham_0_05():
    """Bug thật trong repo: ngưỡng abstain 0.05 của generator.py được hiệu chuẩn
    cho cross-encoder, nhưng điểm RRF tối đa với 2 list chỉ là 2/61 ≈ 0.0328."""
    cao_nhat = max(s for _, s in rrf_fuse([_DENSE, _BM25]))
    assert cao_nhat < 0.05


# ---------------------------------------------------------------------------
# Phần 5 — Rerank
# ---------------------------------------------------------------------------


def _scorer_gia(so_lan_goi):
    def scorer(cap):
        so_lan_goi.append(len(cap))
        # điểm giả: càng nhiều token trùng với query càng cao
        return [
            float(len(set(tokenize(q)) & set(tokenize(d))))
            for q, d in cap
        ]
    return scorer


def test_rerank_goi_scorer_dung_mot_lan():
    so_lan = []
    ung_vien = [Candidate(f"c{i}", t) for i, t in enumerate(_DOCS)]
    rerank("vùng hai", ung_vien, _scorer_gia(so_lan), top_n=2)
    assert so_lan == [3], "phải gọi scorer 1 lần với cả batch 3 cặp"


def test_rerank_tra_top_n_va_gan_diem_moi():
    ung_vien = [Candidate(f"c{i}", t, score=0.99) for i, t in enumerate(_DOCS)]
    ket_qua = rerank("vùng hai", ung_vien, _scorer_gia([]), top_n=2)
    assert len(ket_qua) == 2
    assert ket_qua[0].doc_id == "c1"
    assert ket_qua[0].score == pytest.approx(2.0)   # trùng "vùng" + "hai"
    assert ket_qua[0].score >= ket_qua[1].score


def test_rerank_rong_khong_no():
    assert rerank("bất kỳ", [], _scorer_gia([]), top_n=8) == []
