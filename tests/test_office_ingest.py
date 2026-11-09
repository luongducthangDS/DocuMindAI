"""DOCX / XLSX ingestion: content in order, headers kept per row, hostile files rejected."""
import io
import zipfile

import pytest
from docx import Document
from openpyxl import Workbook

from src.ingestion.loader import load_docx, load_upload, load_xlsx


def _docx_bytes() -> bytes:
    d = Document()
    d.add_paragraph("Quy chế làm thêm giờ của phòng Nhân sự áp dụng từ ngày 01/01/2026 cho toàn bộ nhân viên.")
    t = d.add_table(rows=2, cols=2)
    t.cell(0, 0).text, t.cell(0, 1).text = "Loại ngày", "Hệ số lương"
    t.cell(1, 0).text, t.cell(1, 1).text = "Ngày lễ", "300%"
    d.add_paragraph("Điều khoản cuối: nhân viên phải đăng ký làm thêm trước 24 giờ qua hệ thống nội bộ.")
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _xlsx_bytes() -> bytes:
    wb = Workbook()
    ws = wb.active
    ws.title = "Phep_nam"
    ws.append(["Nhân viên", "Phòng ban", "Số ngày phép còn lại"])
    ws.append(["Nguyễn Văn A", "Kế toán", 7])
    ws.append([None, None, None])  # blank row skipped
    ws.append(["Trần Thị B", "Nhân sự", 12])
    ws["E4"] = "=C4*2"  # formula: cached value is None in a never-opened file, never the formula text
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def test_docx_keeps_paragraph_and_table_order():
    doc = load_docx(_docx_bytes(), "quy_che.docx")
    c = doc["content"]
    assert doc["doc_type"] == "uploaded_docx"
    assert c.index("Quy chế") < c.index("Ngày lễ | 300%") < c.index("Điều khoản cuối")


def test_xlsx_rows_carry_headers():
    doc = load_upload(_xlsx_bytes(), "phep.xlsx")
    lines = doc["content"].splitlines()
    assert lines[0] == "Phep_nam | Nhân viên: Nguyễn Văn A; Phòng ban: Kế toán; Số ngày phép còn lại: 7"
    assert len(lines) == 2 and "Trần Thị B" in lines[1]
    assert "=C4" not in doc["content"]


def test_wrong_magic_bytes_rejected():
    with pytest.raises(ValueError, match="magic bytes"):
        load_xlsx(b"%PDF-1.7 not a workbook" * 10, "fake.xlsx")


def test_zip_bomb_rejected():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", b"0" * (101 * 1_048_576))  # compresses to ~100 KB
    with pytest.raises(ValueError, match="zip bomb"):
        load_docx(buf.getvalue(), "bomb.docx")


def test_unknown_extension_rejected():
    with pytest.raises(ValueError, match="Unsupported"):
        load_upload(b"hello", "notes.txt")
