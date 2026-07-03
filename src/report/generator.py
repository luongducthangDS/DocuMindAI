"""
ReportLab PDF generator for legal document summaries.
Produces structured PDF with header, table of contents, sections, footnotes.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from src.config import get_settings

if TYPE_CHECKING:
    from src.rag.retriever import RetrievedChunk


def _safe_path(reports_dir: Path, filename: str) -> Path:
    """Prevent path traversal when constructing report output path."""
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in filename)
    safe = safe[:80] or "report"
    # Resolve and check it stays inside reports_dir
    out = (reports_dir / f"{safe}.pdf").resolve()
    if not str(out).startswith(str(reports_dir.resolve())):
        raise ValueError(f"Invalid filename: would escape reports directory")
    return out


def create_legal_report(
    title: str,
    summary: str,
    chunks: list["RetrievedChunk"],
    filename: str = "report",
    author: str = "DocuMind AI",
) -> Path:
    """
    Generate a PDF report and save to reports_dir.
    Returns the output file path.
    """
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.platypus import (
            Paragraph,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
            HRFlowable,
        )
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError as exc:
        raise RuntimeError("Install reportlab: pip install reportlab") from exc

    settings = get_settings()
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = _safe_path(settings.reports_dir, filename)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=title[:200],      # cap title length for PDF metadata
        author=author,
        subject="Báo cáo pháp lý — DocuMind AI",
        creator="DocuMind AI v1.0",
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocuTitle",
        parent=styles["Title"],
        fontSize=18,
        spaceAfter=12,
        textColor=colors.HexColor("#1a237e"),
    )
    heading_style = ParagraphStyle(
        "DocuHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=14,
        spaceAfter=6,
        textColor=colors.HexColor("#283593"),
    )
    body_style = ParagraphStyle(
        "DocuBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leading=14,
    )
    caption_style = ParagraphStyle(
        "DocuCaption",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
    )

    story = []

    # Header
    story.append(Paragraph(title[:200], title_style))
    story.append(
        Paragraph(
            f"Tạo bởi DocuMind AI | {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            caption_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#283593")))
    story.append(Spacer(1, 0.4 * cm))

    # Summary section
    if summary:
        story.append(Paragraph("Tóm tắt", heading_style))
        # Escape XML special chars for ReportLab
        safe_summary = summary.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for para in safe_summary.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), body_style))
        story.append(Spacer(1, 0.3 * cm))

    # Sources table
    if chunks:
        story.append(Paragraph("Văn bản tham chiếu", heading_style))

        table_data = [["#", "Văn bản", "Điều khoản", "Độ liên quan"]]
        for i, chunk in enumerate(chunks[:20], 1):  # cap at 20 rows
            table_data.append([
                str(i),
                chunk.metadata.get("title", "")[:40],
                chunk.metadata.get("dieu_header", "")[:50],
                f"{chunk.score:.2f}",
            ])

        tbl = Table(
            table_data,
            colWidths=[0.8 * cm, 6 * cm, 7 * cm, 2 * cm],
            repeatRows=1,
        )
        tbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#283593")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, 0), 9),
            ("FONTSIZE", (0, 1), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#e8eaf6")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#9fa8da")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(tbl)
        story.append(Spacer(1, 0.3 * cm))

    # Detailed excerpts
    if chunks:
        story.append(Paragraph("Nội dung chi tiết", heading_style))
        for i, chunk in enumerate(chunks[:10], 1):
            dieu = chunk.metadata.get("dieu_header", "")
            if dieu:
                story.append(Paragraph(f"[{i}] {dieu[:100]}", body_style))
            excerpt = chunk.text[:600].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(excerpt + "...", caption_style))
            story.append(Spacer(1, 0.15 * cm))

    # Footer note
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(
        Paragraph(
            "Tài liệu này được tạo tự động bởi DocuMind AI. "
            "Nội dung chỉ mang tính tham khảo, không thay thế tư vấn pháp lý chuyên nghiệp.",
            caption_style,
        )
    )

    doc.build(story)
    logger.info("PDF report generated: {}", out_path)
    return out_path
