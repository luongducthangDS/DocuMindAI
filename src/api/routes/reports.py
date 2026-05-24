"""
Report routes:
  POST /api/v1/report/create — generate PDF report
  GET  /api/v1/reports/{filename} — download generated report
"""

from __future__ import annotations

import smtplib
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger

from src.api.schemas import ReportRequest, ReportResponse
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["reports"])


@router.post("/report/create", response_model=ReportResponse)
async def create_report(
    body: ReportRequest,
    background_tasks: BackgroundTasks,
) -> ReportResponse:
    """
    Trigger report generation. Heavy work runs in background thread.
    Returns download URL immediately after enqueuing.
    """
    from src.api.main import ensure_rag_initialized
    from src.agent.tools import generate_pdf_report

    try:
        await ensure_rag_initialized()
        result_msg = await generate_pdf_report.ainvoke({
            "title": body.title,
            "query": body.query,
            "output_filename": body.filename,
        })
    except Exception as exc:
        logger.error("Report generation failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed. Please retry.",
        )

    safe_name = body.filename + ".pdf"
    download_url = f"/api/v1/reports/{safe_name}"

    if body.email_to:
        background_tasks.add_task(_send_report_email, body.email_to, body.title, safe_name)

    return ReportResponse(
        status="success",
        filename=safe_name,
        download_url=download_url,
        message=result_msg,
    )


@router.get("/reports/{filename}")
async def download_report(filename: str) -> FileResponse:
    """
    Download a previously generated report PDF.
    Validates filename to prevent path traversal.
    """
    import re

    if not re.match(r"^[a-zA-Z0-9_-]+\.pdf$", filename):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename",
        )

    settings = get_settings()
    file_path = (settings.reports_dir / filename).resolve()

    # Double-check the resolved path stays in reports_dir
    if not str(file_path).startswith(str(settings.reports_dir.resolve())):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path")

    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{filename}' not found",
        )

    return FileResponse(
        path=str(file_path),
        media_type="application/pdf",
        filename=filename,
    )


def _send_report_email(to_email: str, report_title: str, filename: str) -> None:
    """Send report as email attachment. Runs in background thread."""
    settings = get_settings()

    if not settings.smtp_user or not settings.smtp_password:
        logger.warning("SMTP not configured, skipping email delivery")
        return

    file_path = settings.reports_dir / filename
    if not file_path.exists():
        logger.error("Report file not found for email: {}", filename)
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = settings.smtp_user
        msg["To"] = to_email
        msg["Subject"] = f"DocuMind AI — Báo cáo: {report_title[:80]}"

        body = (
            f"Xin chào,\n\nBáo cáo pháp lý '{report_title}' đã được tạo và đính kèm bên dưới.\n\n"
            "Lưu ý: Nội dung chỉ mang tính tham khảo.\n\n— DocuMind AI"
        )
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with open(file_path, "rb") as f:
            attachment = MIMEApplication(f.read(), _subtype="pdf")
            attachment.add_header("Content-Disposition", "attachment", filename=filename)
            msg.attach(attachment)

        with smtplib.SMTP(settings.smtp_host, settings.smtp_port) as server:
            server.starttls()
            server.login(settings.smtp_user, settings.smtp_password)
            server.send_message(msg)

        logger.info("Report emailed to {} successfully", to_email[:5] + "***")

    except Exception as exc:
        logger.error("Failed to send report email: {}", exc)
