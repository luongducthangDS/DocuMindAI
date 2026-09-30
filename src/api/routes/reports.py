"""
Report routes:
  POST /api/v1/report/create — generate PDF report
  GET  /api/v1/reports/{filename} — download generated report
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger

from src.api.principal import require_admin
from src.api.schemas import ReportRequest, ReportResponse
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["reports"])


@router.post("/report/create", response_model=ReportResponse, dependencies=[Depends(require_admin)])
async def create_report(body: ReportRequest) -> ReportResponse:
    """
    Generate a PDF report (admin: X-Admin-Key — mỗi báo cáo tốn vài lời gọi LLM).
    Tên file do server đặt ngẫu nhiên; file tự xoá sau 24h.
    """
    from src.api.main import ensure_rag_initialized
    from src.agent.tools import create_report_file

    try:
        await ensure_rag_initialized()
        path = await create_report_file(body.title, body.query)
    except Exception as exc:
        logger.error("Report generation failed: {}", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Report generation failed. Please retry.",
        )

    download_url = f"/api/v1/reports/{path.name}"

    return ReportResponse(
        status="success",
        filename=path.name,
        download_url=download_url,
        message=f"Báo cáo đã tạo: {download_url}",
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
    from src.agent.tools import purge_expired_reports

    purge_expired_reports(settings.reports_dir)  # hết 24h thì 404, kể cả chưa ai tạo báo cáo mới
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
