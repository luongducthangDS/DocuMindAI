"""
Report routes:
  POST /api/v1/report/create — generate PDF report
  GET  /api/v1/reports/{filename} — download generated report
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from loguru import logger

from src.api.schemas import ReportRequest, ReportResponse
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["reports"])


@router.post("/report/create", response_model=ReportResponse)
async def create_report(body: ReportRequest) -> ReportResponse:
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
