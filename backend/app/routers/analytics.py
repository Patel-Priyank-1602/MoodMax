"""
GET /api/analytics/summary — Aggregate analytics dashboard data.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db import crud
from ..schemas import AnalyticsSummaryResponse

router = APIRouter(prefix="/api", tags=["analytics"])


@router.get("/analytics/summary", response_model=AnalyticsSummaryResponse)
async def get_analytics_summary(
    batch_job_id: Optional[int] = Query(None, description="Filter by batch job ID"),
    db: AsyncSession = Depends(get_db),
):
    """Get aggregate analytics summary for dashboard."""
    summary = await crud.get_analytics_summary(db, batch_job_id=batch_job_id)

    return AnalyticsSummaryResponse(
        total_analyses=summary["total_analyses"],
        sentiment_breakdown=summary["sentiment_breakdown"],
        emotion_breakdown=summary["emotion_breakdown"],
        top_languages=summary["top_languages"],
    )
