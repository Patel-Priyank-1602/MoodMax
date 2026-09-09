"""
GET /api/history       — Paginated analysis history.
GET /api/history/:id   — Single analysis detail.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db import crud
from ..schemas import HistoryResponse, HistoryItem

router = APIRouter(prefix="/api", tags=["history"])


@router.get("/history", response_model=HistoryResponse)
async def get_history(
    limit: int = Query(20, ge=1, le=100, description="Number of results per page"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    search: str = Query(None, description="Search text in input"),
    db: AsyncSession = Depends(get_db),
):
    """Get paginated analysis history."""
    analyses, total = await crud.get_history(db, limit=limit, offset=offset, search=search)

    items = [
        HistoryItem(
            id=a.id,
            input_text=a.input_text,
            detected_lang=a.detected_lang,
            sentiment_label=a.sentiment_label,
            sentiment_score=a.sentiment_score,
            emotion_scores=a.emotion_scores,
            dominant_emotion=a.dominant_emotion,
            source=a.source,
            created_at=a.created_at,
        )
        for a in analyses
    ]

    return HistoryResponse(
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/history/{analysis_id}", response_model=HistoryItem)
async def get_analysis_detail(
    analysis_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get details of a single analysis."""
    analysis = await crud.get_analysis_by_id(db, analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found.")

    return HistoryItem(
        id=analysis.id,
        input_text=analysis.input_text,
        detected_lang=analysis.detected_lang,
        sentiment_label=analysis.sentiment_label,
        sentiment_score=analysis.sentiment_score,
        emotion_scores=analysis.emotion_scores,
        dominant_emotion=analysis.dominant_emotion,
        source=analysis.source,
        created_at=analysis.created_at,
    )
