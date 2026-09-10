"""
Database CRUD operations for analyses and batch jobs.
"""

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select, func, desc
from sqlalchemy.ext.asyncio import AsyncSession

from .models import Analysis, BatchJob


# ─────────────────────────────────────────────
# Analysis CRUD
# ─────────────────────────────────────────────

async def create_analysis(
    db: AsyncSession,
    input_text: str,
    detected_lang: str,
    sentiment_label: str,
    sentiment_score: float,
    emotion_scores: dict,
    dominant_emotion: str,
    source: str = "single",
    batch_job_id: Optional[int] = None,
) -> Analysis:
    """Create a new analysis record."""
    analysis = Analysis(
        input_text=input_text,
        detected_lang=detected_lang,
        sentiment_label=sentiment_label,
        sentiment_score=sentiment_score,
        emotion_scores=emotion_scores,
        dominant_emotion=dominant_emotion,
        source=source,
        batch_job_id=batch_job_id,
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)
    return analysis


async def get_analysis_by_id(db: AsyncSession, analysis_id: int) -> Optional[Analysis]:
    """Get a single analysis by ID."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    return result.scalar_one_or_none()


async def get_history(
    db: AsyncSession,
    limit: int = 20,
    offset: int = 0,
    search: Optional[str] = None,
) -> tuple[list[Analysis], int]:
    """Get paginated analysis history with optional search."""
    query = select(Analysis).order_by(desc(Analysis.created_at))
    count_query = select(func.count(Analysis.id))

    if search:
        search_filter = Analysis.input_text.ilike(f"%{search}%")
        query = query.where(search_filter)
        count_query = count_query.where(search_filter)

    # Get total count
    total_result = await db.execute(count_query)
    total = total_result.scalar()

    # Get paginated results
    query = query.offset(offset).limit(limit)
    result = await db.execute(query)
    analyses = result.scalars().all()

    return list(analyses), total


# ─────────────────────────────────────────────
# Batch Job CRUD
# ─────────────────────────────────────────────

async def create_batch_job(
    db: AsyncSession,
    filename: str,
    total_items: int,
) -> BatchJob:
    """Create a new batch job."""
    batch_job = BatchJob(
        filename=filename,
        total_items=total_items,
        status="processing",
    )
    db.add(batch_job)
    await db.commit()
    await db.refresh(batch_job)
    return batch_job


async def update_batch_job_status(
    db: AsyncSession,
    batch_job_id: int,
    status: str,
) -> Optional[BatchJob]:
    """Update batch job status."""
    result = await db.execute(select(BatchJob).where(BatchJob.id == batch_job_id))
    batch_job = result.scalar_one_or_none()
    if batch_job:
        batch_job.status = status
        if status in ("done", "failed"):
            batch_job.completed_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(batch_job)
    return batch_job


async def get_batch_job(db: AsyncSession, batch_job_id: int) -> Optional[BatchJob]:
    """Get a batch job by ID."""
    result = await db.execute(select(BatchJob).where(BatchJob.id == batch_job_id))
    return result.scalar_one_or_none()


async def get_batch_results(db: AsyncSession, batch_job_id: int) -> list[Analysis]:
    """Get all analyses belonging to a batch job."""
    result = await db.execute(
        select(Analysis)
        .where(Analysis.batch_job_id == batch_job_id)
        .order_by(Analysis.id)
    )
    return list(result.scalars().all())


# ─────────────────────────────────────────────
# Analytics
# ─────────────────────────────────────────────

async def get_analytics_summary(
    db: AsyncSession,
    batch_job_id: Optional[int] = None,
) -> dict:
    """Get aggregate analytics summary."""
    query = select(Analysis)

    if batch_job_id:
        query = query.where(Analysis.batch_job_id == batch_job_id)

    result = await db.execute(query)
    analyses = result.scalars().all()

    # Sentiment breakdown
    sentiment_breakdown = {"Positive": 0, "Negative": 0, "Neutral": 0}
    emotion_breakdown = {"joy": 0, "sadness": 0, "anger": 0, "fear": 0,
                         "surprise": 0, "disgust": 0, "neutral": 0}
    language_breakdown = {}

    for analysis in analyses:
        # Sentiment
        if analysis.sentiment_label in sentiment_breakdown:
            sentiment_breakdown[analysis.sentiment_label] += 1

        # Dominant emotion
        if analysis.dominant_emotion and analysis.dominant_emotion in emotion_breakdown:
            emotion_breakdown[analysis.dominant_emotion] += 1

        # Language
        lang = analysis.detected_lang or "unknown"
        language_breakdown[lang] = language_breakdown.get(lang, 0) + 1

    return {
        "total_analyses": len(analyses),
        "sentiment_breakdown": sentiment_breakdown,
        "emotion_breakdown": emotion_breakdown,
        "top_languages": dict(sorted(language_breakdown.items(),
                                      key=lambda x: x[1], reverse=True)),
    }
