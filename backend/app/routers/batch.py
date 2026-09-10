"""
POST /api/analyze/batch — CSV batch upload and processing.
GET  /api/batch/:id     — Get batch job status.
"""

import csv
import io
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db, async_session
from ..db import crud
from ..ml.pipeline import ml_pipeline
from ..schemas import BatchJobResponse, BatchJobStatusResponse, HistoryItem

router = APIRouter(prefix="/api", tags=["batch"])


async def process_batch_job(batch_job_id: int, rows: list[dict]):
    """Background task: process all rows in a batch job."""
    async with async_session() as db:
        try:
            processed = 0
            for row in rows:
                text = row.get("text", "").strip()
                if not text:
                    continue

                result = ml_pipeline.analyze(text)

                await crud.create_analysis(
                    db=db,
                    input_text=text,
                    detected_lang=result.detected_lang,
                    sentiment_label=result.sentiment_label,
                    sentiment_score=result.sentiment_score,
                    emotion_scores=result.emotion_scores,
                    dominant_emotion=result.dominant_emotion,
                    source="batch",
                    batch_job_id=batch_job_id,
                )
                processed += 1

            await crud.update_batch_job_status(db, batch_job_id, "done")

        except Exception as e:
            await crud.update_batch_job_status(db, batch_job_id, "failed")
            raise e


@router.post("/analyze/batch", response_model=BatchJobResponse)
async def analyze_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload a CSV file for batch sentiment + emotion analysis."""
    # Validate file type
    if not file.filename or not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Read CSV content
    try:
        content = await file.read()
        text_content = content.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text_content))
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # Check for 'text' column
    if "text" not in rows[0]:
        raise HTTPException(
            status_code=400,
            detail="CSV must have a 'text' column. Found columns: " + ", ".join(rows[0].keys())
        )

    # Create batch job
    batch_job = await crud.create_batch_job(
        db=db,
        filename=file.filename,
        total_items=len(rows),
    )

    # Process in background
    background_tasks.add_task(process_batch_job, batch_job.id, rows)

    return BatchJobResponse(
        batch_job_id=batch_job.id,
        total_items=len(rows),
        status="processing",
    )


@router.get("/batch/{batch_job_id}", response_model=BatchJobStatusResponse)
async def get_batch_status(
    batch_job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get the status of a batch job."""
    batch_job = await crud.get_batch_job(db, batch_job_id)
    if not batch_job:
        raise HTTPException(status_code=404, detail="Batch job not found.")

    return BatchJobStatusResponse(
        id=batch_job.id,
        filename=batch_job.filename,
        total_items=batch_job.total_items,
        status=batch_job.status,
        created_at=batch_job.created_at,
        completed_at=batch_job.completed_at,
    )


@router.get("/batch/{batch_job_id}/results", response_model=list[HistoryItem])
async def get_batch_results(
    batch_job_id: int,
    db: AsyncSession = Depends(get_db),
):
    """Get all individual analysis results for a batch job."""
    batch_job = await crud.get_batch_job(db, batch_job_id)
    if not batch_job:
        raise HTTPException(status_code=404, detail="Batch job not found.")

    analyses = await crud.get_batch_results(db, batch_job_id)

    return [
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
