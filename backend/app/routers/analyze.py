"""
POST /api/analyze — Single text analysis endpoint.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from ..db.database import get_db
from ..db import crud
from ..ml.pipeline import ml_pipeline
from ..schemas import AnalyzeRequest, AnalyzeResponse, SentimentResult

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
):
    """Analyze a single text for sentiment and emotion."""
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace only.")

    # Run ML pipeline
    result = ml_pipeline.analyze(text)

    # Persist to database
    analysis = await crud.create_analysis(
        db=db,
        input_text=text,
        detected_lang=result.detected_lang,
        sentiment_label=result.sentiment_label,
        sentiment_score=result.sentiment_score,
        emotion_scores=result.emotion_scores,
        dominant_emotion=result.dominant_emotion,
        source="single",
    )

    return AnalyzeResponse(
        id=analysis.id,
        detected_lang=result.detected_lang,
        sentiment=SentimentResult(
            label=result.sentiment_label,
            score=result.sentiment_score,
        ),
        emotions=result.emotion_scores,
        dominant_emotion=result.dominant_emotion,
        created_at=analysis.created_at,
    )
