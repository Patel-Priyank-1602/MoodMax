"""
POST /api/analyze — Single text analysis endpoint (stateless).
"""

from fastapi import APIRouter, HTTPException

from ..ml.pipeline import ml_pipeline
from ..schemas import AnalyzeRequest, AnalyzeResponse, SentimentResult

router = APIRouter(prefix="/api", tags=["analyze"])


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest):
    """Analyze a single text for sentiment and emotion."""
    text = request.text.strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty or whitespace only.")

    # Run ML pipeline
    result = ml_pipeline.analyze(text)

    return AnalyzeResponse(
        input_text=text,
        detected_lang=result.detected_lang,
        sentiment=SentimentResult(
            label=result.sentiment_label,
            score=result.sentiment_score,
        ),
        emotions=result.emotion_scores,
        dominant_emotion=result.dominant_emotion,
    )
