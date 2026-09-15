"""
Pydantic schemas for request/response validation.
Stateless Sentiment and Emotion analysis.
"""

from typing import Optional
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    """POST /api/analyze request body."""
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")


# ─────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────

class SentimentResult(BaseModel):
    label: str  # Positive / Negative / Neutral
    score: float


class AnalyzeResponse(BaseModel):
    """POST /api/analyze response."""
    input_text: str
    detected_lang: Optional[str] = None
    sentiment: SentimentResult
    emotions: dict[str, float]  # {"joy": 0.91, "surprise": 0.06, ...}
    dominant_emotion: str
    correction_applied: bool = False
    correction_type: Optional[str] = None
    correction_reason: Optional[str] = None


class BatchItemResult(BaseModel):
    """Individual item in batch analysis result."""
    input_text: str
    detected_lang: Optional[str] = None
    sentiment_label: str
    sentiment_score: float
    emotion_scores: dict[str, float]
    dominant_emotion: str
    correction_applied: bool = False
    correction_type: Optional[str] = None
    correction_reason: Optional[str] = None


class BatchSummary(BaseModel):
    """Aggregated stats for batch."""
    total_analyses: int
    sentiment_breakdown: dict[str, int]
    emotion_breakdown: dict[str, int]
    top_languages: dict[str, int]


class BatchAnalyzeResponse(BaseModel):
    """POST /api/analyze/batch response."""
    filename: Optional[str] = None
    summary: BatchSummary
    results: list[BatchItemResult]


class BatchReportRequest(BaseModel):
    """POST /api/analyze/batch/report request body."""
    filename: Optional[str] = None
    summary: BatchSummary
    results: list[BatchItemResult]


class AspectItem(BaseModel):
    """Single aspect analysis item."""
    aspect: str
    clause_text: str
    sentiment_label: str
    sentiment_score: float
    dominant_emotion: str
    emotion_scores: dict[str, float]
    correction_applied: bool = False
    correction_reason: Optional[str] = None


class AspectAnalyzeResponse(BaseModel):
    """POST /api/analyze/aspects response."""
    input_text: str
    overall_sentiment: str  # Positive / Negative / Neutral / Mixed
    has_multiple_aspects: bool
    aspects: list[AspectItem]


