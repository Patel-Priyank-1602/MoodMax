"""
Pydantic schemas for request/response validation.
Matches the PRD API contract (Section 8).
"""

from datetime import datetime
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
    id: int
    detected_lang: Optional[str] = None
    sentiment: SentimentResult
    emotions: dict[str, float]  # {"joy": 0.91, "surprise": 0.06, ...}
    dominant_emotion: str
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryItem(BaseModel):
    """Single item in history list."""
    id: int
    input_text: str
    detected_lang: Optional[str] = None
    sentiment_label: Optional[str] = None
    sentiment_score: Optional[float] = None
    emotion_scores: Optional[dict] = None
    dominant_emotion: Optional[str] = None
    source: str
    created_at: datetime

    class Config:
        from_attributes = True


class HistoryResponse(BaseModel):
    """GET /api/history response."""
    items: list[HistoryItem]
    total: int
    limit: int
    offset: int


class BatchJobResponse(BaseModel):
    """POST /api/analyze/batch response."""
    batch_job_id: int
    total_items: int
    status: str


class BatchJobStatusResponse(BaseModel):
    """GET /api/batch/:id response."""
    id: int
    filename: Optional[str] = None
    total_items: int
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None


class AnalyticsSummaryResponse(BaseModel):
    """GET /api/analytics/summary response."""
    total_analyses: int
    sentiment_breakdown: dict[str, int]
    emotion_breakdown: dict[str, int]
    top_languages: dict[str, int]
