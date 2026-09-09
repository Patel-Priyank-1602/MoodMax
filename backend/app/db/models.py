"""
SQLAlchemy ORM models matching the PRD database schema.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime, ForeignKey, JSON
)
from sqlalchemy.orm import relationship
from .database import Base


class BatchJob(Base):
    """Tracks batch CSV upload jobs."""
    __tablename__ = "batch_jobs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=True)
    total_items = Column(Integer, default=0)
    status = Column(String(20), default="processing")  # processing | done | failed
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)

    # Relationship
    analyses = relationship("Analysis", back_populates="batch_job")


class Analysis(Base):
    """Individual text analysis result."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_text = Column(Text, nullable=False)
    detected_lang = Column(String(10), nullable=True)
    sentiment_label = Column(String(20), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    emotion_scores = Column(JSON, nullable=True)  # {"joy": 0.91, "surprise": 0.06, ...}
    dominant_emotion = Column(String(30), nullable=True)
    source = Column(String(20), default="single")  # single | batch
    batch_job_id = Column(Integer, ForeignKey("batch_jobs.id"), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationship
    batch_job = relationship("BatchJob", back_populates="analyses")
