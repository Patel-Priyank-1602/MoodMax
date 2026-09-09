"""
MoodMax Backend — FastAPI Application Entry Point.

Multilingual Sentiment & Emotion Analysis API.
Loads ML models on startup, serves analysis endpoints.

Run with: uvicorn app.main:app --reload --port 8000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .db.database import init_db
from .ml.pipeline import ml_pipeline
from .routers import analyze, batch, history, analytics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """App startup/shutdown lifecycle."""
    # Startup
    logger.info("Starting MoodMax backend...")

    # Initialize database tables
    logger.info("Initializing database...")
    await init_db()

    # Load ML models
    logger.info("Loading ML models (this may take a minute on first run)...")
    ml_pipeline.load_models()

    logger.info("MoodMax backend ready! ✓")
    yield

    # Shutdown
    logger.info("Shutting down MoodMax backend...")


# Create FastAPI app
app = FastAPI(
    title="MoodMax API",
    description="Multilingual Sentiment & Emotion Analysis — "
                "Analyze social media text for sentiment (Positive/Negative/Neutral) "
                "and emotion distribution (Joy, Sadness, Anger, Fear, Surprise, Disgust, Neutral).",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",   # Alternate
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(analyze.router)
app.include_router(batch.router)
app.include_router(history.router)
app.include_router(analytics.router)


@app.get("/", tags=["health"])
async def root():
    """Health check / welcome endpoint."""
    return {
        "app": "MoodMax",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Detailed health check."""
    return {
        "status": "healthy",
        "models": {
            "emotion": "loaded" if ml_pipeline._emotion_available else "fallback",
            "sentiment": "loaded" if ml_pipeline._sentiment_available else "fallback",
            "language_detection": "loaded" if ml_pipeline._langdetect_available else "fallback",
        },
    }
