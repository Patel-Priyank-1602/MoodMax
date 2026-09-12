"""
POST /api/analyze/batch — In-memory batch CSV processing.
Stateless and immediate: takes CSV, runs inference, aggregates stats, returns results.
"""

import csv
import io
from fastapi import APIRouter, HTTPException, UploadFile, File

from ..ml.pipeline import ml_pipeline
from ..schemas import BatchAnalyzeResponse, BatchSummary, BatchItemResult

router = APIRouter(prefix="/api", tags=["batch"])


@router.post("/analyze/batch", response_model=BatchAnalyzeResponse)
async def analyze_batch(file: UploadFile = File(...)):
    """Upload a CSV file with a 'text' column for batch sentiment + emotion analysis."""
    # Validate file extension
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    # Read CSV content
    try:
        content = await file.read()
        text_content = content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text_content))
        rows = list(reader)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="CSV file is empty.")

    # Find the text column (case-insensitive search for 'text' or first column fallback)
    fieldnames = reader.fieldnames or []
    text_col = None
    for field in fieldnames:
        if field and field.strip().lower() == "text":
            text_col = field
            break

    if not text_col:
        raise HTTPException(
            status_code=400,
            detail=f"CSV must contain a 'text' column. Found columns: {', '.join(fieldnames)}"
        )

    # Process each row in-memory
    results: list[BatchItemResult] = []
    sentiment_breakdown = {"Positive": 0, "Negative": 0, "Neutral": 0}
    emotion_breakdown = {
        "joy": 0, "sadness": 0, "anger": 0, "fear": 0,
        "surprise": 0, "disgust": 0, "neutral": 0
    }
    language_breakdown: dict[str, int] = {}

    for row in rows:
        text = (row.get(text_col) or "").strip()
        if not text:
            continue

        res = ml_pipeline.analyze(text)

        # Aggregate stats
        if res.sentiment_label in sentiment_breakdown:
            sentiment_breakdown[res.sentiment_label] += 1

        dom_emo = (res.dominant_emotion or "neutral").lower()
        if dom_emo in emotion_breakdown:
            emotion_breakdown[dom_emo] += 1

        lang = res.detected_lang or "unknown"
        language_breakdown[lang] = language_breakdown.get(lang, 0) + 1

        results.append(
            BatchItemResult(
                input_text=text,
                detected_lang=res.detected_lang,
                sentiment_label=res.sentiment_label,
                sentiment_score=res.sentiment_score,
                emotion_scores=res.emotion_scores,
                dominant_emotion=res.dominant_emotion,
            )
        )

    if not results:
        raise HTTPException(status_code=400, detail="No non-empty text rows found in CSV.")

    summary = BatchSummary(
        total_analyses=len(results),
        sentiment_breakdown=sentiment_breakdown,
        emotion_breakdown=emotion_breakdown,
        top_languages=dict(sorted(language_breakdown.items(), key=lambda x: x[1], reverse=True)),
    )

    return BatchAnalyzeResponse(
        filename=file.filename,
        summary=summary,
        results=results,
    )
