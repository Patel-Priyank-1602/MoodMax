"""
ML Inference Pipeline.
Loads and manages all ML models:
  1. Emotion model: fine-tuned DistilBERT or j-hartmann/emotion-english-distilroberta-base (7 classes)
  2. Sentiment model: cardiffnlp/twitter-roberta-base-sentiment-latest (or emotion-derived)
  3. Language detection: fastText lid.176.bin

Models are loaded once at startup and reused for all requests.
"""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .preprocess import clean_text

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────

@dataclass
class AnalysisResult:
    """Combined result from all ML models."""
    detected_lang: str
    sentiment_label: str    # Positive / Negative / Neutral
    sentiment_score: float  # confidence
    emotion_scores: dict    # {"joy": 0.91, "surprise": 0.06, ...}
    dominant_emotion: str


# ─────────────────────────────────────────────
# Emotion & Sentiment labels
# ─────────────────────────────────────────────

EMOTION_LABELS = ["joy", "sadness", "anger", "fear", "surprise", "disgust", "neutral"]

SENTIMENT_LABEL_MAP = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive",
    "label_0": "Negative",
    "label_1": "Neutral",
    "label_2": "Positive",
}


class MLPipeline:
    """Manages all ML models for inference."""

    def __init__(self):
        self.emotion_pipeline = None
        self.sentiment_pipeline = None
        self.lang_detector = None
        self._emotion_available = False
        self._sentiment_available = False
        self._langdetect_available = False

    def load_models(self, model_dir: Optional[str] = None):
        """Load all ML models. Called once at app startup."""
        if model_dir is None:
            model_dir = str(Path(__file__).resolve().parent.parent.parent.parent / "models")

        logger.info("Loading ML models...")

        # Optimize PyTorch CPU memory and threads for cloud containers (512MB tier)
        try:
            import torch
            torch.set_num_threads(1)
        except Exception:
            pass

        # 1. Load emotion model (primary neural engine)
        self._load_emotion_model(model_dir)

        # 2. Load sentiment model (or derive from emotion model to fit 512MB RAM)
        self._load_sentiment_model()

        # 3. Load language detector
        self._load_language_detector(model_dir)

        import gc
        gc.collect()

        logger.info("ML pipeline ready.")
        logger.info(f"  Emotion model:   {'✓ loaded' if self._emotion_available else '✗ fallback mode'}")
        logger.info(f"  Sentiment model: {'✓ loaded' if self._sentiment_available else '✓ derived mode (saves ~500MB RAM)'}")
        logger.info(f"  Language detect: {'✓ loaded' if self._langdetect_available else '✗ fallback mode'}")

    def _load_emotion_model(self, model_dir: str):
        """Load the fine-tuned emotion classification model or pretrained 7-class emotion model."""
        emotion_model_path = os.path.join(model_dir, "emotion-distilbert-multi")

        # 1. Check if user-trained model exists
        if os.path.exists(emotion_model_path) and os.path.exists(
            os.path.join(emotion_model_path, "config.json")
        ):
            try:
                from transformers import pipeline as hf_pipeline

                self.emotion_pipeline = hf_pipeline(
                    "text-classification",
                    model=emotion_model_path,
                    framework="pt",
                    top_k=None,
                    device=-1,
                )
                self._emotion_available = True
                logger.info(f"  Custom emotion model loaded from {emotion_model_path}")
                return
            except Exception as e:
                logger.warning(f"  Failed to load custom emotion model: {e}")

        # 2. Load high-accuracy 7-class emotion model
        try:
            from transformers import pipeline as hf_pipeline

            logger.info("  Loading 7-class emotion model (j-hartmann/emotion-english-distilroberta-base)...")
            self.emotion_pipeline = hf_pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                framework="pt",
                top_k=None,
                device=-1,
            )
            self._emotion_available = True
            logger.info("  Emotion model loaded successfully (7 classes: joy, sadness, anger, fear, surprise, disgust, neutral)")
        except Exception as e:
            logger.warning(f"  Failed to load emotion model: {e}. Using rule-based fallback.")
            self._emotion_available = False

    def _load_sentiment_model(self):
        """Load pretrained Cardiff NLP sentiment model if explicitly enabled, or derive from emotion distribution."""
        # On memory-constrained cloud environments (e.g. Render 512MB free tier),
        # loading two separate full transformer models exceeds 512MB RAM and triggers OOM.
        # Deriving sentiment from the 7-class emotion distribution is mathematically consistent (PRD Sec 4),
        # consumes 0 extra RAM, and runs 2x faster.
        load_separate = os.environ.get("LOAD_SEPARATE_SENTIMENT_MODEL", "false").lower() == "true"
        if not load_separate:
            logger.info("  Sentiment model: Unified with 7-class emotion distribution (optimized for 512MB RAM tier).")
            self._sentiment_available = False
            return

        try:
            from transformers import pipeline as hf_pipeline

            logger.info("  Loading sentiment model (cardiffnlp/twitter-roberta-base-sentiment-latest)...")
            self.sentiment_pipeline = hf_pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                framework="pt",
                top_k=None,
                device=-1,
            )
            self._sentiment_available = True
            logger.info("  Sentiment model loaded (cardiffnlp/twitter-roberta-base-sentiment-latest)")
        except Exception as e:
            logger.warning(f"  Sentiment direct load: {e}. Deriving sentiment from emotion model.")
            self._sentiment_available = False

    def _load_language_detector(self, model_dir: str):
        """Load fastText language detection model."""
        lid_path = os.path.join(model_dir, "lid.176.bin")

        if os.path.exists(lid_path):
            try:
                import fasttext

                fasttext.FastText.eprint = lambda x: None
                self.lang_detector = fasttext.load_model(lid_path)
                self._langdetect_available = True
                logger.info(f"  Language detector loaded from {lid_path}")
            except Exception as e:
                logger.warning(f"  Failed to load fastText: {e}")
                self._langdetect_available = False
        else:
            logger.warning(f"  fastText model not found at {lid_path}. Using fallback 'en'.")
            self._langdetect_available = False

    # ─────────────────────────────────────────
    # Inference methods
    # ─────────────────────────────────────────

    def detect_language(self, text: str) -> str:
        """Detect language of text using fastText."""
        if not self._langdetect_available:
            return "en"

        try:
            clean = text.replace("\n", " ").strip()
            predictions = self.lang_detector.predict(clean, k=1)
            label = predictions[0][0] if predictions and predictions[0] else "__label__en"
            return label.replace("__label__", "")
        except Exception as e:
            logger.debug(f"Language detection fallback: {e}")
            return "en"

    def predict_emotions(self, text: str) -> dict[str, float]:
        """Predict emotion distribution: {emotion: probability}."""
        if not self._emotion_available:
            return self._mock_emotions(text)

        try:
            import torch
            with torch.no_grad():
                results = self.emotion_pipeline(text[:512])
            if results and len(results) > 0:
                scores = results[0] if isinstance(results[0], list) else results
                emotion_dict = {}
                for item in scores:
                    label = item["label"].lower()
                    if label in EMOTION_LABELS:
                        emotion_dict[label] = round(float(item["score"]), 4)
                # Ensure all 7 emotions are present
                for emotion in EMOTION_LABELS:
                    if emotion not in emotion_dict:
                        emotion_dict[emotion] = 0.0
                return emotion_dict
        except Exception as e:
            logger.error(f"Emotion prediction error: {e}")

        return self._mock_emotions(text)

    def predict_sentiment(self, text: str, emotion_scores: Optional[dict] = None) -> tuple[str, float]:
        """Predict sentiment: (label, score) using model or high-accuracy emotion derivation."""
        # 1. Direct sentiment model if available
        if self._sentiment_available:
            try:
                results = self.sentiment_pipeline(text[:512])
                if results and len(results) > 0:
                    scores = results[0] if isinstance(results[0], list) else results
                    best = max(scores, key=lambda x: x["score"])
                    raw_label = best["label"].lower()
                    label = SENTIMENT_LABEL_MAP.get(raw_label, SENTIMENT_LABEL_MAP.get(best["label"], "Neutral"))
                    return label, round(float(best["score"]), 4)
            except Exception as e:
                logger.error(f"Sentiment prediction error: {e}")

        # 2. Derive sentiment from emotion probability distribution (PRD Section 4)
        if emotion_scores:
            pos = emotion_scores.get("joy", 0.0)
            neg = sum(emotion_scores.get(e, 0.0) for e in ["sadness", "anger", "fear", "disgust"])
            neu = emotion_scores.get("neutral", 0.0) + emotion_scores.get("surprise", 0.0) * 0.5

            if pos > neg and pos > neu:
                return "Positive", round(min(max(pos, 0.6), 0.99), 4)
            elif neg > pos and neg > neu:
                return "Negative", round(min(max(neg, 0.6), 0.99), 4)
            else:
                return "Neutral", round(min(max(neu, 0.6), 0.99), 4)

        return self._mock_sentiment(text)

    @staticmethod
    def translate_to_english(text: str, source_lang: Optional[str] = None) -> str:
        """
        Translate non-English text to English to ensure maximum emotion and sentiment accuracy.
        Uses Google Translate API with fallback to original text.
        """
        if not text or not text.strip():
            return text

        # If already known to be English, no translation needed
        if source_lang and source_lang.lower() == "en":
            return text

        try:
            import json
            import urllib.parse
            import urllib.request

            clean = text.strip()
            sl = source_lang.lower() if source_lang else "auto"
            url = (
                "https://translate.googleapis.com/translate_a/single?client=gtx&sl="
                + sl
                + "&tl=en&dt=t&q="
                + urllib.parse.quote(clean)
            )
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=4) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if data and isinstance(data, list) and data[0]:
                    translated = "".join([part[0] for part in data[0] if part and part[0]]).strip()
                    if translated:
                        logger.info(f"Translated [{source_lang or 'auto'} -> en]: '{clean[:40]}...' -> '{translated[:40]}...'")
                        return translated
        except Exception as e:
            logger.warning(f"Translation failed ({e}), continuing with original text.")

        return text

    def analyze(self, text: str) -> AnalysisResult:
        """Run full analysis pipeline on input text."""
        cleaned = clean_text(text)
        if not cleaned:
            cleaned = text.strip() or "empty"

        # 1. Detect language on the user's original text
        detected_lang = self.detect_language(cleaned)

        # 2. For non-English languages, translate to English for model inference
        inference_text = cleaned
        if detected_lang and detected_lang.lower() != "en":
            inference_text = self.translate_to_english(cleaned, source_lang=detected_lang)

        # 3. Predict emotions using English-aligned model features
        emotion_scores = self.predict_emotions(inference_text)

        # 4. Get dominant emotion
        dominant_emotion = max(emotion_scores, key=emotion_scores.get)

        # 5. Predict sentiment (using model or emotion distribution)
        sentiment_label, sentiment_score = self.predict_sentiment(inference_text, emotion_scores=emotion_scores)

        return AnalysisResult(
            detected_lang=detected_lang,
            sentiment_label=sentiment_label,
            sentiment_score=sentiment_score,
            emotion_scores=emotion_scores,
            dominant_emotion=dominant_emotion,
        )

    # ─────────────────────────────────────────
    # Mock fallback (only used if all else fails)
    # ─────────────────────────────────────────

    @staticmethod
    def _mock_sentiment(text: str) -> tuple[str, float]:
        """Simple rule-based sentiment fallback."""
        text_lower = text.lower()
        positive_words = {"happy", "great", "love", "amazing", "wonderful", "good",
                          "excellent", "fantastic", "joy", "beautiful", "best", "dream",
                          "excited", "awesome", "brilliant", "perfect", "glad"}
        negative_words = {"sad", "angry", "hate", "terrible", "awful", "bad", "worst",
                          "horrible", "fear", "disgusting", "pathetic", "ugly", "miserable",
                          "disappointed", "furious", "annoyed", "depressed"}

        words = set(text_lower.split())
        pos_count = len(words & positive_words)
        neg_count = len(words & negative_words)

        if pos_count > neg_count:
            return "Positive", round(0.7 + 0.2 * min(pos_count / 3, 1), 4)
        elif neg_count > pos_count:
            return "Negative", round(0.7 + 0.2 * min(neg_count / 3, 1), 4)
        else:
            return "Neutral", 0.6

    @staticmethod
    def _mock_emotions(text: str) -> dict[str, float]:
        """Simple rule-based emotion fallback."""
        text_lower = text.lower()
        emotion_keywords = {
            "joy": ["happy", "joy", "love", "great", "amazing", "wonderful", "excited",
                     "dream", "awesome", "glad", "delighted", "fantastic"],
            "sadness": ["sad", "depressed", "cry", "grief", "miss", "lonely", "heartbreak",
                         "miserable", "disappointed", "sorrow"],
            "anger": ["angry", "furious", "hate", "annoyed", "rage", "mad", "irritated",
                       "outraged", "hostile"],
            "fear": ["afraid", "scared", "fear", "terrified", "anxious", "worried",
                      "nervous", "panic", "dread"],
            "surprise": ["surprised", "shocked", "unexpected", "wow", "unbelievable",
                          "astonished", "amazed", "sudden"],
            "disgust": ["disgusting", "gross", "vile", "revolting", "nasty", "sick",
                         "repulsive", "awful"],
            "neutral": [],
        }

        words = set(text_lower.split())
        scores = {}
        total = 0

        for emotion, keywords in emotion_keywords.items():
            matches = len(words & set(keywords))
            scores[emotion] = matches
            total += matches

        if total == 0:
            return {e: (0.7 if e == "neutral" else round(0.3 / 6, 4)) for e in EMOTION_LABELS}

        result = {}
        for emotion in EMOTION_LABELS:
            result[emotion] = round(scores.get(emotion, 0) / total, 4)

        remaining = 1.0 - sum(result.values())
        if remaining > 0:
            result["neutral"] = round(result.get("neutral", 0) + remaining, 4)

        return result


# Global singleton
ml_pipeline = MLPipeline()
