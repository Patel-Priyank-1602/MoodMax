"""
Aspect-Based Emotion & Sentiment Analysis (ABSA).
Stateless syntactic clause segmentation using spaCy to extract target
noun phrases and analyze localized sentiment/emotion via DistilBERT.
"""

import re
import logging
from typing import List, Dict, Any
from .correction import get_nlp

logger = logging.getLogger(__name__)

PRONOUNS_SET = {"i", "we", "you", "they", "he", "she", "it", "one", "someone", "anyone", "everyone"}

def clean_aspect_name(chunk_text: str) -> str:
    """Clean determiners, adverbs, and possessives from noun chunk."""
    cleaned = re.sub(
        r"^(?:the|a|an|this|that|these|those|my|our|their|its|your|his|her)\s+",
        "",
        chunk_text.strip(),
        flags=re.IGNORECASE
    )
    cleaned = cleaned.strip()
    if not cleaned or cleaned.lower() in PRONOUNS_SET:
        return ""
    return cleaned.capitalize()


def segment_into_aspects(text: str) -> List[Dict[str, str]]:
    """
    Syntactically segment input text into distinct aspect clauses with target noun phrases.
    Uses regex contrast markers and spaCy dependency parsing.
    """
    nlp = get_nlp()
    if nlp is None:
        return [{"aspect": "General Experience", "clause_text": text.strip()}]

    doc = nlp(text.strip())
    clauses = []

    for sent in doc.sents:
        sent_text = sent.text.strip()
        if not sent_text:
            continue

        # Split on contrastive conjunctions, semicolons, and transitional adverbs
        split_pattern = (
            r"(?:,\s*|\s+)(?:but|however|although|yet|though|while|whereas|on\s+the\s+other\s+hand|except\s+that)\b|;\s*"
        )
        raw_parts = re.split(split_pattern, sent_text, flags=re.IGNORECASE)

        refined_parts = []
        for p in raw_parts:
            # Check for coordinating conjunctions starting new clauses
            subparts = re.split(
                r",\s*and\s+|\s+and\s+(?=(?:the|this|that|my|our|[a-z0-9_-]+)\s+[a-z0-9_-]+\s+(?:is|was|are|were|has|have|had|felt|feels|seems|works?))",
                p,
                flags=re.IGNORECASE
            )
            refined_parts.extend([s.strip() for s in subparts if s.strip()])

        if len(refined_parts) <= 1:
            # Single clause in this sentence
            aspect_target = None
            sent_doc = nlp(sent_text)
            for chunk in sent_doc.noun_chunks:
                if chunk.root.dep_ in {"nsubj", "nsubjpass"} and chunk.root.pos_ not in {"PRON"}:
                    name = clean_aspect_name(chunk.text)
                    if name:
                        aspect_target = name
                        break
            if not aspect_target:
                for ch in sent_doc.noun_chunks:
                    if ch.root.pos_ not in {"PRON"}:
                        name = clean_aspect_name(ch.text)
                        if name:
                            aspect_target = name
                            break

            clauses.append({
                "aspect": aspect_target or "General Experience",
                "clause_text": sent_text
            })
        else:
            for part in refined_parts:
                part_doc = nlp(part)
                aspect_target = None
                for chunk in part_doc.noun_chunks:
                    if chunk.root.dep_ in {"nsubj", "nsubjpass"} and chunk.root.pos_ not in {"PRON"}:
                        name = clean_aspect_name(chunk.text)
                        if name:
                            aspect_target = name
                            break

                if not aspect_target:
                    for ch in part_doc.noun_chunks:
                        if ch.root.pos_ not in {"PRON"}:
                            name = clean_aspect_name(ch.text)
                            if name:
                                aspect_target = name
                                break

                clauses.append({
                    "aspect": aspect_target or "General Experience",
                    "clause_text": part
                })

    if not clauses:
        clauses = [{"aspect": "General Experience", "clause_text": text.strip()}]

    return clauses


def analyze_aspect_sentiment(text: str, pipeline) -> Dict[str, Any]:
    """
    Run Aspect-Based Emotion and Sentiment Analysis (ABSA) on input text.
    1. Segments into aspect clauses.
    2. Runs each clause through the neural pipeline.
    3. Synthesizes overall sentiment (supporting 'Mixed').
    """
    raw_clauses = segment_into_aspects(text)
    aspect_results = []

    sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}

    for item in raw_clauses:
        clause_text = item["clause_text"]
        aspect_name = item["aspect"]

        # Run pipeline inference on clause
        res = pipeline.analyze(clause_text)

        sentiment_label = res.sentiment_label
        if sentiment_label in sentiment_counts:
            sentiment_counts[sentiment_label] += 1

        aspect_results.append({
            "aspect": aspect_name,
            "clause_text": clause_text,
            "sentiment_label": sentiment_label,
            "sentiment_score": res.sentiment_score,
            "dominant_emotion": res.dominant_emotion,
            "emotion_scores": res.emotion_scores,
            "correction_applied": res.correction_applied,
            "correction_reason": res.correction_reason,
        })

    # Synthesize overall sentiment
    has_pos = sentiment_counts["Positive"] > 0
    has_neg = sentiment_counts["Negative"] > 0

    if has_pos and has_neg:
        overall_sentiment = "Mixed"
    elif has_pos:
        overall_sentiment = "Positive"
    elif has_neg:
        overall_sentiment = "Negative"
    else:
        overall_sentiment = "Neutral"

    return {
        "input_text": text,
        "overall_sentiment": overall_sentiment,
        "has_multiple_aspects": len(aspect_results) > 1,
        "aspects": aspect_results,
    }
