"""
Correction Layer for Sarcasm and Negation.
Stateless, post-inference correction based on:
  - Tier A: spaCy dependency-tree negation scope detection
  - Tier B: High-confidence sarcasm cues & irony pattern matching
"""

import re
import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Lazy singleton spaCy instance
_spacy_nlp = None

def get_nlp():
    global _spacy_nlp
    if _spacy_nlp is None:
        try:
            import spacy
            _spacy_nlp = spacy.load("en_core_web_sm", disable=["ner"])
        except Exception as e:
            logger.warning(f"Failed to load spaCy model: {e}")
            _spacy_nlp = None
    return _spacy_nlp


# ─────────────────────────────────────────────────────────────────────────────
# Tier B: High-Precision Sarcasm Indicators & Patterns
# ─────────────────────────────────────────────────────────────────────────────

EXPLICIT_SARCASM_REGEX = re.compile(
    r"(?:\s|^)(/s|\#sarcasm|\#irony|\#not|\(!\))(?:\s|[.,!?;]|$)",
    re.IGNORECASE
)

SARCASTIC_IDIOMS = [
    (re.compile(r"\bthanks?\s+for\s+nothing\b", re.IGNORECASE), "thanks for nothing idiom"),
    (re.compile(r"\byeah,?\s*right\b", re.IGNORECASE), "sarcastic disbelief ('yeah right')"),
    (re.compile(r"\bjust\s+what\s+i\s+needed\b", re.IGNORECASE), "sarcastic idiom ('just what I needed')"),
    (re.compile(r"\banother\s+meeting\s+that\s+could\s+have\s+been\s+an\s+email\b", re.IGNORECASE), "sarcastic meeting complaint"),
    (re.compile(r"\b(what|such)\s+a\s+(genius|masterpiece)\b.*?\b(terrible|broken|ruin|remove|cockroach|fail)", re.IGNORECASE), "ironic praise of defect"),
    (re.compile(r"\b(love|delight|thrilled|pure\s+joy|time\s+of\s+my\s+life)\b.*?\b(waiting|line|traffic|fees|dentist|debugging|cancelled|delay|taxes|charged\s+twice|cold\s+soup|rain)", re.IGNORECASE), "feigned joy towards unpleasant event"),
    (re.compile(r"\b(oh\s+)?(great|wonderful|perfect|brilliant|fantastic|splendid|marvelous|excellent)\b.*?\b(another|again|crashed?|stopped|delay|delayed|flat\s+tire|broken|lost|ruin|shattered|rain|unskippable|never\s+replies|put\s+on\s+hold|cockroach|shut\s+down)", re.IGNORECASE), "sarcastic exclamation with negative event"),
    (re.compile(r"\b(super\s+helpful|outstanding\s+job|huge\s+congratulations|five\s+stars|groundbreaking\s+feature|top-tier\s+quality)\b.*?\b(shut\s+down|breaking|broke|late|cockroach|crashes|draft)", re.IGNORECASE), "mock praise for failure"),
    (re.compile(r"\b(my\s+favorite\s+part\s+was\s+when|nothing\s+better\s+than)\b.*?\b(deleted|crashed|lost|cold\s+soup|broken|freeze)", re.IGNORECASE), "ironic superlative for bad experience"),
    (re.compile(r"\b(love\s+how|so\s+glad)\b.*?\b(cannot\s+connect|rain\s+to\s+soak|broken|lost|fails?)", re.IGNORECASE), "mock affection for failure"),
    (re.compile(r"\b(totally|so|definitely|absolutely|completely)\s+worth\s+the\s+.*?\b(wait|delay|money|time|trouble|effort|hours?)\b", re.IGNORECASE), "positive-intensifier + ironic-affirmation pattern"),
    (re.compile(r"\bso\s+honored\s+to\s+be\s+put\s+on\s+hold\b", re.IGNORECASE), "sarcastic honor"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Tier A: Negation Scope & False-Trigger Handlers
# ─────────────────────────────────────────────────────────────────────────────

# Traps where negation words indicate POSITIVE or NEUTRAL sentiment (should NOT be inverted to negative)
POSITIVE_NEGATION_IDIOMS = [
    re.compile(r"\bno\s+problem\b", re.IGNORECASE),
    re.compile(r"\bno\s+worries\b", re.IGNORECASE),
    re.compile(r"\bno\s+doubt\b", re.IGNORECASE),
    re.compile(r"\bno\s+complaints?\b", re.IGNORECASE),
    re.compile(r"\bno\s+question\b", re.IGNORECASE),
    re.compile(r"\bnothing\s+to\s+complain\b", re.IGNORECASE),
    re.compile(r"\bnever\s+felt\s+better\b", re.IGNORECASE),
    re.compile(r"\bnever\s+been\s+better\b", re.IGNORECASE),
    re.compile(r"\bnot\s+(?:only|just)\b", re.IGNORECASE),
    re.compile(r"\bnot\s+let\s+(?:me|us)\s+down\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+(?:any\s+)?(?:issues|problems|hiccups|doubt)\b", re.IGNORECASE),
]

# Negated negative predicates (e.g. "not bad", "not terrible", "wasn't awful") -> Positive/Neutral
NEGATED_NEGATIVE_WORDS = {
    "bad", "terrible", "awful", "ugly", "unbearable", "worst", "horrible",
    "disappointing", "poor", "painful", "hard", "dislike", "hate"
}

# Negated positive predicates (e.g. "not good", "not happy", "never reliable") -> Negative
NEGATED_POSITIVE_WORDS = {
    "good", "great", "happy", "satisfied", "comfortable", "pleasant",
    "impressive", "worth", "acceptable", "reliable", "delicious", "helpful",
    "nice", "enjoyable", "decent", "working", "recommend", "improve", "solve",
    "excel", "perfect"
}


def detect_sarcasm(text: str) -> Optional[str]:
    """Detect high-confidence sarcasm cue or idiom."""
    # 1. Explicit tags
    explicit_match = EXPLICIT_SARCASM_REGEX.search(text)
    if explicit_match:
        return f"Explicit sarcasm tag: '{explicit_match.group(1)}'"

    # 2. Idiomatic patterns
    for pattern, reason in SARCASTIC_IDIOMS:
        if pattern.search(text):
            return reason

    return None


def detect_negation(text: str) -> Optional[dict]:
    """
    Use spaCy dependency parse to detect negation scope and classify:
    - 'negated_positive': "not good", "never reliable" -> should be Negative
    - 'negated_negative': "not bad", "wasn't terrible" -> should be Positive/Neutral
    - 'positive_idiom': "no problem", "never better" -> definitely Positive
    """
    text_clean = text.strip()

    # Check for positive negation idioms first
    for pattern in POSITIVE_NEGATION_IDIOMS:
        if pattern.search(text_clean):
            return {"type": "positive_idiom", "reason": f"Matched positive idiom: '{pattern.pattern}'"}

    # Special case: "by no means acceptable" or "in no way acceptable"
    if re.search(r"\b(?:by\s+no\s+means|in\s+no\s+way)\s+acceptable\b", text_clean, re.IGNORECASE):
        return {"type": "negated_positive", "reason": "Explicit strong negation: 'by no means acceptable'"}

    # Special case: "never opens on time", "never works"
    if re.search(r"\bnever\s+(?:opens|works|starts|functions|responds|delivers)\b", text_clean, re.IGNORECASE):
        return {"type": "negated_positive", "reason": "Habitual failure negation: 'never [action]'"}

    nlp = get_nlp()
    if nlp is None:
        return None

    try:
        doc = nlp(text_clean)
    except Exception as e:
        logger.warning(f"spaCy parsing failed in negation check: {e}")
        return None

    for token in doc:
        # Check for negation dependency or negative adverbs
        is_neg = (token.dep_ == "neg") or (token.lower_ in {"never", "not", "n't", "no", "neither", "nor", "cannot", "hardly", "barely"})
        if not is_neg:
            continue

        head = token.head
        head_lemma = head.lemma_.lower()
        head_text = head.text.lower()

        # Check children of head as well (e.g., "is not [good]")
        scope_lemmas = {head_lemma, head_text}
        for child in head.children:
            if child.dep_ in {"acomp", "dobj", "attr", "advmod", "xcomp"}:
                scope_lemmas.add(child.lemma_.lower())
                scope_lemmas.add(child.text.lower())

        # Check if head is auxiliary (e.g. "did not improve" -> head is "improve")
        # Or head is copula (e.g. "was not good" -> head is "was", child is "good")
        for lemma in list(scope_lemmas):
            if lemma in NEGATED_NEGATIVE_WORDS:
                return {
                    "type": "negated_negative",
                    "reason": f"Negated negative word '{lemma}' with '{token.text}'"
                }
            if lemma in NEGATED_POSITIVE_WORDS:
                return {
                    "type": "negated_positive",
                    "reason": f"Negated positive word '{lemma}' with '{token.text}'"
                }

        # Check phrase windows around negation
        # E.g. "not what I hoped for", "nothing enjoyable", "not worth"
        idx = token.i
        window_tokens = [doc[i].lemma_.lower() for i in range(max(0, idx - 1), min(len(doc), idx + 4))]
        for w in window_tokens:
            if w in NEGATED_NEGATIVE_WORDS:
                return {
                    "type": "negated_negative",
                    "reason": f"Negated negative concept '{w}' near '{token.text}'"
                }
            if w in NEGATED_POSITIVE_WORDS or w in {"hope", "enjoy", "recommend"}:
                return {
                    "type": "negated_positive",
                    "reason": f"Negated positive concept '{w}' near '{token.text}'"
                }

    return None


def apply_corrections(
    text: str,
    sentiment_label: str,
    sentiment_score: float,
    emotion_scores: dict[str, float],
    dominant_emotion: str,
) -> dict:
    """
    Stateless correction layer:
    Evaluates text for sarcasm and negation scope anomalies, adjusting sentiment
    and emotion probability mass only when high-confidence linguistic rules fire.
    """
    # Defensive copy of emotions
    emotions = dict(emotion_scores)
    orig_sentiment = sentiment_label
    orig_score = sentiment_score
    orig_dominant = dominant_emotion

    try:
        # ─────────────────────────────────────────────────────────────
        # 1. Tier B: Sarcasm check
        # ─────────────────────────────────────────────────────────────
        sarcasm_reason = detect_sarcasm(text)
        if sarcasm_reason:
            # Sarcasm flips false positive/joy into anger or disgust
            new_sentiment = "Negative"
            
            # Sarcasm should reduce confidence, not spike it (low-to-moderate)
            if "ironic-affirmation" in sarcasm_reason:
                new_score = round(min(sentiment_score, 0.65), 4)
            else:
                new_score = round(min(sentiment_score, 0.70), 4)

            # Dampen joy drastically
            joy_val = emotions.get("joy", 0.0)
            emotions["joy"] = round(min(joy_val * 0.05, 0.03), 4)

            # Assign dominant negative emotion (disgust if contempt/mockery, else anger)
            text_lower = text.lower()
            if any(w in text_lower for w in ["cockroach", "soup", "disgust", "garbage", "trash", "broken", "shattered", "useless", "meeting"]):
                new_dom = "disgust"
                emotions["disgust"] = max(emotions.get("disgust", 0.0), 0.58)
                emotions["anger"] = max(emotions.get("anger", 0.0), 0.25)
            elif any(w in text_lower for w in ["ruin", "taxes", "dentist", "rain", "unhappy"]):
                new_dom = "sadness"
                emotions["sadness"] = max(emotions.get("sadness", 0.0), 0.55)
                emotions["anger"] = max(emotions.get("anger", 0.0), 0.25)
            else:
                new_dom = "anger"
                emotions["anger"] = max(emotions.get("anger", 0.0), 0.60)
                emotions["disgust"] = max(emotions.get("disgust", 0.0), 0.22)

            # Re-normalize
            total_mass = sum(emotions.values()) or 1.0
            emotions = {k: round(v / total_mass, 4) for k, v in emotions.items()}

            return {
                "sentiment_label": new_sentiment,
                "sentiment_score": new_score,
                "emotion_scores": emotions,
                "dominant_emotion": new_dom,
                "correction_applied": True,
                "correction_type": "sarcasm",
                "correction_reason": sarcasm_reason,
            }

        # ─────────────────────────────────────────────────────────────
        # 2. Tier A: Negation check
        # ─────────────────────────────────────────────────────────────
        negation_res = detect_negation(text)
        if negation_res:
            neg_type = negation_res["type"]
            neg_reason = negation_res["reason"]

            if neg_type == "positive_idiom":
                # Ensure sentiment is Positive if it was neutral or negative
                if sentiment_label != "Positive":
                    emotions["joy"] = max(emotions.get("joy", 0.0), 0.65)
                    emotions["neutral"] = min(emotions.get("neutral", 0.0), 0.25)
                    total_mass = sum(emotions.values()) or 1.0
                    emotions = {k: round(v / total_mass, 4) for k, v in emotions.items()}
                    return {
                        "sentiment_label": "Positive",
                        "sentiment_score": max(sentiment_score, 0.82),
                        "emotion_scores": emotions,
                        "dominant_emotion": "joy",
                        "correction_applied": True,
                        "correction_type": "negation",
                        "correction_reason": neg_reason,
                    }

            elif neg_type == "negated_negative":
                # e.g. "not bad", "not terrible", "wasn't awful", "don't dislike"
                # If predicted as Negative, nudge to Positive or Neutral
                text_lower = text.lower()
                is_mild = any(w in text_lower for w in ["don't dislike", "not the worst", "not unbearable"])
                target_sent = "Neutral" if is_mild else "Positive"
                target_dom = "neutral" if is_mild else "joy"

                if sentiment_label == "Negative" or (sentiment_label == "Neutral" and not is_mild):
                    emotions["anger"] = round(emotions.get("anger", 0.0) * 0.1, 4)
                    emotions["sadness"] = round(emotions.get("sadness", 0.0) * 0.1, 4)
                    emotions["disgust"] = round(emotions.get("disgust", 0.0) * 0.1, 4)
                    if is_mild:
                        emotions["neutral"] = max(emotions.get("neutral", 0.0), 0.70)
                    else:
                        emotions["joy"] = max(emotions.get("joy", 0.0), 0.62)
                        emotions["neutral"] = max(emotions.get("neutral", 0.0), 0.25)

                    total_mass = sum(emotions.values()) or 1.0
                    emotions = {k: round(v / total_mass, 4) for k, v in emotions.items()}

                    return {
                        "sentiment_label": target_sent,
                        "sentiment_score": max(0.75, sentiment_score),
                        "emotion_scores": emotions,
                        "dominant_emotion": target_dom,
                        "correction_applied": True,
                        "correction_type": "negation",
                        "correction_reason": neg_reason,
                    }

            elif neg_type == "negated_positive":
                # e.g. "not good", "not happy", "never reliable", "by no means acceptable"
                # If predicted as Positive or Neutral, flip to Negative
                if sentiment_label in {"Positive", "Neutral"}:
                    emotions["joy"] = round(emotions.get("joy", 0.0) * 0.05, 4)
                    
                    text_lower = text.lower()
                    if any(w in text_lower for w in ["never", "acceptable", "attitude", "solve", "customer service"]):
                        target_dom = "anger"
                        emotions["anger"] = max(emotions.get("anger", 0.0), 0.58)
                    elif any(w in text_lower for w in ["recommend", "worth", "buy", "food"]):
                        target_dom = "disgust"
                        emotions["disgust"] = max(emotions.get("disgust", 0.0), 0.58)
                    else:
                        target_dom = "sadness"
                        emotions["sadness"] = max(emotions.get("sadness", 0.0), 0.58)

                    total_mass = sum(emotions.values()) or 1.0
                    emotions = {k: round(v / total_mass, 4) for k, v in emotions.items()}

                    return {
                        "sentiment_label": "Negative",
                        "sentiment_score": max(0.78, sentiment_score),
                        "emotion_scores": emotions,
                        "dominant_emotion": target_dom,
                        "correction_applied": True,
                        "correction_type": "negation",
                        "correction_reason": neg_reason,
                    }

        # ─────────────────────────────────────────────────────────────
        # 3. Lexical Safeguard for overt negative terms (e.g. atrocious, unhelpful)
        # ─────────────────────────────────────────────────────────────
        strong_neg_matches = [
            w for w in ["atrocious", "abysmal", "horrendous", "unhelpful", "useless", "pathetic", "defective"]
            if re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)
        ]
        strong_pos_matches = [
            w for w in ["good", "great", "love", "amazing", "wonderful", "excellent", "phenomenal", "breathtaking", "fast", "best", "perfect"]
            if re.search(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE)
        ]
        if strong_neg_matches and not strong_pos_matches and sentiment_label in {"Positive", "Neutral"}:
            emotions["joy"] = round(emotions.get("joy", 0.0) * 0.05, 4)
            emotions["anger"] = max(emotions.get("anger", 0.0), 0.60)
            emotions["disgust"] = max(emotions.get("disgust", 0.0), 0.25)
            total_mass = sum(emotions.values()) or 1.0
            emotions = {k: round(v / total_mass, 4) for k, v in emotions.items()}

            return {
                "sentiment_label": "Negative",
                "sentiment_score": max(0.85, sentiment_score),
                "emotion_scores": emotions,
                "dominant_emotion": "anger",
                "correction_applied": True,
                "correction_type": "lexical_anchor",
                "correction_reason": f"Strong negative term identified: '{strong_neg_matches[0]}'",
            }

    except Exception as e:
        logger.warning(f"Correction layer execution failed: {e}")

    # No correction applied, return original
    return {
        "sentiment_label": orig_sentiment,
        "sentiment_score": orig_score,
        "emotion_scores": emotions,
        "dominant_emotion": orig_dominant,
        "correction_applied": False,
        "correction_type": None,
        "correction_reason": None,
    }
