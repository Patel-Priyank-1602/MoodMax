"""
Part-of-Speech (POS) Tagging and Syntactic Analysis Module.
Extracts token-level universal POS tags, fine-grained Penn Treebank tags,
dependency relations, lemmas, and sentiment-syntactic roles using spaCy.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from .correction import get_nlp

logger = logging.getLogger(__name__)

# Universal POS tag human-readable mapping & color scheme
UPOS_METADATA = {
    "ADJ": {
        "label": "Adjective",
        "description": "Qualifies or describes a noun (e.g., beautiful, horrible)",
        "color": "#8B5CF6",     # Purple / Violet
        "bg": "#F5F3FF",
        "border": "#DDD6FE",
        "role": "descriptor",
    },
    "ADV": {
        "label": "Adverb",
        "description": "Modifies a verb, adjective, or clause (e.g., very, quickly)",
        "color": "#F59E0B",     # Amber / Gold
        "bg": "#FFFBEB",
        "border": "#FDE68A",
        "role": "descriptor",
    },
    "NOUN": {
        "label": "Noun",
        "description": "Person, place, thing, or concept (e.g., movie, service)",
        "color": "#267F4B",     # Brand Emerald / Green
        "bg": "#EAF5EE",
        "border": "#A7F3D0",
        "role": "target",
    },
    "PROPN": {
        "label": "Proper Noun",
        "description": "Named entity or specific individual/brand (e.g., Google, Paris)",
        "color": "#059669",     # Forest Green
        "bg": "#ECFDF5",
        "border": "#6EE7B7",
        "role": "target",
    },
    "VERB": {
        "label": "Verb",
        "description": "Expresses action, occurrence, or state of being (e.g., recommend, broke)",
        "color": "#3B82F6",     # Blue
        "bg": "#EFF6FF",
        "border": "#BFDBFE",
        "role": "action",
    },
    "AUX": {
        "label": "Auxiliary Verb",
        "description": "Helping verb (e.g., was, is, have, should)",
        "color": "#6366F1",     # Indigo
        "bg": "#EEF2FF",
        "border": "#C7D2FE",
        "role": "action",
    },
    "PRON": {
        "label": "Pronoun",
        "description": "Replaces a noun (e.g., I, you, it, they)",
        "color": "#06B6D4",     # Cyan
        "bg": "#ECFEFF",
        "border": "#A5F3FC",
        "role": "target",
    },
    "DET": {
        "label": "Determiner",
        "description": "Specifies reference of a noun (e.g., the, a, this)",
        "color": "#64748B",     # Slate
        "bg": "#F8FAFC",
        "border": "#E2E8F0",
        "role": "grammatical",
    },
    "ADP": {
        "label": "Preposition",
        "description": "Expresses spatial, temporal, or logical relationship (e.g., in, on, with)",
        "color": "#78716C",     # Stone
        "bg": "#FAFAF9",
        "border": "#E7E5E4",
        "role": "grammatical",
    },
    "CCONJ": {
        "label": "Coordinating Conjunction",
        "description": "Links words or phrases equally (e.g., and, but, or)",
        "color": "#EC4899",     # Pink / Rose
        "bg": "#FDF2F8",
        "border": "#FBCFE8",
        "role": "connector",
    },
    "SCONJ": {
        "label": "Subordinating Conjunction",
        "description": "Introduces dependent clause (e.g., although, because, if)",
        "color": "#DB2777",     # Deep Pink
        "bg": "#FDF2F8",
        "border": "#F472B6",
        "role": "connector",
    },
    "PART": {
        "label": "Particle",
        "description": "Function word associated with verb/negation (e.g., not, 's, to)",
        "color": "#EF4444",     # Red (often negation)
        "bg": "#FEF2F2",
        "border": "#FECACA",
        "role": "modifier",
    },
    "NUM": {
        "label": "Numeral",
        "description": "Number or quantity (e.g., 10, two, million)",
        "color": "#D97706",     # Ochre
        "bg": "#FFFBEB",
        "border": "#FCD34D",
        "role": "modifier",
    },
    "INTJ": {
        "label": "Interjection",
        "description": "Expressive exclamation (e.g., wow, yay, ouch, alas)",
        "color": "#F43F5E",     # Rose Red
        "bg": "#FFF1F2",
        "border": "#FECDD3",
        "role": "descriptor",
    },
    "PUNCT": {
        "label": "Punctuation",
        "description": "Punctuation marks (e.g., ., !, ?, ,)",
        "color": "#94A3B8",     # Muted Slate
        "bg": "#F1F5F9",
        "border": "#CBD5E1",
        "role": "punctuation",
    },
    "SYM": {
        "label": "Symbol",
        "description": "Symbols and emojis (e.g., $, %, 👍, ❤️)",
        "color": "#14B8A6",     # Teal
        "bg": "#F0FDFA",
        "border": "#99F6E4",
        "role": "symbol",
    },
    "X": {
        "label": "Other",
        "description": "Foreign word, abbreviation, or unclassified token",
        "color": "#6B7280",     # Gray
        "bg": "#F9FAFB",
        "border": "#E5E7EB",
        "role": "other",
    },
}

# Syntactic dependency descriptions
DEP_DESCRIPTIONS = {
    "ROOT": "Root sentence predicate",
    "nsubj": "Nominal subject",
    "nsubjpass": "Passive nominal subject",
    "dobj": "Direct object",
    "iobj": "Indirect object",
    "pobj": "Object of preposition",
    "acomp": "Adjectival complement",
    "amod": "Adjectival modifier",
    "advmod": "Adverbial modifier",
    "compound": "Compound noun modifier",
    "conj": "Coordinated conjunct",
    "cc": "Coordinating conjunction",
    "det": "Determiner",
    "prep": "Prepositional modifier",
    "aux": "Auxiliary verb",
    "auxpass": "Passive auxiliary",
    "neg": "Negation modifier",
    "attr": "Attribute predicate",
    "punct": "Punctuation mark",
    "poss": "Possessive modifier",
    "appos": "Appositional modifier",
    "relcl": "Relative clause modifier",
    "advcl": "Adverbial clause modifier",
    "xcomp": "Open clausal complement",
    "ccomp": "Clausal complement with internal subject",
    "mark": "Subordinating clause marker",
    "prt": "Phrasal verb particle",
    "nummod": "Numeric modifier",
}

# Negation words to explicitly classify as negators
NEGATION_WORDS = {"not", "no", "never", "n't", "none", "neither", "nor", "nowhere", "hardly", "barely", "scarcely"}


def clean_tag_explanation(tag: str, spacy_expl: Optional[str]) -> str:
    """Format fine-grained tag explanation into clean, concise string."""
    if not spacy_expl:
        return tag
    # Strip language disclaimers like "(English), other noun-modifier (Chinese)"
    clean = re.sub(r"\(English\)[^,]*,?", "", spacy_expl)
    clean = re.sub(r"\([^)]*\)", "", clean)
    clean = re.sub(r",\s*other[^,]*", "", clean, flags=re.IGNORECASE)
    clean = re.sub(r"\s+", " ", clean).strip().rstrip(",")
    return clean.capitalize() if clean else tag


def analyze_pos_tagging(text: str) -> Dict[str, Any]:
    """
    Perform deep Part-of-Speech tagging and syntactic token analysis on the given text.
    Returns:
      - tokens: list of token dicts with pos, tag, lemma, dep, role, coordinates
      - summary: aggregated statistics, category breakdown, key descriptors & targets
    """
    cleaned_text = text.strip()
    if not cleaned_text:
        return {
            "input_text": text,
            "tokens": [],
            "summary": {
                "total_tokens": 0,
                "total_words": 0,
                "lexical_density": 0.0,
                "categories": [],
                "key_descriptors": [],
                "key_targets": [],
                "sentiment_bearing_count": 0,
            }
        }

    nlp = get_nlp()
    tokens_list: List[Dict[str, Any]] = []

    if nlp is not None:
        doc = nlp(cleaned_text)
        try:
            import spacy
            explain_func = spacy.explain
        except Exception:
            explain_func = lambda x: ""

        for idx, token in enumerate(doc):
            upos = token.pos_ or "X"
            meta = UPOS_METADATA.get(upos, UPOS_METADATA["X"])
            fine_tag = token.tag_ or upos
            raw_expl = explain_func(fine_tag) or ""
            tag_desc = clean_tag_explanation(fine_tag, raw_expl)

            # Determine sentiment syntactic role
            text_lower = token.text.lower()
            if text_lower in NEGATION_WORDS or token.dep_ == "neg":
                role = "negator"
            elif upos in {"ADJ", "ADV", "INTJ"}:
                role = "descriptor"
            elif upos in {"NOUN", "PROPN"}:
                role = "target"
            elif upos in {"VERB", "AUX"}:
                role = "action"
            elif upos in {"CCONJ", "SCONJ"}:
                role = "connector"
            elif upos == "PUNCT":
                role = "punctuation"
            else:
                role = meta.get("role", "grammatical")

            dep_str = token.dep_ or ""
            dep_lbl = DEP_DESCRIPTIONS.get(dep_str, dep_str.replace("_", " ").title() if dep_str else "Unknown")

            tokens_list.append({
                "id": idx,
                "text": token.text,
                "lemma": token.lemma_,
                "pos": upos,
                "pos_label": meta["label"],
                "tag": fine_tag,
                "tag_description": tag_desc,
                "dep": dep_str,
                "dep_label": dep_lbl,
                "head_text": token.head.text,
                "is_stop": token.is_stop,
                "is_punct": token.is_punct,
                "sentiment_role": role,
                "color": meta["color"],
                "bg": meta["bg"],
                "border": meta["border"],
                "start_char": token.idx,
                "end_char": token.idx + len(token.text),
            })
    else:
        # Resilient regex fallback if spaCy is unavailable
        words = re.findall(r"\w+|[^\w\s]", cleaned_text, re.UNICODE)
        offset = 0
        for idx, w in enumerate(words):
            start = cleaned_text.find(w, offset)
            if start == -1:
                start = offset
            end = start + len(w)
            offset = end

            is_punct = not w.isalnum()
            upos = "PUNCT" if is_punct else "NOUN"
            meta = UPOS_METADATA.get(upos, UPOS_METADATA["X"])

            tokens_list.append({
                "id": idx,
                "text": w,
                "lemma": w.lower(),
                "pos": upos,
                "pos_label": meta["label"],
                "tag": upos,
                "tag_description": meta["label"],
                "dep": "punct" if is_punct else "dep",
                "dep_label": "Punctuation" if is_punct else "Dependent",
                "head_text": w,
                "is_stop": False,
                "is_punct": is_punct,
                "sentiment_role": "punctuation" if is_punct else "target",
                "color": meta["color"],
                "bg": meta["bg"],
                "border": meta["border"],
                "start_char": start,
                "end_char": end,
            })

    # Aggregated Summary Calculations
    total_tokens = len(tokens_list)
    word_tokens = [t for t in tokens_list if not t["is_punct"]]
    total_words = len(word_tokens)

    # Content words (Noun, Verb, Adjective, Adverb) for lexical density
    content_words = [t for t in word_tokens if t["pos"] in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}]
    lexical_density = round((len(content_words) / total_words * 100), 1) if total_words > 0 else 0.0

    # Group counts by UPOS
    counts_by_pos: Dict[str, List[str]] = {}
    for t in tokens_list:
        pos_code = t["pos"]
        counts_by_pos.setdefault(pos_code, []).append(t["text"])

    # High-level category ordering for clean UI
    display_order = ["ADJ", "ADV", "NOUN", "PROPN", "VERB", "AUX", "PRON", "DET", "ADP", "CCONJ", "SCONJ", "PART", "NUM", "PUNCT", "SYM", "X"]
    categories = []

    for code in display_order:
        if code in counts_by_pos:
            token_instances = counts_by_pos[code]
            count = len(token_instances)
            pct = round((count / total_tokens * 100), 1) if total_tokens > 0 else 0.0
            meta = UPOS_METADATA.get(code, UPOS_METADATA["X"])
            categories.append({
                "category": meta["label"],
                "pos_code": code,
                "count": count,
                "percentage": pct,
                "color": meta["color"],
                "tokens": list(dict.fromkeys(token_instances)),  # Unique preservation
            })

    # Sentiment-relevant driver extractions
    key_descriptors = list(dict.fromkeys([
        t["text"].lower() for t in tokens_list
        if t["sentiment_role"] == "descriptor" and not t["is_stop"]
    ]))

    key_targets = list(dict.fromkeys([
        t["text"] for t in tokens_list
        if t["sentiment_role"] == "target" and not t["is_stop"] and len(t["text"]) > 1
    ]))

    sentiment_bearing_count = len([t for t in tokens_list if t["sentiment_role"] in {"descriptor", "negator", "target"}])

    return {
        "input_text": cleaned_text,
        "tokens": tokens_list,
        "summary": {
            "total_tokens": total_tokens,
            "total_words": total_words,
            "lexical_density": lexical_density,
            "categories": categories,
            "key_descriptors": key_descriptors,
            "key_targets": key_targets,
            "sentiment_bearing_count": sentiment_bearing_count,
        }
    }
