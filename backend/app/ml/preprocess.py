"""
Text preprocessing for social media text.
Cleans, normalizes, and prepares text for ML model input.
Includes emoji translation into semantic text descriptions.
"""

import re
import unicodedata


def expand_emojis(text: str) -> str:
    """Convert raw emojis into readable semantic text tokens."""
    try:
        import emoji
        demojized = emoji.demojize(text)
        # Convert :smiling_face_with_heart-eyes: -> smiling face with heart eyes
        expanded = re.sub(
            r':([a-zA-Z0-9_\-]+):',
            lambda m: ' ' + m.group(1).replace('_', ' ').replace('-', ' ') + ' ',
            demojized
        )
        return re.sub(r'\s+', ' ', expanded).strip()
    except Exception:
        return text


def clean_text(text: str) -> str:
    """
    Clean social media text while preserving and expanding emojis into semantic descriptions.
    
    Steps:
      1. Strip leading/trailing whitespace
      2. Expand emojis into meaningful words
      3. Remove URLs
      4. Remove @mentions (keep the text readable)
      5. Remove hashtag symbols (keep the word)
      6. Normalize unicode characters
      7. Collapse multiple whitespace
      8. Handle HTML entities
    """
    if not text or not text.strip():
        return ""

    # Strip whitespace
    text = text.strip()

    # Expand emojis into text representation
    text = expand_emojis(text)

    # Remove HTML entities
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'&nbsp;', ' ', text)

    # Remove URLs
    text = re.sub(
        r'https?://\S+|www\.\S+',
        '',
        text
    )

    # Remove @mentions but keep readability
    text = re.sub(r'@\w+', '', text)

    # Remove hashtag symbol but keep the word
    text = re.sub(r'#(\w+)', r'\1', text)

    # Remove Reddit-specific markers
    text = re.sub(r'\[.*?\]', '', text)  # [deleted], [removed], etc.
    text = re.sub(r'/r/\w+', '', text)   # subreddit references
    text = re.sub(r'/u/\w+', '', text)   # user references

    # Normalize unicode (NFC form)
    text = unicodedata.normalize('NFC', text)

    # Collapse multiple whitespace into single space
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def is_valid_text(text: str) -> bool:
    """Check if text is valid for analysis (non-empty after cleaning)."""
    cleaned = clean_text(text)
    return len(cleaned) > 0


def truncate_text(text: str, max_length: int = 512) -> str:
    """Truncate text to max character length (for display/storage)."""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."
