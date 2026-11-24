"""Text cleaning, label mapping and URL normalisation. Does not require torch."""
import re
import unicodedata

import pandas as pd

from configs.config import BTMD_LABEL_MAP, EXCLUDED_BTMD_LABELS

URL_RE = re.compile(r"https?://\S+")
SPACE_RE = re.compile(r"\s+")


def clean_bangla_text(text) -> str:
    """Unicode-normalise, remove URLs and fix whitespace.

    Punctuation, negation, emoji and English words are kept, because sarcasm and
    negation depend on them.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFC", text)
    text = URL_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text)
    return text.strip()


def map_label(raw_label):
    """Map a BTMD label to a project label.

    Returns None for excluded labels and raises ValueError for unknown labels.
    """
    key = str(raw_label).strip()
    if key in BTMD_LABEL_MAP:
        return BTMD_LABEL_MAP[key]
    if key in EXCLUDED_BTMD_LABELS:
        return None
    raise ValueError(f"Unknown BTMD label: {key!r}. Add a mapping in configs/config.py.")


def normalize_url(url):
    """Normalise a URL for comparison: lowercase, drop query/fragment and trailing slash."""
    if url is None or (isinstance(url, float) and pd.isna(url)):
        return None
    u = str(url).strip().lower().split("?")[0].split("#")[0].rstrip("/")
    return u or None
