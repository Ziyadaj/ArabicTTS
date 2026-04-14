"""Arabic text normalization for XTTS fine-tuning.

XTTS tokenizes Arabic directly, so we don't convert to phonemes — we just
normalize to the form the tokenizer expects: stripped diacritics (tashkeel),
collapsed whitespace, and normalized alef/ya/ta-marbuta forms.
"""

from __future__ import annotations

import re
import unicodedata

import pyarabic.araby as araby

_PUNCT_FIXES = {
    "،": ",",
    "؛": ";",
    "؟": "?",
    "٪": "%",
    "٬": ",",
    "٫": ".",
}

_WS_RE = re.compile(r"\s+")


def normalize_arabic(text: str, *, strip_tashkeel: bool = True) -> str:
    """Normalize Arabic text for XTTS input.

    - NFKC unicode normalization
    - optional tashkeel/tatweel strip
    - Arabic punctuation → ASCII equivalents (so XTTS text cleaner handles them)
    - whitespace collapse
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFKC", text)
    text = araby.strip_tatweel(text)
    if strip_tashkeel:
        text = araby.strip_tashkeel(text)

    for src, dst in _PUNCT_FIXES.items():
        text = text.replace(src, dst)

    text = _WS_RE.sub(" ", text).strip()
    return text
