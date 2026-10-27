"""Shared multilingual name normalisation for the RLdata10000 comparison.

Both schemes must see exactly the same strings before they encode them, otherwise
the comparison measures the normaliser instead of the matching algorithm.

The RLdata10000 corpus is German and stores names in ASCII transliteration
(``MUELLER``).  A native query may arrive as ``MÜLLER``.  The canonical form
therefore applies, in order:

1. Unicode NFKC + casefold;
2. German transliteration ``ä→ae ö→oe ü→ue ß→ss``;
3. Unicode combining-mark removal (handles é, ñ, ... that are not German);
4. ``anyascii`` for non-Latin scripts (Cyrillic, Greek, Han, ...);
5. punctuation removal and whitespace collapse.

``name_variants`` additionally returns the diacritic-folded form (``ü→u``)
and the pure ``anyascii`` form.  The runners use the canonical form; the
multilingual ablation may search all variants.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from anyascii import anyascii

_GERMAN = str.maketrans(
    {
        "ä": "ae",
        "ö": "oe",
        "ü": "ue",
        "ß": "ss",
        "Ä": "ae",
        "Ö": "oe",
        "Ü": "ue",
    }
)

_NON_LETTER = re.compile(r"[^a-z ]+")


def _clean_ascii(value: str) -> str:
    value = value.lower()
    value = _NON_LETTER.sub(" ", value)
    return re.sub(r"\s+", " ", value).strip()


def _strip_marks(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    return "".join(
        char for char in decomposed if unicodedata.category(char)[0] != "M"
    )


def normalize_name(value: object) -> str:
    """Canonical ASCII form shared by both schemes."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = text.translate(_GERMAN)
    text = _strip_marks(text)
    if any(ord(char) > 127 for char in text):
        text = anyascii(text)
    return _clean_ascii(text)


def name_variants(value: object) -> list[str]:
    """Ordered, de-duplicated normalised variants for multilingual search."""
    if value is None:
        return []
    raw = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    variants: list[str] = []

    german = _clean_ascii(_strip_marks(raw.translate(_GERMAN)))
    variants.append(german)

    folded = _clean_ascii(_strip_marks(raw))
    variants.append(folded)

    transliterated = _clean_ascii(anyascii(raw))
    variants.append(transliterated)

    result: list[str] = []
    seen: set[str] = set()
    for item in variants:
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def unique_variants(values: Iterable[object]) -> list[str]:
    """Return all distinct normalised variants while preserving order."""
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        for variant in name_variants(value):
            if variant not in seen:
                seen.add(variant)
                output.append(variant)
    return output
