"""Versioned, registerable name normalizers."""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from anyascii import anyascii

from preprocessing.text_cleaner import clean_name, clean_name_unicode

from .contracts import NameRecord, NormalizedValue, QueryRecord


class NameNormalizer(Protocol):
    name: str

    def normalize(self, record: NameRecord | QueryRecord) -> NormalizedValue: ...


NormalizerFactory = Callable[[Mapping[str, Any]], NameNormalizer]
_NORMALIZERS: dict[str, NormalizerFactory] = {}


def register_normalizer(
    name: str,
    factory: NormalizerFactory,
    *,
    replace: bool = False,
) -> None:
    key = name.strip().lower()
    if not key:
        raise ValueError("Normalizer name cannot be empty")
    if key in _NORMALIZERS and not replace:
        raise ValueError(f"Normalizer already registered: {key}")
    _NORMALIZERS[key] = factory


def create_normalizer(
    name: str,
    options: Mapping[str, Any] | None = None,
) -> NameNormalizer:
    key = name.strip().lower()
    try:
        factory = _NORMALIZERS[key]
    except KeyError as exc:
        raise ValueError(
            f"Unknown normalizer: {name}. Available: {sorted(_NORMALIZERS)}"
        ) from exc
    return factory(dict(options or {}))


class EnglishNameNormalizer:
    name = "english_v1"

    def __init__(self, options: Mapping[str, Any] | None = None):
        options = dict(options or {})
        unknown = set(options) - {"reject_non_ascii"}
        if unknown:
            raise ValueError(f"Unknown english_v1 options: {sorted(unknown)}")
        self.reject_non_ascii = bool(options.get("reject_non_ascii", False))

    def normalize(self, record: NameRecord | QueryRecord) -> NormalizedValue:
        raw_name = _require_string(record.raw_name)
        warnings: list[str] = []
        if self.reject_non_ascii and any(ord(char) > 127 for char in raw_name):
            return NormalizedValue(
                canonical_name="",
                script=detect_script(raw_name),
                variants={"native": ""},
                warnings=("non_ascii_input", "empty_after_normalization"),
            )
        canonical = clean_name(raw_name)
        if not canonical:
            warnings.append("empty_after_normalization")
        return NormalizedValue(
            canonical_name=canonical,
            script="Latn" if canonical else None,
            variants={"native": canonical},
            warnings=tuple(warnings),
        )


class UnicodeNameNormalizer:
    """Unicode-preserving normalizer with Latin diacritic folding."""

    name = "unicode_v1"

    def __init__(self, options: Mapping[str, Any] | None = None):
        options = dict(options or {})
        unknown = set(options) - {"casefold", "transliterate"}
        if unknown:
            raise ValueError(f"Unknown unicode_v1 options: {sorted(unknown)}")
        self.use_casefold = bool(options.get("casefold", True))
        self.use_transliteration = bool(options.get("transliterate", True))

    def normalize(self, record: NameRecord | QueryRecord) -> NormalizedValue:
        raw_name = _require_string(record.raw_name)
        native = clean_name_unicode(raw_name)
        if not self.use_casefold:
            native = _restore_case_without_punctuation(raw_name)
        script = detect_script(native)
        latin_folded = _fold_latin(native)
        canonical = latin_folded if script == "Latn" and latin_folded else native
        warnings = () if canonical else ("empty_after_normalization",)
        variants = {"native": native}
        if latin_folded and latin_folded != native:
            variants["latin_folded"] = latin_folded
        if self.use_transliteration and script != "Latn":
            transliterated = clean_name(anyascii(native))
            if transliterated and transliterated != canonical:
                variants["latin_transliterated"] = transliterated
                compact = transliterated.replace(" ", "")
                if compact != transliterated:
                    variants["latin_compact"] = compact
        return NormalizedValue(
            canonical_name=canonical,
            script=script,
            variants=variants,
            warnings=warnings,
        )


def _require_string(value: Any) -> str:
    if not isinstance(value, str):
        raise TypeError(f"Name must be a string, got {type(value).__name__}")
    return value


def detect_script(value: str) -> str | None:
    scripts: set[str] = set()
    for char in value:
        codepoint = ord(char)
        if "a" <= char.lower() <= "z" or "LATIN" in unicodedata.name(char, ""):
            scripts.add("Latn")
        elif 0x4E00 <= codepoint <= 0x9FFF:
            scripts.add("Han")
        elif 0x3040 <= codepoint <= 0x30FF:
            scripts.add("Jpan")
        elif 0xAC00 <= codepoint <= 0xD7AF:
            scripts.add("Kore")
        elif 0x0600 <= codepoint <= 0x06FF:
            scripts.add("Arab")
        elif (
            0x1000 <= codepoint <= 0x109F
            or 0xA9E0 <= codepoint <= 0xA9FF
            or 0xAA60 <= codepoint <= 0xAA7F
        ):
            scripts.add("Mymr")
        elif 0x0900 <= codepoint <= 0x097F:
            scripts.add("Deva")
        elif 0x0E00 <= codepoint <= 0x0E7F:
            scripts.add("Thai")
    if not scripts:
        return None
    if len(scripts) == 1:
        return next(iter(scripts))
    return "+".join(sorted(scripts))


def _fold_latin(value: str) -> str:
    replacements = str.maketrans(
        {
            "ł": "l",
            "đ": "d",
            "ð": "d",
            "þ": "th",
            "æ": "ae",
            "œ": "oe",
            "ø": "o",
            "ß": "ss",
        }
    )
    decomposed = unicodedata.normalize("NFKD", value.translate(replacements))
    without_marks = "".join(
        char for char in decomposed if unicodedata.category(char)[0] != "M"
    )
    return clean_name(without_marks)


def _restore_case_without_punctuation(value: str) -> str:
    value = unicodedata.normalize("NFKC", value).strip()
    value = "".join(
        char
        for char in value
        if char.isspace() or unicodedata.category(char)[0] in {"L", "M"}
    )
    return " ".join(value.split())


register_normalizer("english_v1", lambda options: EnglishNameNormalizer(options))
register_normalizer("unicode_v1", lambda options: UnicodeNameNormalizer(options))
