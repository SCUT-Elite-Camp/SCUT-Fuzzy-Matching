"""Name cleaning utilities used before MinHash encoding."""

import re
import unicodedata


def clean_name(name: str) -> str:
    """Normalize a name string for character shingling."""
    if not isinstance(name, str):
        name = str(name)
    name = unicodedata.normalize("NFKC", name)
    name = name.lower().strip()
    name = re.sub(r"[^a-z\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def clean_name_unicode(name: str) -> str:
    """Normalize letters from any Unicode script for character shingling.

    This intentionally keeps letters and combining marks, while applying the
    same punctuation-removal and whitespace behavior as :func:`clean_name`.
    English-only dataset preparation still uses ``clean_name`` so its existing
    contract does not change.
    """
    if not isinstance(name, str):
        name = str(name)
    name = unicodedata.normalize("NFKC", name).casefold().strip()
    name = "".join(
        char
        for char in name
        if char.isspace() or unicodedata.category(char)[0] in {"L", "M"}
    )
    return re.sub(r"\s+", " ", name).strip()
