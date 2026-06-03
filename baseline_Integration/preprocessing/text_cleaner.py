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

