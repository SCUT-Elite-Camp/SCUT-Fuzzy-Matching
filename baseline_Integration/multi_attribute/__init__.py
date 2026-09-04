"""Multi-attribute privacy-preserving matching extension."""

from .config import MultiAttributeConfig
from .encoder import encode_record_vectors, plaintext_similarity
from .model import MatchRecord

__all__ = [
    "MatchRecord",
    "MultiAttributeConfig",
    "encode_record_vectors",
    "plaintext_similarity",
]
