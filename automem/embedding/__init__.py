"""Embedding provider module for AutoMem.

Backend: HuggingFaceLocalProvider (sentence-transformers, AITeamVN/Vietnamese_Embedding)
"""

from .provider import EmbeddingProvider
from .hf_local import HuggingFaceLocalProvider

__all__ = [
    "EmbeddingProvider",
    "HuggingFaceLocalProvider",
]
