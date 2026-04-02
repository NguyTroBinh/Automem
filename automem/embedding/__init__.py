"""Embedding provider module for AutoMem.

Backends:
- HuggingFaceLocalProvider: AITeamVN/Vietnamese_Embedding on CUDA (default)
- OllamaEmbeddingProvider: Ollama local server (fallback, set EMBEDDING_PROVIDER=ollama)
"""

from .provider import EmbeddingProvider
from .hf_local import HuggingFaceLocalProvider
from .ollama import OllamaEmbeddingProvider

__all__ = [
    "EmbeddingProvider",
    "HuggingFaceLocalProvider",
    "OllamaEmbeddingProvider",
]
