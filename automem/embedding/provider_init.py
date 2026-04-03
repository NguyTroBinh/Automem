"""Embedding provider initialization."""

from __future__ import annotations

import os
from typing import Any


def init_embedding_provider(
    *,
    state: Any,
    logger: Any,
    vector_size_config: int,
    embedding_model: str,
) -> None:
    if state.embedding_provider is not None:
        return

    if state.qdrant is None and state.effective_vector_size != vector_size_config:
        state.effective_vector_size = vector_size_config

    model_name = os.getenv("HF_EMBEDDING_MODEL", embedding_model)
    from automem.embedding.hf_local import HuggingFaceLocalProvider

    state.embedding_provider = HuggingFaceLocalProvider(
        model_name=model_name,
        dimension=state.effective_vector_size,
    )
    logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
