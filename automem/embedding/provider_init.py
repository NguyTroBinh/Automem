"""Embedding provider initialization.

Thứ tự ưu tiên:
1. HuggingFace local (sentence-transformers + CUDA) — EMBEDDING_PROVIDER=local hoặc mặc định
2. Ollama                                           — EMBEDDING_PROVIDER=ollama
   hoặc tự động fallback nếu torch/DLL bị chặn bởi Windows AppLocker/WDAC
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# Các loại lỗi cho thấy DLL/torch bị chặn bởi Windows policy
_BLOCKED_INDICATORS = (
    "WinError 4551",
    "Application Control policy",
    "torch_global_deps",
    "OSError",
    "DLL load failed",
)


def _is_dll_blocked(exc: Exception) -> bool:
    msg = str(exc)
    return any(indicator in msg for indicator in _BLOCKED_INDICATORS)


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

    provider_config = (os.getenv("EMBEDDING_PROVIDER", "local") or "local").strip().lower()

    if provider_config == "ollama":
        _init_ollama(state=state, logger=logger, vector_size=state.effective_vector_size)
        return

    # Mặc định: thử HuggingFace local, fallback Ollama nếu torch bị chặn
    model_name = os.getenv("HF_EMBEDDING_MODEL", embedding_model)
    try:
        _init_hf_local(
            state=state,
            logger=logger,
            model_name=model_name,
            dimension=state.effective_vector_size,
        )
    except Exception as exc:
        if _is_dll_blocked(exc):
            logger.warning(
                "torch/DLL bị chặn bởi Windows Application Control policy (%s). "
                "Tự động chuyển sang Ollama embedding. "
                "Để dùng HuggingFace local, hãy thêm thư mục venv vào whitelist policy "
                "hoặc đặt EMBEDDING_PROVIDER=ollama trong .env.",
                type(exc).__name__,
            )
            _init_ollama(state=state, logger=logger, vector_size=state.effective_vector_size)
        else:
            raise


def _init_hf_local(*, state: Any, logger: Any, model_name: str, dimension: int) -> None:
    from automem.embedding.hf_local import HuggingFaceLocalProvider

    state.embedding_provider = HuggingFaceLocalProvider(
        model_name=model_name,
        dimension=dimension,
    )
    logger.info("Embedding provider: %s", state.embedding_provider.provider_name())


def _init_ollama(*, state: Any, logger: Any, vector_size: int) -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model = os.getenv("OLLAMA_EMBEDDING_MODEL", os.getenv("OLLAMA_MODEL", "nomic-embed-text"))
    try:
        timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))
        max_retries = int(os.getenv("OLLAMA_MAX_RETRIES", "2"))
    except ValueError as exc:
        raise RuntimeError(f"OLLAMA_TIMEOUT hoặc OLLAMA_MAX_RETRIES không hợp lệ: {exc}") from exc

    from automem.embedding.ollama import OllamaEmbeddingProvider

    state.embedding_provider = OllamaEmbeddingProvider(
        base_url=base_url,
        model=model,
        dimension=vector_size,
        timeout=timeout,
        max_retries=max_retries,
    )
    logger.info("Embedding provider: %s", state.embedding_provider.provider_name())
