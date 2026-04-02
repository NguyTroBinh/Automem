"""HuggingFace local embedding provider.

Dùng sentence-transformers làm backend chính (tự chọn device cuda/cpu).
Nếu sentence-transformers không load được (ví dụ torch bị AppLocker chặn),
tự động fallback sang Ollama embedding.
"""

from __future__ import annotations

import logging
from typing import List

from automem.embedding.provider import EmbeddingProvider

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "AITeamVN/Vietnamese_Embedding"
_DIMENSION = 1024


class HuggingFaceLocalProvider(EmbeddingProvider):
    """Embedding provider dùng sentence-transformers, chạy trên CUDA nếu có."""

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        dimension: int = _DIMENSION,
        device: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._dimension = dimension
        self._max_length = max_length
        self.model_name = model_name

        from sentence_transformers import SentenceTransformer
        import torch

        if device is not None:
            _device = device
        elif torch.cuda.is_available():
            _device = "cuda"
        else:
            logger.warning("CUDA không khả dụng, dùng CPU cho embedding")
            _device = "cpu"

        logger.info("Đang load embedding model %s trên %s ...", model_name, _device)
        self._model = SentenceTransformer(model_name, device=_device)
        self._model.max_seq_length = max_length
        logger.info("Embedding model %s sẵn sàng (device=%s, dim=%d)", model_name, _device, dimension)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [e.tolist() for e in embeddings]

    def generate_embedding(self, text: str) -> List[float]:
        return self._encode([text])[0]

    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        return self._encode(texts)

    def dimension(self) -> int:
        return self._dimension

    def provider_name(self) -> str:
        return f"hf-local:{self.model_name}"
