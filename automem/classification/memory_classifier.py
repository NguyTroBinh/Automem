"""Memory classifier using local Ollama LLM."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)


class MemoryClassifier:
    """Classifies memories into specific types using regex patterns + local Ollama LLM."""

    PATTERNS = {
        "Decision": [
            r"quyết định",
            r"chọn .+ thay vì",
            r"đi với",
            r"chọn",
            r"lựa chọn",
            r"sẽ dùng",
            r"đang chọn",
            r"quyết chọn",
        ],
        "Pattern": [
            r"thường",
            r"thông thường",
            r"có xu hướng",
            r"tôi nhận ra",
            r"hay",
            r"thường xuyên",
            r"đều đặn",
            r"nhất quán",
        ],
        "Preference": [
            r"thích hơn",
            r"thích.+hơn",
            r"yêu thích",
            r"luôn dùng",
            r"hơn là",
            r"thay vì",
            r"ưa",
        ],
        "Style": [
            r"viết.+theo.+phong cách",
            r"giao tiếp",
            r"phản hồi",
            r"định dạng theo",
            r"dùng.+giọng",
            r"diễn đạt theo",
        ],
        "Habit": [
            r"\bluôn\b(?!\s+dùng\b)",
            r"mỗi lần",
            r"theo thói quen",
            r"thói quen",
            r"hàng ngày",
            r"hàng tuần",
            r"hàng tháng",
        ],
        "Insight": [
            r"nhận ra",
            r"phát hiện",
            r"học được rằng",
            r"hiểu ra",
            r"tìm ra",
            r"tìm thấy",
            r"bật ra",
        ],
        "Context": [
            r"trong lúc",
            r"trong khi làm",
            r"trong bối cảnh",
            r"khi",
            r"vào lúc đó",
            r"tình huống lúc đó",
        ],
    }

    SYSTEM_PROMPT = """
    You are a memory classification system. Classify each memory into exactly ONE of these types:
    TYPES:
    - **Decision**: Choices made, selected options, what was decided
    - **Pattern**: Recurring behaviors, typical approaches, consistent tendencies
    - **Preference**: Likes/dislikes, favorites, personal tastes
    - **Style**: Communication approach, formatting, tone used
    - **Habit**: Regular routines, repeated actions, schedules
    - **Insight**: Discoveries, learnings, realizations, key findings
    - **Context**: Situational background, what was happening, circumstances
    CONFIDENCE SCORING:
    0.9-1.0 = very obvious single type, 0.7-0.89 = clear but could overlap, 0.5-0.69 = ambiguous between types, 0.3-0.49 = weak signal, mostly guessing.
    OUTPUT FORMAT: 
    Output ONLY this JSON, nothing else, no explanation, no markdown: {"type": "<type>", "confidence": <0.0-1.0>}
    FEW-SHOT:
    {"type": "Context", "confidence": 0.8}
    """

    def __init__(
        self,
        normalize_memory_type: Any,
        classification_model: str | None = None,
        logger: Any = None,
    ) -> None:
        self._normalize_memory_type = normalize_memory_type
        self._logger = logger or logging.getLogger(__name__)
        self._ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
        self._model = (
            classification_model
            or os.getenv("OLLAMA_CLASSIFICATION_MODEL")
        )
        self._timeout = float(os.getenv("OLLAMA_TIMEOUT", "30"))

    def classify(self, content: str, *, use_llm: bool = True) -> tuple[str, float]:
        """Classify memory type; regex first, then Ollama LLM."""
        content_lower = content.lower()

        for memory_type, patterns in self.PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content_lower):
                    matches = sum(1 for p in patterns if re.search(p, content_lower))
                    confidence = min(0.95, 0.6 + matches * 0.1)
                    return memory_type, confidence

        if use_llm:
            try:
                result = self._classify_with_ollama(content)
                if result:
                    return result
            except Exception:
                self._logger.exception("Ollama classification failed, using fallback")

        return "Context", 0.3

    def _classify_with_ollama(self, content: str) -> Optional[tuple[str, float]]:
        url = f"{self._ollama_base_url}/api/chat"
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": content[:1000]},
            ],
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.1,
                "num_predict": 60,
            },
        }

        try:
            response = requests.post(url, json=payload, timeout=self._timeout)
            response.raise_for_status()
            data = response.json()

            raw_content = ((data.get("message")).get("content")).strip()

            if not raw_content:
                self._logger.warning(
                    "Phân loại bộ nhớ: Không có kết quả"
                )
                return None

            result = json.loads(raw_content)
            raw_type = result.get("type", "Context")
            confidence = float(result.get("confidence", 0.7))

            memory_type, was_normalized = self._normalize_memory_type(raw_type)
            if not memory_type:
                self._logger.warning("Ollama trả về type không hợp lệ '%s', dùng Context", raw_type)
                return "Context", 0.5

            if was_normalized and memory_type != raw_type:
                self._logger.debug("Type normalized '%s' -> '%s'", raw_type, memory_type)

            self._logger.info("Ollama phân loại: %s (confidence=%.2f)", memory_type, confidence)
            return memory_type, confidence

        except (json.JSONDecodeError, TypeError) as exc:
            self._logger.warning(
                "Ollama trả về JSON không hợp lệ từ model '%s': %s", self._model, exc
            )
            return None
        except requests.RequestException as exc:
            self._logger.warning("Ollama classification request thất bại: %s", exc)
            return None
