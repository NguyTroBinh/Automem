from __future__ import annotations

from typing import Any, Callable


def get_memory_graph(*, state: Any, init_falkordb_fn: Callable[[], None]) -> Any:
    init_falkordb_fn()
    return state.memory_graph


def get_qdrant_client(*, state: Any, init_qdrant_fn: Callable[[], None]) -> Any:
    init_qdrant_fn()
    return state.qdrant
