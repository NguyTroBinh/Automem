from __future__ import annotations

from typing import Any, List, Optional

from qdrant_client import models as qdrant_models

from automem.utils.tags import _prepare_tag_filters
from automem.utils.tenant import build_qdrant_tenant_filter, merge_qdrant_filters


def _build_qdrant_tag_filter(
    tags: Optional[List[str]],
    mode: str = "any",
    match: str = "exact",
):
    """Build a Qdrant filter for tag constraints, supporting mode/match semantics.

    Extracted for reuse by Qdrant interactions.
    """
    normalized_tags = _prepare_tag_filters(tags)
    if not normalized_tags:
        return None

    target_key = "tag_prefixes" if match == "prefix" else "tags"
    normalized_mode = "all" if mode == "all" else "any"

    if normalized_mode == "any":
        return qdrant_models.Filter(
            must=[
                qdrant_models.FieldCondition(
                    key=target_key,
                    match=qdrant_models.MatchAny(any=normalized_tags),
                )
            ]
        )

    must_conditions = [
        qdrant_models.FieldCondition(
            key=target_key,
            match=qdrant_models.MatchValue(value=tag),
        )
        for tag in normalized_tags
    ]

    return qdrant_models.Filter(must=must_conditions)


def _build_qdrant_search_filter(
    tags: Optional[List[str]],
    mode: str = "any",
    match: str = "exact",
    *,
    tenant_id: Optional[str] = None,
    user_id: Optional[str] = None,
) -> Any:
    """Build a combined Qdrant filter including tag constraints AND tenant/user isolation.

    This is the preferred entry-point for all search/recall queries.
    """
    tag_filter = _build_qdrant_tag_filter(tags, mode, match)
    tenant_filter = build_qdrant_tenant_filter(tenant_id, user_id, qdrant_models)
    return merge_qdrant_filters(tag_filter, tenant_filter)
