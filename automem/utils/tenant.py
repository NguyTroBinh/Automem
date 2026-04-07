"""tenant.py — Centralized helpers for multi-tenancy (tenant_id + user_id) filtering."""

from __future__ import annotations

from typing import Any, Dict, Optional


def tenant_where(alias: str = "m") -> str:
    """Return a Cypher WHERE fragment for tenant + user filtering.

    Usage::

        f"MATCH ({alias}:Memory) WHERE {tenant_where(alias)} AND ..."
    """
    return f"{alias}.tenant_id = $tenant_id AND {alias}.user_id = $user_id"


def tenant_params(tenant_id: str, user_id: str) -> Dict[str, str]:
    """Return a params dict snippet for use with Cypher parameterized queries."""
    return {"tenant_id": tenant_id, "user_id": user_id}


def build_qdrant_tenant_filter(
    tenant_id: Optional[str],
    user_id: Optional[str],
    qdrant_models: Any,
) -> Any:
    """Build a Qdrant Filter with must-conditions for tenant + user.

    Returns None if both identifiers are missing.
    """
    conditions = []
    if tenant_id:
        conditions.append(
            qdrant_models.FieldCondition(
                key="tenant_id",
                match=qdrant_models.MatchValue(value=tenant_id),
            )
        )
    if user_id:
        conditions.append(
            qdrant_models.FieldCondition(
                key="user_id",
                match=qdrant_models.MatchValue(value=user_id),
            )
        )
    if not conditions:
        return None
    return qdrant_models.Filter(must=conditions)


def merge_qdrant_filters(*filters: Any) -> Any:
    """Merge multiple Qdrant Filter objects into a single Filter.

    All ``must`` conditions are combined.  ``None`` filters are skipped.
    Returns ``None`` if no conditions remain.
    """
    must: list[Any] = []
    for f in filters:
        if f is None:
            continue
        if hasattr(f, "must") and f.must:
            must.extend(f.must)
        if hasattr(f, "should") and f.should:
            # Wrap should-clauses in a single must entry to preserve semantics
            must.append(f)
    if not must:
        return None

    # Re-use the first filter's class to construct a new Filter
    # We need qdrant_models.Filter — detect it from any existing filter
    for f in filters:
        if f is not None:
            filter_cls = type(f)
            return filter_cls(must=must)

    return None
