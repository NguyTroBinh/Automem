"""consolidator.py — MemoryConsolidator: học và cập nhật bộ nhớ theo thời gian.

4 task chạy định kỳ:
- decay   (hàng ngày)  : giảm importance của memory cũ ít được truy cập
- creative (hàng tuần) : tổng hợp memories liên quan thành insight mới
- cluster  (hàng tháng): gom nhóm memories tương đồng, tạo meta-memory
- forget   (tắt mặc định): archive/xóa memory có importance quá thấp

Multi-tenancy: mỗi instance được cấp ``tenant_id`` + ``user_id`` và
tất cả Cypher query đều scoped theo cặp giá trị này.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Set, Tuple


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


class MemoryConsolidator:
    """Consolidates memories over time: decay, creative synthesis, clustering, forgetting."""

    # Interval mặc định — bị override bởi apply_scheduler_overrides()
    DEFAULT_INTERVALS: Dict[str, timedelta] = {
        "decay":    timedelta(seconds=86400),    # 1 ngày
        "creative": timedelta(seconds=604800),   # 7 ngày
        "cluster":  timedelta(seconds=2592000),  # 30 ngày
        "forget":   timedelta(seconds=0),        # tắt
    }

    def __init__(
        self,
        graph: Any,
        vector_store: Any,
        *,
        tenant_id: Optional[str] = None,
        user_id: Optional[str] = None,
        delete_threshold: float = 0.0,
        archive_threshold: float = 0.0,
        grace_period_days: int = 90,
        importance_protection_threshold: float = 0.7,
        protected_types: Set[str] | None = None,
        base_decay_rate: float = 0.01,
        importance_floor_factor: float = 0.3,
    ) -> None:
        self.graph = graph
        self.vector_store = vector_store
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.delete_threshold = delete_threshold
        self.archive_threshold = archive_threshold
        self.grace_period_days = grace_period_days
        self.importance_protection_threshold = importance_protection_threshold
        self.protected_types = protected_types or {"Decision", "Insight"}
        self.base_decay_rate = base_decay_rate
        self.importance_floor_factor = importance_floor_factor

        # schedules dict — apply_scheduler_overrides() ghi trực tiếp vào đây
        self.schedules: Dict[str, Dict[str, Any]] = {
            task: {"interval": interval, "last_run": None}
            for task, interval in self.DEFAULT_INTERVALS.items()
        }

    # ------------------------------------------------------------------
    # Tenant helpers
    # ------------------------------------------------------------------

    def _tenant_where(self, alias: str = "m") -> str:
        """Build a Cypher WHERE fragment for the current tenant+user scope."""
        parts: list[str] = []
        if self.tenant_id is not None:
            parts.append(f"{alias}.tenant_id = $tenant_id")
        if self.user_id is not None:
            parts.append(f"{alias}.user_id = $user_id")
        return " AND ".join(parts) if parts else ""

    def _tenant_params(self) -> Dict[str, str]:
        params: Dict[str, str] = {}
        if self.tenant_id is not None:
            params["tenant_id"] = self.tenant_id
        if self.user_id is not None:
            params["user_id"] = self.user_id
        return params

    def _and_tenant(self, alias: str = "m") -> str:
        """Return ' AND <tenant_clause>' or empty string."""
        tw = self._tenant_where(alias)
        return f" AND {tw}" if tw else ""

    # ------------------------------------------------------------------
    # Public interface (được gọi bởi run_consolidation_tick)
    # ------------------------------------------------------------------

    def run_scheduled_tasks(
        self, *, decay_threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """Kiểm tra và chạy các task đến lịch. Trả về list kết quả."""
        now = _utc_now()
        results = []

        task_runners = {
            "decay":    lambda: self._run_decay(decay_threshold),
            "creative": self._run_creative,
            "cluster":  self._run_cluster,
            "forget":   self._run_forget,
        }

        for task, runner in task_runners.items():
            schedule = self.schedules.get(task, {})
            interval: timedelta = schedule.get("interval", timedelta(seconds=0))

            # interval = 0 → task bị tắt
            if interval.total_seconds() <= 0:
                continue

            last_run = _parse_dt(schedule.get("last_run"))
            if last_run is not None and (now - last_run) < interval:
                continue  # chưa đến lịch

            started_at = now.isoformat()
            try:
                steps = runner()
                result = {
                    "mode": task,
                    "success": True,
                    "started_at": started_at,
                    "completed_at": _utc_now().isoformat(),
                    "steps": {task: steps},
                }
            except Exception as exc:
                result = {
                    "mode": task,
                    "success": False,
                    "started_at": started_at,
                    "completed_at": _utc_now().isoformat(),
                    "steps": {},
                    "error": str(exc),
                }

            self.schedules[task]["last_run"] = _utc_now()
            results.append(result)

        return results

    def get_next_runs(self) -> Dict[str, str]:
        """Trả về thời điểm dự kiến chạy tiếp theo của từng task."""
        now = _utc_now()
        next_runs = {}
        for task, schedule in self.schedules.items():
            interval: timedelta = schedule.get("interval", timedelta(seconds=0))
            if interval.total_seconds() <= 0:
                next_runs[task] = "disabled"
                continue
            last_run = _parse_dt(schedule.get("last_run"))
            if last_run is None:
                next_runs[task] = now.isoformat()
            else:
                next_runs[task] = (last_run + interval).isoformat()
        return next_runs

    # ------------------------------------------------------------------
    # Task: Decay — giảm importance theo thời gian
    # ------------------------------------------------------------------

    def _run_decay(self, decay_threshold: Optional[float] = None) -> Dict[str, Any]:
        """
        Giảm importance của memory cũ ít được truy cập.

        Công thức:
            age_days = (now - last_accessed) / 86400
            floor = importance * importance_floor_factor
            new_importance = max(floor, importance - base_decay_rate * age_days)
        """
        now = _utc_now()
        cutoff = (now - timedelta(days=1)).isoformat()

        try:
            result = self.graph.query(
                f"""
                MATCH (m:Memory)
                WHERE m.last_accessed < $cutoff
                  AND coalesce(m.archived, false) = false{self._and_tenant()}
                RETURN m.id, m.importance, m.last_accessed, m.type, m.timestamp
                LIMIT 500
                """,
                {"cutoff": cutoff, **self._tenant_params()},
            )
        except Exception as exc:
            raise RuntimeError(f"Decay query failed: {exc}") from exc

        updated = 0
        skipped = 0
        now_iso = now.isoformat()

        for row in getattr(result, "result_set", []) or []:
            memory_id, importance, last_accessed, mem_type, timestamp = row[:5]

            # Bảo vệ memory quan trọng và type được bảo vệ
            if importance is not None and float(importance) >= self.importance_protection_threshold:
                skipped += 1
                continue
            if mem_type in self.protected_types:
                skipped += 1
                continue

            # Tính age từ last_accessed hoặc timestamp
            ref_time = _parse_dt(last_accessed) or _parse_dt(timestamp) or now
            age_days = max(0.0, (now - ref_time).total_seconds() / 86400.0)

            current = float(importance) if importance is not None else 0.5
            floor = current * self.importance_floor_factor
            new_importance = max(floor, current - self.base_decay_rate * age_days)
            new_importance = round(new_importance, 4)

            if new_importance >= current:
                skipped += 1
                continue

            # Áp dụng decay_threshold: chỉ decay nếu importance > threshold
            if decay_threshold is not None and current <= decay_threshold:
                skipped += 1
                continue

            try:
                self.graph.query(
                    """
                    MATCH (m:Memory {id: $id})
                    SET m.importance = $importance,
                        m.decay_updated_at = $now
                    """,
                    {"id": memory_id, "importance": new_importance, "now": now_iso},
                )
                updated += 1
            except Exception:
                skipped += 1

        return {"updated": updated, "skipped": skipped}

    # ------------------------------------------------------------------
    # Task: Creative — tổng hợp memories thành insight mới
    # ------------------------------------------------------------------

    def _run_creative(self) -> Dict[str, Any]:
        """
        Tìm các cụm memory cùng type có nhiều SIMILAR_TO edges,
        tổng hợp thành một Insight meta-memory mới nếu chưa có.
        """
        try:
            result = self.graph.query(
                f"""
                MATCH (m:Memory)-[:SIMILAR_TO]->(n:Memory)
                WHERE m.type = n.type
                  AND coalesce(m.archived, false) = false
                  AND coalesce(n.archived, false) = false{self._and_tenant()}
                WITH m.type AS mem_type,
                     collect(DISTINCT m.id)[..10] AS ids,
                     collect(DISTINCT m.content)[..5] AS contents,
                     avg(m.importance) AS avg_importance
                WHERE size(ids) >= 3
                RETURN mem_type, ids, contents, avg_importance
                LIMIT 10
                """,
                {**self._tenant_params()},
            )
        except Exception as exc:
            raise RuntimeError(f"Creative query failed: {exc}") from exc

        created = 0
        now_iso = _utc_now().isoformat()

        for row in getattr(result, "result_set", []) or []:
            mem_type, ids, contents, avg_importance = row[:4]

            # Kiểm tra đã có meta-memory cho nhóm này chưa
            try:
                check = self.graph.query(
                    f"""
                    MATCH (m:Memory {{meta: true, source_type: $type}})
                    WHERE m.timestamp > $since{self._and_tenant()}
                    RETURN m.id LIMIT 1
                    """,
                    {
                        "type": mem_type,
                        "since": (
                            _utc_now() - timedelta(days=7)
                        ).isoformat(),
                        **self._tenant_params(),
                    },
                )
                if getattr(check, "result_set", None):
                    continue
            except Exception:
                pass

            # Tạo nội dung tổng hợp
            sample_contents = [str(c) for c in (contents or []) if c][:3]
            summary = (
                f"Tổng hợp {len(ids)} ký ức loại {mem_type}: "
                + " | ".join(c[:80] for c in sample_contents)
            )

            meta_id = str(uuid.uuid4())
            try:
                # Build create params with tenant/user
                create_params: Dict[str, Any] = {
                    "id": meta_id,
                    "content": summary,
                    "type": "Insight",
                    "importance": min(0.9, float(avg_importance or 0.5) + 0.1),
                    "now": now_iso,
                    "source_type": mem_type,
                    "source_ids": list(ids or []),
                    "tags": ["meta-memory", f"type:{mem_type.lower()}", "consolidation"],
                    "tag_prefixes": ["meta-memory", f"type:{mem_type.lower()}", "consolidation"],
                    "metadata": json.dumps({
                        "consolidation": {
                            "task": "creative",
                            "source_count": len(ids or []),
                            "source_type": mem_type,
                        }
                    }),
                }

                # Build SET clause for tenant/user
                tenant_set = ""
                if self.tenant_id is not None:
                    create_params["tenant_id"] = self.tenant_id
                    tenant_set += ",\n                        tenant_id: $tenant_id"
                if self.user_id is not None:
                    create_params["user_id"] = self.user_id
                    tenant_set += ",\n                        user_id: $user_id"

                self.graph.query(
                    f"""
                    CREATE (m:Memory {{
                        id: $id,
                        content: $content,
                        type: $type,
                        importance: $importance,
                        timestamp: $now,
                        updated_at: $now,
                        last_accessed: $now,
                        meta: true,
                        source_type: $source_type,
                        source_ids: $source_ids,
                        tags: $tags,
                        tag_prefixes: $tag_prefixes,
                        processed: false,
                        metadata: $metadata{tenant_set}
                    }})
                    """,
                    create_params,
                )
                for src_id in (ids or [])[:5]:
                    try:
                        self.graph.query(
                            """
                            MATCH (meta:Memory {id: $meta_id})
                            MATCH (src:Memory {id: $src_id})
                            MERGE (meta)-[:DERIVED_FROM {
                                transformation: 'creative_synthesis',
                                confidence: 0.7
                            }]->(src)
                            """,
                            {"meta_id": meta_id, "src_id": src_id},
                        )
                    except Exception:
                        pass
                created += 1
            except Exception:
                pass

        return {"created": created}

    # ------------------------------------------------------------------
    # Task: Cluster — gom nhóm memories tương đồng
    # ------------------------------------------------------------------

    def _run_cluster(self) -> Dict[str, Any]:
        """
        Tìm memories có nhiều SIMILAR_TO edges nhất (hub nodes),
        tạo cluster meta-memory đại diện cho nhóm.
        """
        try:
            result = self.graph.query(
                f"""
                MATCH (m:Memory)-[:SIMILAR_TO]->(n:Memory)
                WHERE coalesce(m.archived, false) = false
                  AND coalesce(m.meta, false) = false{self._and_tenant()}
                WITH m, count(n) AS neighbor_count
                WHERE neighbor_count >= 3
                ORDER BY neighbor_count DESC
                LIMIT 5
                RETURN m.id, m.content, m.type, m.importance, neighbor_count
                """,
                {**self._tenant_params()},
            )
        except Exception as exc:
            raise RuntimeError(f"Cluster query failed: {exc}") from exc

        meta_memories_created = 0
        now_iso = _utc_now().isoformat()

        for row in getattr(result, "result_set", []) or []:
            hub_id, hub_content, hub_type, hub_importance, neighbor_count = row[:5]

            # Lấy các neighbors
            try:
                neighbors_result = self.graph.query(
                    """
                    MATCH (m:Memory {id: $id})-[:SIMILAR_TO]->(n:Memory)
                    WHERE coalesce(n.archived, false) = false
                    RETURN n.id, n.content
                    LIMIT 10
                    """,
                    {"id": hub_id},
                )
                neighbor_ids = [
                    r[0] for r in (getattr(neighbors_result, "result_set", []) or [])
                ]
            except Exception:
                neighbor_ids = []

            # Kiểm tra cluster đã tồn tại chưa
            try:
                check = self.graph.query(
                    f"""
                    MATCH (m:Memory {{cluster_hub: $hub_id}})
                    WHERE true{self._and_tenant()}
                    RETURN m.id LIMIT 1
                    """,
                    {"hub_id": hub_id, **self._tenant_params()},
                )
                if getattr(check, "result_set", None):
                    continue
            except Exception:
                pass

            cluster_id = str(uuid.uuid4())
            cluster_content = (
                f"Cluster {neighbor_count + 1} ký ức liên quan: "
                + str(hub_content or "")[:120]
            )

            try:
                create_params: Dict[str, Any] = {
                    "id": cluster_id,
                    "content": cluster_content,
                    "type": hub_type or "Context",
                    "importance": min(0.85, float(hub_importance or 0.5) + 0.05),
                    "now": now_iso,
                    "hub_id": hub_id,
                    "size": len(neighbor_ids) + 1,
                    "tags": ["cluster", "consolidation", f"type:{(hub_type or 'context').lower()}"],
                    "tag_prefixes": ["cluster", "consolidation"],
                    "metadata": json.dumps({
                        "consolidation": {
                            "task": "cluster",
                            "hub_id": hub_id,
                            "cluster_size": len(neighbor_ids) + 1,
                        }
                    }),
                }

                tenant_set = ""
                if self.tenant_id is not None:
                    create_params["tenant_id"] = self.tenant_id
                    tenant_set += ",\n                        tenant_id: $tenant_id"
                if self.user_id is not None:
                    create_params["user_id"] = self.user_id
                    tenant_set += ",\n                        user_id: $user_id"

                self.graph.query(
                    f"""
                    CREATE (c:Memory {{
                        id: $id,
                        content: $content,
                        type: $type,
                        importance: $importance,
                        timestamp: $now,
                        updated_at: $now,
                        last_accessed: $now,
                        meta: true,
                        cluster_hub: $hub_id,
                        cluster_size: $size,
                        tags: $tags,
                        tag_prefixes: $tag_prefixes,
                        processed: false,
                        metadata: $metadata{tenant_set}
                    }})
                    """,
                    create_params,
                )
                for member_id in [hub_id] + neighbor_ids[:9]:
                    try:
                        self.graph.query(
                            """
                            MATCH (c:Memory {id: $cluster_id})
                            MATCH (m:Memory {id: $member_id})
                            MERGE (m)-[:PART_OF {role: 'cluster_member', context: 'auto_cluster'}]->(c)
                            """,
                            {"cluster_id": cluster_id, "member_id": member_id},
                        )
                    except Exception:
                        pass
                meta_memories_created += 1
            except Exception:
                pass

        return {"meta_memories_created": meta_memories_created}

    # ------------------------------------------------------------------
    # Task: Forget — archive hoặc xóa memory có importance thấp
    # ------------------------------------------------------------------

    def _run_forget(self) -> Dict[str, Any]:
        """
        Archive hoặc xóa memory có importance thấp, đủ cũ,
        không thuộc protected types, không trong grace period.
        """
        if self.delete_threshold <= 0.0 and self.archive_threshold <= 0.0:
            return {"archived": 0, "deleted": 0, "skipped": 0}

        grace_cutoff = (
            _utc_now() - timedelta(days=self.grace_period_days)
        ).isoformat()
        now_iso = _utc_now().isoformat()

        try:
            result = self.graph.query(
                f"""
                MATCH (m:Memory)
                WHERE m.timestamp < $grace_cutoff
                  AND coalesce(m.archived, false) = false
                  AND coalesce(m.meta, false) = false
                  AND m.importance < $max_threshold{self._and_tenant()}
                RETURN m.id, m.importance, m.type
                LIMIT 200
                """,
                {
                    "grace_cutoff": grace_cutoff,
                    "max_threshold": max(
                        self.delete_threshold, self.archive_threshold
                    ),
                    **self._tenant_params(),
                },
            )
        except Exception as exc:
            raise RuntimeError(f"Forget query failed: {exc}") from exc

        archived = 0
        deleted = 0
        skipped = 0

        for row in getattr(result, "result_set", []) or []:
            memory_id, importance, mem_type = row[:3]
            imp = float(importance) if importance is not None else 0.0

            if imp >= self.importance_protection_threshold:
                skipped += 1
                continue
            if mem_type in self.protected_types:
                skipped += 1
                continue

            if self.delete_threshold > 0.0 and imp < self.delete_threshold:
                try:
                    self.graph.query(
                        "MATCH (m:Memory {id: $id}) DETACH DELETE m",
                        {"id": memory_id},
                    )
                    deleted += 1
                except Exception:
                    skipped += 1
            elif self.archive_threshold > 0.0 and imp < self.archive_threshold:
                try:
                    self.graph.query(
                        """
                        MATCH (m:Memory {id: $id})
                        SET m.archived = true, m.archived_at = $now
                        """,
                        {"id": memory_id, "now": now_iso},
                    )
                    archived += 1
                except Exception:
                    skipped += 1
            else:
                skipped += 1

        return {"archived": archived, "deleted": deleted, "skipped": skipped}
