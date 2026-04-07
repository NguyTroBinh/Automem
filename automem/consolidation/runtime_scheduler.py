from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple


def _get_unique_tenant_user_pairs(graph: Any, logger: Any) -> List[Tuple[str, str]]:
    """Query FalkorDB for all distinct (tenant_id, user_id) pairs."""
    try:
        result = graph.query(
            """
            MATCH (m:Memory)
            WHERE m.tenant_id IS NOT NULL AND m.user_id IS NOT NULL
            RETURN DISTINCT m.tenant_id, m.user_id
            """
        )
        pairs = []
        for row in getattr(result, "result_set", []) or []:
            if len(row) >= 2 and row[0] and row[1]:
                pairs.append((str(row[0]), str(row[1])))
        return pairs
    except Exception:
        logger.exception("Failed to query tenant/user pairs for consolidation")
        return []


def run_consolidation_tick(
    *,
    get_memory_graph_fn: Callable[[], Any],
    build_scheduler_from_graph_fn: Callable[..., Any],
    persist_consolidation_run_fn: Callable[[Any, Dict[str, Any]], None],
    decay_importance_threshold: Optional[float],
    emit_event_fn: Callable[[str, Dict[str, Any], Callable[[], str]], None],
    utc_now_fn: Callable[[], str],
    perf_counter_fn: Callable[[], float],
    logger: Any,
) -> None:
    graph = get_memory_graph_fn()
    if graph is None:
        return

    # --- Per-(tenant, user) consolidation ---
    pairs = _get_unique_tenant_user_pairs(graph, logger)
    if not pairs:
        # Fallback: run once with no tenant/user filter (backward compat)
        pairs = [(None, None)]  # type: ignore[list-item]

    for tenant_id, user_id in pairs:
        scheduler = build_scheduler_from_graph_fn(graph, tenant_id=tenant_id, user_id=user_id)
        if scheduler is None:
            continue

        try:
            results = scheduler.run_scheduled_tasks(decay_threshold=decay_importance_threshold)
            for result in results:
                task_start = perf_counter_fn()
                persist_consolidation_run_fn(graph, result)

                task_type = result.get("mode", "unknown")
                steps = result.get("steps", {})
                affected_count = 0

                if "decay" in steps:
                    affected_count += steps["decay"].get("updated", 0)
                if "creative" in steps:
                    affected_count += steps["creative"].get("created", 0)
                if "cluster" in steps:
                    affected_count += steps["cluster"].get("meta_memories_created", 0)
                if "forget" in steps:
                    affected_count += steps["forget"].get("archived", 0)
                    affected_count += steps["forget"].get("deleted", 0)

                elapsed_ms = int((perf_counter_fn() - task_start) * 1000)
                next_runs = scheduler.get_next_runs()

                emit_event_fn(
                    "consolidation.run",
                    {
                        "task_type": task_type,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "affected_count": affected_count,
                        "elapsed_ms": elapsed_ms,
                        "success": result.get("success", False),
                        "next_scheduled": next_runs.get(task_type, "unknown"),
                        "steps": list(steps.keys()),
                    },
                    utc_now_fn,
                )
        except Exception:
            logger.exception(
                "Consolidation scheduler tick failed for tenant=%s user=%s",
                tenant_id, user_id,
            )


def consolidation_worker(
    *,
    state: Any,
    logger: Any,
    consolidation_tick_seconds: int,
    run_consolidation_tick_fn: Callable[[], None],
) -> None:
    """Background loop that triggers consolidation tasks."""
    logger.info("Consolidation scheduler thread started")
    while state.consolidation_stop_event and not state.consolidation_stop_event.wait(
        consolidation_tick_seconds
    ):
        run_consolidation_tick_fn()


def init_consolidation_scheduler(
    *,
    state: Any,
    logger: Any,
    stop_event_cls: Any,
    thread_cls: Any,
    worker_target: Callable[[], None],
    run_consolidation_tick_fn: Callable[[], None],
) -> None:
    """Ensure the background consolidation scheduler is running."""
    if state.consolidation_thread and state.consolidation_thread.is_alive():
        return

    stop_event = stop_event_cls()
    state.consolidation_stop_event = stop_event
    state.consolidation_thread = thread_cls(
        target=worker_target,
        daemon=True,
        name="consolidation-scheduler",
    )
    state.consolidation_thread.start()
    run_consolidation_tick_fn()
    logger.info("Consolidation scheduler initialized")
