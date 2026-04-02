"""memory_core.py — Single entry point that wires all AutoMem subsystems together.

Usage:
    from automem.memory_core import MemoryCore
    core = MemoryCore(logger=logger)
    core.initialize()
    app.register_blueprint(core.memory_blueprint())
    app.register_blueprint(core.recall_blueprint())
    app.register_blueprint(core.enrichment_blueprint())
"""
from __future__ import annotations

import time
from queue import Empty, Queue
from threading import Event, Thread
from typing import Any, Callable, Dict, List, Optional

from automem.classification.memory_classifier import MemoryClassifier
from automem.config import (
    CLASSIFICATION_MODEL,
    COLLECTION_NAME,
    CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
    CONSOLIDATION_TICK_SECONDS,
    EMBEDDING_MODEL,
    ENRICHMENT_ENABLE_SUMMARIES,
    ENRICHMENT_FAILURE_BACKOFF_SECONDS,
    ENRICHMENT_IDLE_SLEEP_SECONDS,
    ENRICHMENT_MAX_ATTEMPTS,
    ENRICHMENT_SIMILARITY_LIMIT,
    ENRICHMENT_SIMILARITY_THRESHOLD,
    FALKORDB_PORT,
    GRAPH_NAME,
    VECTOR_SIZE,
    AUTHORABLE_RELATIONS,
    DEFAULT_EXPAND_RELATIONS,
    FILTERABLE_RELATIONS,
    RECALL_RELATION_LIMIT,
    RELATIONSHIP_TYPES,
    normalize_memory_type,
)
from automem.embedding.provider_init import init_embedding_provider
from automem.embedding.runtime_bindings import create_embedding_runtime
from automem.embedding.runtime_helpers import (
    coerce_embedding as _coerce_embedding_raw,
    coerce_importance,
    generate_placeholder_embedding,
    generate_real_embedding as _generate_real_embedding_raw,
    generate_real_embeddings_batch as _generate_real_embeddings_batch_raw,
    normalize_tags,
)
from automem.enrichment.runtime_bindings import create_enrichment_runtime
from automem.enrichment.runtime_worker import (
    enqueue_enrichment as _enqueue_enrichment_raw,
    enrichment_worker as _enrichment_worker_raw,
    init_enrichment_pipeline,
    update_last_accessed,
)
from automem.search.runtime_recall_helpers import (
    _graph_keyword_search,
    _result_passes_filters,
    _vector_filter_only_tag_search,
    _vector_search,
    configure_recall_helpers,
)
from automem.service_runtime import get_memory_graph as _get_memory_graph_rt
from automem.service_runtime import get_qdrant_client as _get_qdrant_client_rt
from automem.service_state import EnrichmentJob, ServiceState
from automem.stores.graph_store import _build_graph_tag_predicate
from automem.stores.runtime_clients import (
    ensure_qdrant_collection,
    init_falkordb,
    init_qdrant,
)
from automem.stores.vector_store import _build_qdrant_tag_filter
from automem.utils.graph import _serialize_node, _summarize_relation_node
from automem.utils.scoring import _parse_metadata_field as parse_metadata_field
from automem.utils.tags import _compute_tag_prefixes as compute_tag_prefixes
from automem.utils.tags import _normalize_tag_list as normalize_tag_list
from automem.utils.tags import _prepare_tag_filters
from automem.utils.text import _extract_keywords as extract_keywords
from automem.utils.time import _normalize_timestamp as normalize_timestamp
from automem.utils.time import _parse_iso_datetime, _parse_time_expression, utc_now
from automem.utils.validation import get_effective_vector_size


class MemoryCore:
    """Wires all AutoMem subsystems and exposes Flask blueprints."""

    def __init__(self, *, logger: Any) -> None:
        self.logger = logger
        self.state = ServiceState()
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        if self._initialized:
            return

        self._init_falkordb()
        self._init_qdrant()
        self._init_embedding_provider()
        self._init_enrichment_pipeline()
        self._init_embedding_pipeline()
        self._configure_recall_helpers()
        self._initialized = True

    def _init_falkordb(self) -> None:
        try:
            from falkordb import FalkorDB
            falkordb_cls = FalkorDB
        except ImportError:
            self.logger.warning("falkordb package not installed")
            return
        init_falkordb(
            state=self.state,
            logger=self.logger,
            falkordb_cls=falkordb_cls,
            graph_name=GRAPH_NAME,
            falkordb_port=FALKORDB_PORT,
        )

    def _init_qdrant(self) -> None:
        try:
            from qdrant_client import QdrantClient
            qdrant_cls = QdrantClient
        except ImportError:
            self.logger.warning("qdrant_client package not installed")
            return

        try:
            from qdrant_client.http import models as qdrant_models
            vector_params_cls = qdrant_models.VectorParams
            distance_enum = qdrant_models.Distance
            payload_schema_type_enum = qdrant_models.PayloadSchemaType
        except Exception:
            vector_params_cls = distance_enum = payload_schema_type_enum = None

        def _ensure_collection() -> None:
            ensure_qdrant_collection(
                state=self.state,
                logger=self.logger,
                collection_name=COLLECTION_NAME,
                vector_size_config=VECTOR_SIZE,
                get_effective_vector_size_fn=get_effective_vector_size,
                vector_params_cls=vector_params_cls,
                distance_enum=distance_enum,
                payload_schema_type_enum=payload_schema_type_enum,
            )

        init_qdrant(
            state=self.state,
            logger=self.logger,
            qdrant_client_cls=qdrant_cls,
            ensure_collection_fn=_ensure_collection,
        )

    def _init_embedding_provider(self) -> None:
        init_embedding_provider(
            state=self.state,
            logger=self.logger,
            vector_size_config=VECTOR_SIZE,
            embedding_model=EMBEDDING_MODEL,
        )

    def _init_enrichment_pipeline(self) -> None:
        init_enrichment_pipeline(
            state=self.state,
            logger=self.logger,
            queue_cls=Queue,
            thread_cls=Thread,
            worker_target=self._enrichment_worker_target,
        )

    def _init_embedding_pipeline(self) -> None:
        self._embedding_runtime = create_embedding_runtime(
            get_state_fn=lambda: self.state,
            logger=self.logger,
            queue_cls=Queue,
            thread_cls=Thread,
            batch_size=32,
            batch_timeout_seconds=0.5,
            empty_exc=Empty,
            sleep_fn=time.sleep,
            time_fn=time.time,
            get_qdrant_client_fn=self.get_qdrant_client,
            get_memory_graph_fn=self.get_memory_graph,
            collection_name=COLLECTION_NAME,
            point_struct_cls=self._point_struct_cls(),
            utc_now_fn=utc_now,
            generate_real_embedding_fn=self.generate_real_embedding,
            generate_real_embeddings_batch_fn=self.generate_real_embeddings_batch,
        )
        self._embedding_runtime.init_embedding_pipeline()

    def _configure_recall_helpers(self) -> None:
        configure_recall_helpers(
            parse_iso_datetime=self._parse_iso_datetime,
            prepare_tag_filters=_prepare_tag_filters,
            build_graph_tag_predicate=_build_graph_tag_predicate,
            build_qdrant_tag_filter=_build_qdrant_tag_filter,
            serialize_node=_serialize_node,
            fetch_relations=self._fetch_relations_wrapper,
            extract_keywords=extract_keywords,
            coerce_embedding=self.coerce_embedding,
            generate_real_embedding=self.generate_real_embedding,
            logger=self.logger,
            collection_name=COLLECTION_NAME,
        )

    def _fetch_relations_wrapper(self, graph: Any, memory_id: str) -> list:
        from automem.search.runtime_relations import fetch_relations
        try:
            return fetch_relations(
                graph=graph,
                memory_id=memory_id,
                relation_limit=10,
                serialize_node_fn=_serialize_node,
                summarize_relation_node_fn=_summarize_relation_node,
                logger=self.logger,
            )
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    def get_memory_graph(self) -> Any:
        return _get_memory_graph_rt(
            state=self.state,
            init_falkordb_fn=self._init_falkordb,
        )

    def get_qdrant_client(self) -> Any:
        return _get_qdrant_client_rt(
            state=self.state,
            init_qdrant_fn=self._init_qdrant,
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _placeholder_embedding(self, content: str) -> List[float]:
        return generate_placeholder_embedding(content, self.state.effective_vector_size)

    def generate_real_embedding(self, content: str) -> List[float]:
        return _generate_real_embedding_raw(
            content,
            init_embedding_provider=lambda: init_embedding_provider(
                state=self.state,
                logger=self.logger,
                vector_size_config=VECTOR_SIZE,
                embedding_model=EMBEDDING_MODEL,
            ),
            state=self.state,
            logger=self.logger,
            placeholder_embedding=self._placeholder_embedding,
        )

    def generate_real_embeddings_batch(self, contents: List[str]) -> List[List[float]]:
        return _generate_real_embeddings_batch_raw(
            contents,
            init_embedding_provider=lambda: init_embedding_provider(
                state=self.state,
                logger=self.logger,
                vector_size_config=VECTOR_SIZE,
                embedding_model=EMBEDDING_MODEL,
            ),
            state=self.state,
            logger=self.logger,
            placeholder_embedding=self._placeholder_embedding,
        )

    def coerce_embedding(self, value: Any) -> Optional[List[float]]:
        return _coerce_embedding_raw(value, self.state.effective_vector_size)

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    @property
    def classifier(self) -> MemoryClassifier:
        if not hasattr(self, "_classifier"):
            self._classifier = MemoryClassifier(
                normalize_memory_type=normalize_memory_type,
                classification_model=CLASSIFICATION_MODEL,
                logger=self.logger,
            )
        return self._classifier

    def memory_classify(self, content: str) -> tuple[str, float]:
        return self.classifier.classify(content)

    # ------------------------------------------------------------------
    # Enrichment
    # ------------------------------------------------------------------

    def _enrichment_worker_target(self) -> None:
        _enrichment_worker_raw(
            state=self.state,
            logger=self.logger,
            enrichment_idle_sleep_seconds=ENRICHMENT_IDLE_SLEEP_SECONDS,
            enrichment_max_attempts=ENRICHMENT_MAX_ATTEMPTS,
            enrichment_failure_backoff_seconds=ENRICHMENT_FAILURE_BACKOFF_SECONDS,
            empty_exc=Empty,
            enrich_memory_fn=self._enrichment_runtime().enrich_memory,
            emit_event_fn=self._emit_event,
            utc_now_fn=utc_now,
            enqueue_enrichment_fn=self.enqueue_enrichment,
            perf_counter_fn=time.perf_counter,
            sleep_fn=time.sleep,
        )

    def _enrichment_runtime(self):
        if not hasattr(self, "_enrichment_rt"):
            from automem.utils.entity_extraction import extract_entities, _slugify
            from automem.utils.text import SEARCH_STOPWORDS

            self._enrichment_rt = create_enrichment_runtime(
                get_memory_graph_fn=self.get_memory_graph,
                get_qdrant_client_fn=self.get_qdrant_client,
                parse_metadata_field_fn=parse_metadata_field,
                normalize_tag_list_fn=normalize_tag_list,
                extract_entities_fn=extract_entities,
                slugify_fn=_slugify,
                compute_tag_prefixes_fn=compute_tag_prefixes,
                classify_memory_fn=self.memory_classify,
                search_stopwords=SEARCH_STOPWORDS,
                enrichment_enable_summaries=ENRICHMENT_ENABLE_SUMMARIES,
                generate_summary_fn=self._generate_summary,
                utc_now_fn=utc_now,
                collection_name=COLLECTION_NAME,
                enrichment_similarity_limit=ENRICHMENT_SIMILARITY_LIMIT,
                enrichment_similarity_threshold=ENRICHMENT_SIMILARITY_THRESHOLD,
                unexpected_response_exc=Exception,
                logger=self.logger,
            )
        return self._enrichment_rt

    def enqueue_enrichment(self, memory_id: str, *, forced: bool = False, attempt: int = 0) -> None:
        _enqueue_enrichment_raw(
            state=self.state,
            memory_id=memory_id,
            forced=forced,
            attempt=attempt,
            enrichment_job_cls=EnrichmentJob,
        )

    def enqueue_embedding(self, memory_id: str, content: str) -> None:
        if hasattr(self, "_embedding_runtime"):
            self._embedding_runtime.enqueue_embedding(memory_id, content)

    def on_access(self, memory_ids: List[str]) -> None:
        update_last_accessed(
            memory_ids=memory_ids,
            get_memory_graph_fn=self.get_memory_graph,
            utc_now_fn=utc_now,
            logger=self.logger,
        )

    # ------------------------------------------------------------------
    # Blueprints
    # ------------------------------------------------------------------

    def memory_blueprint(self):
        from automem.api.memory import create_memory_blueprint_full

        return create_memory_blueprint_full(
            get_memory_graph=self.get_memory_graph,
            get_qdrant_client=self.get_qdrant_client,
            normalize_tags=normalize_tags,
            normalize_tag_list=normalize_tag_list,
            compute_tag_prefixes=compute_tag_prefixes,
            coerce_importance=coerce_importance,
            coerce_embedding=self.coerce_embedding,
            normalize_timestamp=normalize_timestamp,
            utc_now=utc_now,
            serialize_node=_serialize_node,
            parse_metadata_field=parse_metadata_field,
            generate_real_embedding=self.generate_real_embedding,
            enqueue_enrichment=self.enqueue_enrichment,
            enqueue_embedding=self.enqueue_embedding,
            memory_classify=self.memory_classify,
            point_struct=self._point_struct_cls(),
            collection_name=COLLECTION_NAME,
            authorable_relations=AUTHORABLE_RELATIONS,
            relation_types=RELATIONSHIP_TYPES,
            state=self.state,
            logger=self.logger,
            on_access=self.on_access,
            generate_real_embeddings_batch=self.generate_real_embeddings_batch,
        )

    def recall_blueprint(self):
        from automem.api.recall import create_recall_blueprint

        return create_recall_blueprint(
            get_memory_graph=self.get_memory_graph,
            get_qdrant_client=self.get_qdrant_client,
            normalize_tag_list=normalize_tag_list,
            normalize_timestamp=normalize_timestamp,
            parse_time_expression=_parse_time_expression,
            extract_keywords=extract_keywords,
            compute_metadata_score=self._compute_metadata_score,
            result_passes_filters=_result_passes_filters,
            graph_keyword_search=_graph_keyword_search,
            vector_search=_vector_search,
            vector_filter_only_tag_search=_vector_filter_only_tag_search,
            recall_max_limit=100,
            logger=self.logger,
            filterable_relations=FILTERABLE_RELATIONS,
            default_expand_relations=DEFAULT_EXPAND_RELATIONS,
            relation_limit=RECALL_RELATION_LIMIT,
            serialize_node=_serialize_node,
            on_access=self.on_access,
            jit_enrich_fn=self._enrichment_runtime().jit_enrich_lightweight,
        )

    def enrichment_blueprint(self):
        from automem.api.enrichment import create_enrichment_blueprint

        return create_enrichment_blueprint(
            require_admin_token=self._require_admin_token,
            state=self.state,
            enqueue_enrichment=self.enqueue_enrichment,
            max_attempts=ENRICHMENT_MAX_ATTEMPTS,
        )

    # ------------------------------------------------------------------
    # Consolidation
    # ------------------------------------------------------------------

    def start_consolidation_scheduler(self) -> None:
        from automem.consolidation.runtime_scheduler import (
            consolidation_worker,
            init_consolidation_scheduler,
            run_consolidation_tick,
        )
        from automem.consolidation.runtime_helpers import (
            build_scheduler_from_graph,
            persist_consolidation_run,
        )

        def _tick() -> None:
            run_consolidation_tick(
                get_memory_graph_fn=self.get_memory_graph,
                build_scheduler_from_graph_fn=build_scheduler_from_graph,
                persist_consolidation_run_fn=persist_consolidation_run,
                decay_importance_threshold=CONSOLIDATION_DECAY_IMPORTANCE_THRESHOLD,
                emit_event_fn=self._emit_event,
                utc_now_fn=utc_now,
                perf_counter_fn=time.perf_counter,
                logger=self.logger,
            )

        def _worker() -> None:
            consolidation_worker(
                state=self.state,
                logger=self.logger,
                consolidation_tick_seconds=CONSOLIDATION_TICK_SECONDS,
                run_consolidation_tick_fn=_tick,
            )

        init_consolidation_scheduler(
            state=self.state,
            logger=self.logger,
            stop_event_cls=Event,
            thread_cls=Thread,
            worker_target=_worker,
            run_consolidation_tick_fn=_tick,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _point_struct_cls(self) -> Any:
        try:
            from qdrant_client.http.models import PointStruct
            return PointStruct
        except ImportError:
            return None

    @staticmethod
    def _parse_iso_datetime(value: Any) -> Optional[Any]:
        return _parse_iso_datetime(value)

    def _generate_summary(self, content: str, client: Any) -> Optional[str]:
        from automem.utils.entity_extraction import generate_summary
        return generate_summary(content, client)

    def _compute_metadata_score(
        self,
        result: Dict[str, Any],
        query_text: str,
        query_tokens: List[str],
        context_profile: Optional[Dict[str, Any]],
    ) -> tuple[float, Dict[str, float]]:
        from automem.utils.scoring import _compute_metadata_score
        return _compute_metadata_score(result, query_text, query_tokens, context_profile)

    def _emit_event(
        self,
        event_name: str,
        payload: Dict[str, Any],
        utc_now_fn: Callable[[], str],
    ) -> None:
        self.logger.debug("event:%s %s", event_name, payload)

    def _require_admin_token(self) -> None:
        from flask import abort, request
        from automem.config import ADMIN_TOKEN
        if ADMIN_TOKEN:
            token = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
            if token != ADMIN_TOKEN:
                abort(403, description="Admin token required")
