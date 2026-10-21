"""Wiring: settings -> embedder + database -> RagService."""

from __future__ import annotations

import logging
import threading
from typing import Optional

from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.db import create_pool, migrate, pgvector_version
from aiven_rag_mcp.embeddings import Embedder, FastEmbedEmbedder, FastEmbedReranker, Reranker
from aiven_rag_mcp.service import RagService
from aiven_rag_mcp.store import PgStore

logger = logging.getLogger(__name__)

ITERATIVE_SCAN_MIN_PGVECTOR = (0, 8, 0)


def build_service(
    settings: Settings,
    embedder: Optional[Embedder] = None,
    reranker: Optional[Reranker] = None,
    *,
    use_default_reranker: bool = True,
) -> RagService:
    embedder = embedder or FastEmbedEmbedder.from_settings(settings)
    if reranker is None and use_default_reranker:
        reranker = FastEmbedReranker.from_settings(settings)
    pool = create_pool(settings.database_url, max_size=settings.db_pool_max_size)
    applied = migrate(pool, embedding_model=embedder.model_name, embedding_dim=embedder.dimension)
    if applied:
        logger.info("Applied migrations: %s", ", ".join(applied))
    iterative_scan = pgvector_version(pool) >= ITERATIVE_SCAN_MIN_PGVECTOR
    store = PgStore(pool, iterative_scan=iterative_scan)
    return RagService(settings, embedder, store, reranker)


class LazyService:
    """Builds the service on first use so the MCP handshake never waits on the DB or model.

    A failed build is not cached: the next tool call retries, which lets the server
    recover once Postgres comes up.
    """

    def __init__(self, settings: Settings, embedder: Optional[Embedder] = None) -> None:
        self._settings = settings
        self._embedder = embedder
        self._service: Optional[RagService] = None
        self._lock = threading.Lock()

    def get(self) -> RagService:
        if self._service is None:
            with self._lock:
                if self._service is None:
                    self._service = build_service(self._settings, self._embedder)
        return self._service

    def warm_up_in_background(self) -> None:
        def run() -> None:
            try:
                service = self.get()
                service.search("warm up", k=1)
                logger.info("Service ready")
            except Exception as exc:
                logger.warning("Warm-up failed (will retry on first tool call): %s", exc)

        threading.Thread(target=run, name="rag-warmup", daemon=True).start()
