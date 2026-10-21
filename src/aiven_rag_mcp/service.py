"""Ingest pipeline and search orchestration, independent of the MCP transport."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

from aiven_rag_mcp.chunking import Chunker
from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.embeddings import Embedder, EmbedderTokenCounter, Reranker
from aiven_rag_mcp.errors import InvalidInputError, RagError
from aiven_rag_mcp.loaders import DocumentLoader, LoadedSource
from aiven_rag_mcp.models import (
    Chunk,
    DeleteReport,
    DocFormat,
    FailedSource,
    IngestReport,
    RawDocument,
    SearchMode,
    SearchResponse,
    SourceListing,
)
from aiven_rag_mcp.store import PgStore

logger = logging.getLogger(__name__)

CHUNKER_VERSION = "1"
"""Bump when chunking output changes so existing sources are re-embedded on the next ingest."""
EMBED_BATCH_CHUNKS = 64
MAX_REPORTED_SOURCES = 50

ProgressCallback = Callable[[int, str], None]


@dataclass
class _Pending:
    document: RawDocument
    content_hash: str
    chunks: list[Chunk]


class RagService:
    def __init__(
        self,
        settings: Settings,
        embedder: Embedder,
        store: PgStore,
        reranker: Optional[Reranker] = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._loader = DocumentLoader(settings)
        self._chunker = Chunker(
            EmbedderTokenCounter(embedder),
            max_tokens=settings.chunk_max_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )

    @property
    def store(self) -> PgStore:
        return self._store

    def close(self) -> None:
        self._store.close()

    def ingest(
        self,
        *,
        text: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        title: Optional[str] = None,
        source_id: Optional[str] = None,
        collection: Optional[str] = None,
        text_format: DocFormat = "markdown",
        prune: bool = False,
        progress: Optional[ProgressCallback] = None,
    ) -> IngestReport:
        provided = [name for name, value in (("text", text), ("path", path), ("url", url)) if value]
        if len(provided) != 1:
            raise InvalidInputError("Provide exactly one of 'text', 'path' or 'url'.")
        with self._load(text, path, url, title, source_id, collection, text_format) as loaded:
            return self.ingest_loaded(loaded, prune=prune, progress=progress)

    @contextmanager
    def _load(
        self,
        text: Optional[str],
        path: Optional[str],
        url: Optional[str],
        title: Optional[str],
        source_id: Optional[str],
        collection: Optional[str],
        text_format: DocFormat,
    ) -> Iterator[LoadedSource]:
        if text:
            yield self._loader.from_text(
                text, title=title, source_id=source_id, collection=collection, fmt=text_format
            )
        elif path:
            yield self._loader.from_path(path, collection=collection, title=title)
        else:
            assert url is not None
            with self._loader.from_url(url, collection=collection, title=title) as loaded:
                yield loaded

    def ingest_loaded(
        self,
        loaded: LoadedSource,
        *,
        prune: bool = False,
        progress: Optional[ProgressCallback] = None,
    ) -> IngestReport:
        started = time.perf_counter()
        report = IngestReport()
        seen: list[str] = []
        pending: list[_Pending] = []
        processed = 0

        def flush() -> None:
            if pending:
                self._embed_and_store(pending, report)
                pending.clear()

        for document in loaded.documents:
            seen.append(document.uri)
            processed += 1
            try:
                content_hash = self._content_hash(document)
                if self._store.content_hash(document.uri) == content_hash:
                    report.unchanged += 1
                    continue
                chunks = self._chunker.chunk(document.text, document.format, document.title)
                if not chunks:
                    report.empty += 1
                    continue
                pending.append(_Pending(document, content_hash, chunks))
                if sum(len(p.chunks) for p in pending) >= EMBED_BATCH_CHUNKS:
                    flush()
            except RagError as exc:
                report.failed.append(FailedSource(uri=document.uri, error=str(exc)))
            except Exception as exc:
                # One malformed document must not abort a bulk ingest of hundreds of files.
                logger.exception("Failed to process %s", document.uri)
                report.failed.append(
                    FailedSource(uri=document.uri, error=f"{type(exc).__name__}: {exc}")
                )
            if progress is not None and processed % 25 == 0:
                progress(processed, f"Processed {processed} documents")
        flush()

        if prune and loaded.complete_collection and not report.failed:
            report.pruned = self._store.prune(loaded.collection, seen)
        report.seconds = round(time.perf_counter() - started, 3)
        logger.info(
            "Ingest %s: added=%d updated=%d unchanged=%d failed=%d chunks=%d in %.1fs",
            loaded.collection,
            report.added,
            report.updated,
            report.unchanged,
            len(report.failed),
            report.chunks_written,
            report.seconds,
        )
        return report

    def _embed_and_store(self, pending: list[_Pending], report: IngestReport) -> None:
        texts = [chunk.embedding_text for item in pending for chunk in item.chunks]
        try:
            vectors = self._embedder.embed_documents(texts)
        except Exception as exc:
            if len(pending) == 1:
                logger.exception("Embedding failed for %s", pending[0].document.uri)
                report.failed.append(
                    FailedSource(uri=pending[0].document.uri, error=f"embedding failed: {exc}")
                )
                return
            # Retry one document at a time so a single bad document only fails itself.
            for item in pending:
                self._embed_and_store([item], report)
            return
        offset = 0
        for item in pending:
            item_vectors = vectors[offset : offset + len(item.chunks)]
            offset += len(item.chunks)
            try:
                outcome = self._store.replace_source(
                    item.document, item.content_hash, item.chunks, item_vectors
                )
            except RagError as exc:
                report.failed.append(FailedSource(uri=item.document.uri, error=str(exc)))
                continue
            if outcome == "added":
                report.added += 1
            else:
                report.updated += 1
            report.chunks_written += len(item.chunks)
            if len(report.sources) < MAX_REPORTED_SOURCES:
                report.sources.append(item.document.uri)

    def _content_hash(self, document: RawDocument) -> str:
        fingerprint = "|".join(
            [
                CHUNKER_VERSION,
                self._embedder.model_name,
                str(self._settings.chunk_max_tokens),
                str(self._settings.chunk_overlap_tokens),
                document.title,
                document.text,
            ]
        )
        return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()

    def search(
        self,
        query: str,
        *,
        k: Optional[int] = None,
        mode: SearchMode = "hybrid",
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
        rerank: bool = True,
    ) -> SearchResponse:
        query = query.strip()
        if not query:
            raise InvalidInputError("'query' is empty.")
        limit = min(max(k or self._settings.search_default_k, 1), self._settings.search_max_k)
        use_reranker = rerank and self._reranker is not None
        fetch = max(limit, self._settings.rerank_candidates) if use_reranker else limit
        vector = self._embedder.embed_query(query) if mode != "keyword" else None
        hits = self._store.search(
            vector,
            query,
            k=fetch,
            mode=mode,
            candidates=max(self._settings.search_candidates, fetch),
            ef_search=self._settings.hnsw_ef_search,
            collection=collection,
            uri_prefix=uri_prefix,
        )
        if use_reranker and hits:
            assert self._reranker is not None
            scores = self._reranker.score(query, [f"{h.heading}\n\n{h.content}" for h in hits])
            ranked = sorted(zip(scores, hits, strict=True), key=lambda pair: -pair[0])[:limit]
            hits = [
                hit.model_copy(update={"rank": i + 1, "score": round(score, 4)})
                for i, (score, hit) in enumerate(ranked)
            ]
        return SearchResponse(query=query, mode=mode, reranked=use_reranker, results=hits)

    def list_sources(
        self,
        *,
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SourceListing:
        return self._store.list_sources(
            collection=collection,
            uri_prefix=uri_prefix,
            limit=min(max(limit, 1), 500),
            offset=max(offset, 0),
        )

    def delete_source(
        self,
        *,
        uri: Optional[str] = None,
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
    ) -> DeleteReport:
        if not (uri or collection or uri_prefix):
            raise InvalidInputError("Provide 'uri', 'collection' or 'uri_prefix'.")
        sources, chunks = self._store.delete(uri=uri, collection=collection, uri_prefix=uri_prefix)
        return DeleteReport(deleted_sources=sources, deleted_chunks=chunks)
