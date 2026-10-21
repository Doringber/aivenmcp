"""End-to-end ingest and search against a real PostgreSQL + pgvector database."""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from aiven_rag_mcp.app import build_service
from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.db import create_pool, migrate
from aiven_rag_mcp.errors import ConfigurationError, InvalidInputError
from aiven_rag_mcp.service import RagService
from tests.support.fakes import HashingEmbedder

pytestmark = pytest.mark.integration

KAFKA_URI_SUFFIX = "docs/products/kafka/reference/advanced-params.rst"
PG_URI_SUFFIX = "docs/products/postgresql/reference/pg-connection-limits.rst"


def ingest_corpus(service: RagService, corpus: Path, **kwargs: object) -> None:
    report = service.ingest(path=str(corpus), collection="docs", **kwargs)  # type: ignore[arg-type]
    assert report.failed == []


class TestIngest:
    def test_ingest_directory_then_search_returns_cited_chunks(
        self, service: RagService, corpus: Path
    ) -> None:
        report = service.ingest(path=str(corpus), collection="docs")
        assert (report.added, report.updated, report.unchanged) == (3, 0, 0)

        response = service.search("auto_create_topics_enable", k=3)
        top = response.results[0]
        assert top.source_uri.endswith(KAFKA_URI_SUFFIX)
        assert "auto_create_topics_enable" in top.content
        assert top.heading.startswith("Advanced parameters for Aiven for Apache Kafka")
        assert top.collection == "docs"

    def test_reingesting_unchanged_content_is_a_no_op(
        self, service: RagService, corpus: Path
    ) -> None:
        ingest_corpus(service, corpus)
        report = service.ingest(path=str(corpus), collection="docs")
        assert (report.added, report.updated, report.unchanged, report.chunks_written) == (
            0,
            0,
            3,
            0,
        )

    def test_changed_document_replaces_all_old_chunks(
        self, service: RagService, corpus: Path
    ) -> None:
        ingest_corpus(service, corpus)
        page = corpus / PG_URI_SUFFIX
        page.write_text("Connection limits\n=================\n\nNow 9999 connections.\n")

        report = service.ingest(path=str(corpus), collection="docs")
        assert (report.updated, report.unchanged) == (1, 2)
        hits = service.search("connections", k=10, uri_prefix=str(page)).results
        assert hits
        assert all("9999" in h.content for h in hits)

    def test_prune_removes_sources_deleted_upstream(
        self, service: RagService, corpus: Path
    ) -> None:
        ingest_corpus(service, corpus)
        (corpus / "README.md").unlink()
        report = service.ingest(path=str(corpus), collection="docs", prune=True)
        assert report.pruned == 1
        assert service.list_sources().total_sources == 2

    def test_one_failing_document_does_not_abort_the_batch(
        self, settings: Settings, corpus: Path
    ) -> None:
        rag = build_service(settings, HashingEmbedder(fail_on="9999"), use_default_reranker=False)
        try:
            (corpus / "bad.md").write_text("# Bad\n\n9999\n")
            report = rag.ingest(path=str(corpus), collection="docs", prune=True)
            assert [f.uri for f in report.failed] == [str(corpus / "bad.md")]
            assert report.added == 3
            assert report.pruned == 0, "prune must not run when the ingest was incomplete"
        finally:
            rag.close()

    def test_text_ingest_with_source_id_is_replaceable(self, service: RagService) -> None:
        service.ingest(text="# Runbook\n\nRestart the broker.", source_id="runbook")
        service.ingest(text="# Runbook\n\nDrain the node first.", source_id="runbook")
        listing = service.list_sources(uri_prefix="runbook")
        assert listing.total_sources == 1
        assert "Drain" in service.search("drain node", k=1).results[0].content

    def test_exactly_one_input_is_required(self, service: RagService) -> None:
        with pytest.raises(InvalidInputError):
            service.ingest()
        with pytest.raises(InvalidInputError):
            service.ingest(text="a", url="https://example.com")


class TestSearch:
    @pytest.mark.parametrize("mode", ["hybrid", "vector", "keyword"])
    def test_every_mode_finds_the_exact_parameter(
        self, service: RagService, corpus: Path, mode: str
    ) -> None:
        ingest_corpus(service, corpus)
        results = service.search("min_insync_replicas", k=3, mode=mode).results  # type: ignore[arg-type]
        assert any(r.source_uri.endswith(KAFKA_URI_SUFFIX) for r in results)

    def test_keyword_branch_tolerates_punctuation_and_stopwords(
        self, service: RagService, corpus: Path
    ) -> None:
        ingest_corpus(service, corpus)
        for query in ["what's the 'max_connections' (limit)?", "the of and", "a:b & c | !d"]:
            service.search(query, k=3, mode="keyword")

    def test_filters_restrict_results(self, service: RagService, corpus: Path) -> None:
        ingest_corpus(service, corpus)
        service.ingest(text="# Other\n\nmax_connections elsewhere", collection="other")
        only_other = service.search("max_connections", k=10, collection="other").results
        assert {h.collection for h in only_other} == {"other"}
        prefix = str(corpus / "docs/products/postgresql")
        by_prefix = service.search("max_connections", k=10, uri_prefix=prefix).results
        assert by_prefix
        assert all(h.source_uri.startswith(prefix) for h in by_prefix)

    def test_uri_prefix_wildcards_are_escaped(self, service: RagService) -> None:
        service.ingest(text="alpha beta", source_id="a_b/doc")
        service.ingest(text="alpha beta", source_id="axb/doc")
        hits = service.search("alpha", k=10, uri_prefix="a_b").results
        assert [h.source_uri for h in hits] == ["a_b/doc"]

    def test_k_is_capped(self, service: RagService, corpus: Path) -> None:
        ingest_corpus(service, corpus)
        assert len(service.search("connections", k=10_000).results) <= 20

    def test_empty_query_is_rejected(self, service: RagService) -> None:
        with pytest.raises(InvalidInputError):
            service.search("   ")


class TestSourcesAndDeletion:
    def test_list_sources_summarizes_collections_and_paginates(
        self, service: RagService, corpus: Path
    ) -> None:
        ingest_corpus(service, corpus)
        page = service.list_sources(limit=2)
        assert page.total_sources == 3
        assert [c.collection for c in page.collections] == ["docs"]
        assert len(page.sources) == 2
        assert len(service.list_sources(limit=2, offset=2).sources) == 1

    def test_delete_cascades_to_chunks(
        self, service: RagService, corpus: Path, clean_database: str
    ) -> None:
        ingest_corpus(service, corpus)
        report = service.delete_source(collection="docs")
        assert report.deleted_sources == 3
        with psycopg.connect(clean_database) as conn:
            row = conn.execute("SELECT count(*) FROM chunks").fetchone()
        assert row == (0,)

    def test_delete_requires_a_filter(self, service: RagService) -> None:
        with pytest.raises(InvalidInputError):
            service.delete_source()


class TestSchema:
    def test_migrations_are_idempotent(self, clean_database: str) -> None:
        pool = create_pool(clean_database, max_size=2)
        try:
            first = migrate(pool, embedding_model="m", embedding_dim=8)
            second = migrate(pool, embedding_model="m", embedding_dim=8)
        finally:
            pool.close()
        assert first == ["001_init.sql"]
        assert second == []

    def test_switching_embedding_model_fails_loudly(self, clean_database: str) -> None:
        pool = create_pool(clean_database, max_size=2)
        try:
            migrate(pool, embedding_model="model-a", embedding_dim=8)
            with pytest.raises(ConfigurationError, match="model-a"):
                migrate(pool, embedding_model="model-b", embedding_dim=16)
        finally:
            pool.close()

    def test_survives_database_restart(self, service: RagService, clean_database: str) -> None:
        """A Postgres restart kills every pooled connection; the next call must still work."""
        service.ingest(text="Pooled connections must be revalidated.", source_id="note")
        with psycopg.connect(clean_database, autocommit=True) as admin:
            admin.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = current_database() AND pid <> pg_backend_pid()"
            )
        assert service.list_sources().total_sources == 1
        assert service.search("revalidated", k=1).results
