from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from aiven_rag_mcp.app import build_service
from aiven_rag_mcp.config import Settings
from aiven_rag_mcp.service import RagService
from tests.support.corpus import write_corpus
from tests.support.fakes import HashingEmbedder

TABLES = ("chunks", "sources", "rag_meta", "schema_migrations")


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set (needs PostgreSQL with pgvector)")
    return url


@pytest.fixture
def clean_database(database_url: str) -> str:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("DROP TABLE IF EXISTS " + ", ".join(TABLES) + " CASCADE")
    return database_url


@pytest.fixture
def settings(clean_database: str) -> Settings:
    return Settings(
        database_url=clean_database,
        chunk_max_tokens=64,
        chunk_overlap_tokens=8,
        reranker_model="",
        search_candidates=20,
    )


@pytest.fixture
def service(settings: Settings) -> Iterator[RagService]:
    rag = build_service(settings, HashingEmbedder(), use_default_reranker=False)
    yield rag
    rag.close()


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    return write_corpus(tmp_path / "repo")
