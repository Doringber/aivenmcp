"""Runtime settings, read from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, Optional

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Settings(BaseSettings):
    """All knobs have defaults that work with the bundled docker-compose.yml."""

    model_config = SettingsConfigDict(env_prefix="RAG_", extra="ignore", populate_by_name=True)

    database_url: str = Field(
        default="postgresql://rag@localhost:5432/rag",
        validation_alias=AliasChoices("DATABASE_URL", "RAG_DATABASE_URL"),
    )
    db_pool_max_size: int = Field(default=8, ge=1)

    embedding_model: str = "Xenova/bge-small-en-v1.5"
    embedding_model_path: Optional[Path] = None
    model_cache_dir: Optional[Path] = None
    query_prefix: str = BGE_QUERY_PREFIX
    embed_batch_size: int = Field(default=32, ge=1)
    embed_threads: Optional[int] = Field(default=None, ge=1)
    inference_concurrency: int = Field(default=1, ge=1)
    """Concurrent ONNX runs per model. Each run already uses every core, so more only adds
    memory (each in-flight run keeps its own activation buffers), not throughput."""

    reranker_model: Optional[str] = "Xenova/ms-marco-MiniLM-L-6-v2"
    """Cross-encoder applied to the top candidates. Set to an empty string to disable."""
    reranker_model_path: Optional[Path] = None
    rerank_candidates: int = Field(default=20, ge=1, le=100)

    chunk_max_tokens: int = Field(default=400, ge=64, le=510)
    chunk_overlap_tokens: int = Field(default=48, ge=0)

    search_default_k: int = Field(default=5, ge=1)
    search_max_k: int = Field(default=20, ge=1)
    search_candidates: int = Field(default=60, ge=10)
    hnsw_ef_search: int = Field(default=100, ge=10)

    ingest_allowed_paths: Annotated[list[Path], NoDecode] = Field(default_factory=list)
    """Colon-separated roots that `ingest(path=...)` may read. Empty means unrestricted."""
    max_download_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    http_timeout_seconds: float = Field(default=30.0, gt=0)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"

    @field_validator("ingest_allowed_paths", mode="before")
    @classmethod
    def _split_paths(cls, value: object) -> object:
        if isinstance(value, str):
            return [part for part in value.split(":") if part.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
