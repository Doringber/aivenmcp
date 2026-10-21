"""Data models shared by loaders, chunker, store, service and the MCP layer."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

DocFormat = Literal["markdown", "rst", "text"]
SourceKind = Literal["text", "file", "url", "git"]
SearchMode = Literal["hybrid", "vector", "keyword"]


class RawDocument(BaseModel):
    """A document produced by a loader, before chunking."""

    uri: str
    collection: str
    kind: SourceKind
    title: str
    text: str
    format: DocFormat
    web_url: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    ordinal: int
    heading: str
    content: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        return f"{self.heading}\n\n{self.content}" if self.heading else self.content


class SearchHit(BaseModel):
    rank: int
    score: float = Field(
        description="Relevance score (cross-encoder logit when reranked, else fused RRF score); "
        "higher is better. Only comparable within one response."
    )
    source_uri: str
    title: str
    heading: str = Field(description="Document title and section path of the chunk.")
    web_url: Optional[str] = None
    collection: str
    chunk_ordinal: int
    content: str


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    reranked: bool = False
    results: list[SearchHit]


class SourceInfo(BaseModel):
    uri: str
    title: str
    collection: str
    kind: SourceKind
    web_url: Optional[str] = None
    chunk_count: int
    char_count: int
    updated_at: datetime


class CollectionSummary(BaseModel):
    collection: str
    source_count: int
    chunk_count: int


class SourceListing(BaseModel):
    total_sources: int
    total_chunks: int
    collections: list[CollectionSummary]
    sources: list[SourceInfo]
    offset: int
    limit: int


class FailedSource(BaseModel):
    uri: str
    error: str


class IngestReport(BaseModel):
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    empty: int = 0
    pruned: int = 0
    failed: list[FailedSource] = Field(default_factory=list)
    chunks_written: int = 0
    seconds: float = 0.0
    sources: list[str] = Field(
        default_factory=list, description="URIs of added or updated sources (first 50)."
    )


class DeleteReport(BaseModel):
    deleted_sources: int
    deleted_chunks: int
