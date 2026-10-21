"""PostgreSQL + pgvector persistence and hybrid retrieval."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any, Literal, Optional

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool, PoolTimeout

from aiven_rag_mcp.errors import StorageError
from aiven_rag_mcp.models import (
    Chunk,
    CollectionSummary,
    RawDocument,
    SearchHit,
    SearchMode,
    SourceInfo,
    SourceListing,
)

RRF_K = 60
"""Reciprocal Rank Fusion constant from Cormack et al. (2009); damps the weight of top ranks."""

# Turn the question into an OR of its stemmed lexemes: questions are long and an AND of every
# word would rarely match. Quoting each lexeme keeps punctuation from being parsed as operators.
_TSQUERY = """
SELECT to_tsquery(
    'simple', coalesce(string_agg(quote_literal(replace(lex, '\\', '')), ' | '), '')
) AS tsq
FROM unnest(tsvector_to_array(to_tsvector('english', %(query_text)s))) AS lex
"""


def _like_prefix(prefix: str) -> str:
    return prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _filters(
    collection: Optional[str], uri_prefix: Optional[str]
) -> tuple[sql.Composable, dict[str, Any]]:
    clauses: list[sql.Composable] = [sql.SQL("TRUE")]
    params: dict[str, Any] = {}
    if collection:
        clauses.append(sql.SQL("s.collection = %(collection)s"))
        params["collection"] = collection
    if uri_prefix:
        clauses.append(sql.SQL("s.uri LIKE %(uri_prefix)s"))
        params["uri_prefix"] = _like_prefix(uri_prefix)
    return sql.SQL(" AND ").join(clauses), params


class PgStore:
    def __init__(self, pool: ConnectionPool, *, iterative_scan: bool) -> None:
        self._pool = pool
        self._iterative_scan = iterative_scan

    def close(self) -> None:
        self._pool.close()

    @contextmanager
    def _connection(self) -> Iterator[psycopg.Connection[Any]]:
        try:
            with self._pool.connection() as conn:
                yield conn
        except PoolTimeout as exc:
            raise StorageError(f"Cannot connect to PostgreSQL: {exc}") from exc
        except psycopg.OperationalError as exc:
            raise StorageError(f"Database error: {exc}") from exc

    def content_hash(self, uri: str) -> Optional[str]:
        with self._connection() as conn:
            row = conn.execute("SELECT content_hash FROM sources WHERE uri = %s", (uri,)).fetchone()
        return None if row is None else str(row[0])

    def replace_source(
        self,
        document: RawDocument,
        content_hash: str,
        chunks: Sequence[Chunk],
        vectors: Sequence[Sequence[float]],
    ) -> Literal["added", "updated"]:
        """Atomically write a source and all its chunks, replacing any previous version."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        try:
            with self._connection() as conn, conn.transaction():
                row = conn.execute(
                    """
                    INSERT INTO sources (uri, collection, kind, title, web_url, content_hash,
                                         char_count, chunk_count, metadata)
                    VALUES (%(uri)s, %(collection)s, %(kind)s, %(title)s, %(web_url)s, %(hash)s,
                            %(chars)s, %(chunks)s, %(metadata)s)
                    ON CONFLICT (uri) DO UPDATE SET
                        collection = EXCLUDED.collection, kind = EXCLUDED.kind,
                        title = EXCLUDED.title, web_url = EXCLUDED.web_url,
                        content_hash = EXCLUDED.content_hash, char_count = EXCLUDED.char_count,
                        chunk_count = EXCLUDED.chunk_count, metadata = EXCLUDED.metadata,
                        updated_at = now()
                    RETURNING id, (xmax = 0) AS inserted
                    """,
                    {
                        "uri": document.uri,
                        "collection": document.collection,
                        "kind": document.kind,
                        "title": document.title,
                        "web_url": document.web_url,
                        "hash": content_hash,
                        "chars": len(document.text),
                        "chunks": len(chunks),
                        "metadata": Jsonb(document.metadata),
                    },
                ).fetchone()
                assert row is not None
                source_id, inserted = int(row[0]), bool(row[1])
                if not inserted:
                    conn.execute("DELETE FROM chunks WHERE source_id = %s", (source_id,))
                with (
                    conn.cursor() as cur,
                    cur.copy(
                        "COPY chunks (source_id, ordinal, heading, content, token_count, embedding)"
                        " FROM STDIN WITH (FORMAT BINARY)"
                    ) as copy,
                ):
                    copy.set_types(["int8", "int4", "text", "text", "int4", "vector"])
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        copy.write_row(
                            (
                                source_id,
                                chunk.ordinal,
                                chunk.heading,
                                chunk.content,
                                chunk.token_count,
                                list(vector),
                            )
                        )
        except psycopg.Error as exc:
            raise StorageError(f"Failed to store {document.uri}: {exc}") from exc
        return "added" if inserted else "updated"

    def search(
        self,
        query_vector: Optional[Sequence[float]],
        query_text: str,
        *,
        k: int,
        mode: SearchMode,
        candidates: int,
        ef_search: int,
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
    ) -> list[SearchHit]:
        where, params = _filters(collection, uri_prefix)
        params |= {"k": k, "candidates": candidates, "rrf_k": RRF_K, "query_text": query_text}
        ctes: list[sql.Composable] = []
        branches: list[sql.Composable] = []
        if mode in ("hybrid", "vector"):
            if query_vector is None:
                raise ValueError("query_vector is required for vector search")
            params["query_vector"] = list(query_vector)
            ctes.append(
                sql.SQL(
                    """vec AS (
                    SELECT id, row_number() OVER (ORDER BY dist) AS rnk FROM (
                        SELECT c.id, c.embedding <=> %(query_vector)s::vector AS dist
                        FROM chunks c JOIN sources s ON s.id = c.source_id
                        WHERE {where}
                        ORDER BY dist LIMIT %(candidates)s) v)"""
                ).format(where=where)
            )
            branches.append(sql.SQL("SELECT id, rnk FROM vec"))
        if mode in ("hybrid", "keyword"):
            ctes.append(sql.SQL("q AS (" + _TSQUERY + ")"))
            ctes.append(
                sql.SQL(
                    """kw AS (
                    SELECT id, row_number() OVER (ORDER BY score DESC, id) AS rnk FROM (
                        SELECT c.id, ts_rank_cd(c.tsv, q.tsq) AS score
                        FROM chunks c JOIN sources s ON s.id = c.source_id, q
                        WHERE c.tsv @@ q.tsq AND {where}
                        ORDER BY score DESC LIMIT %(candidates)s) t)"""
                ).format(where=where)
            )
            branches.append(sql.SQL("SELECT id, rnk FROM kw"))
        query = sql.SQL(
            """WITH {ctes},
            fused AS (
                SELECT id, sum(1.0 / (%(rrf_k)s + rnk)) AS score
                FROM ({branches}) u GROUP BY id)
            SELECT f.score, c.ordinal, c.heading, c.content,
                   s.uri, s.title, s.web_url, s.collection
            FROM fused f
            JOIN chunks c ON c.id = f.id
            JOIN sources s ON s.id = c.source_id
            ORDER BY f.score DESC, c.id
            LIMIT %(k)s"""
        ).format(ctes=sql.SQL(", ").join(ctes), branches=sql.SQL(" UNION ALL ").join(branches))
        try:
            with self._connection() as conn, conn.transaction():
                conn.execute(
                    "SELECT set_config('hnsw.ef_search', %s, true)",
                    (str(max(ef_search, candidates)),),
                )
                if self._iterative_scan:
                    conn.execute("SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)")
                rows = conn.execute(query, params).fetchall()
        except psycopg.Error as exc:
            raise StorageError(f"Search failed: {exc}") from exc
        return [
            SearchHit(
                rank=i + 1,
                score=round(float(score), 6),
                chunk_ordinal=ordinal,
                heading=heading,
                content=content,
                source_uri=uri,
                title=title,
                web_url=web_url,
                collection=coll,
            )
            for i, (score, ordinal, heading, content, uri, title, web_url, coll) in enumerate(rows)
        ]

    def list_sources(
        self,
        *,
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> SourceListing:
        where, params = _filters(collection, uri_prefix)
        params |= {"limit": limit, "offset": offset}
        with self._connection() as conn:
            summaries = conn.execute(
                sql.SQL(
                    "SELECT s.collection, count(*), coalesce(sum(s.chunk_count), 0) FROM sources s"
                    " WHERE {where} GROUP BY s.collection ORDER BY s.collection"
                ).format(where=where),
                params,
            ).fetchall()
            rows = conn.execute(
                sql.SQL(
                    "SELECT s.uri, s.title, s.collection, s.kind, s.web_url, s.chunk_count,"
                    " s.char_count, s.updated_at FROM sources s WHERE {where}"
                    " ORDER BY s.collection, s.uri LIMIT %(limit)s OFFSET %(offset)s"
                ).format(where=where),
                params,
            ).fetchall()
        collections = [
            CollectionSummary(collection=c, source_count=int(n), chunk_count=int(ch))
            for c, n, ch in summaries
        ]
        return SourceListing(
            total_sources=sum(c.source_count for c in collections),
            total_chunks=sum(c.chunk_count for c in collections),
            collections=collections,
            sources=[
                SourceInfo(
                    uri=uri,
                    title=title,
                    collection=coll,
                    kind=kind,
                    web_url=web_url,
                    chunk_count=chunks,
                    char_count=chars,
                    updated_at=updated,
                )
                for uri, title, coll, kind, web_url, chunks, chars, updated in rows
            ],
            offset=offset,
            limit=limit,
        )

    def delete(
        self,
        *,
        uri: Optional[str] = None,
        collection: Optional[str] = None,
        uri_prefix: Optional[str] = None,
    ) -> tuple[int, int]:
        """Delete matching sources (chunks cascade). Returns (sources, chunks) deleted."""
        clauses: list[sql.Composable] = []
        params: dict[str, Any] = {}
        if uri:
            clauses.append(sql.SQL("uri = %(uri)s"))
            params["uri"] = uri
        if collection:
            clauses.append(sql.SQL("collection = %(collection)s"))
            params["collection"] = collection
        if uri_prefix:
            clauses.append(sql.SQL("uri LIKE %(uri_prefix)s"))
            params["uri_prefix"] = _like_prefix(uri_prefix)
        if not clauses:
            raise ValueError("At least one delete filter is required")
        query = sql.SQL(
            "WITH d AS (DELETE FROM sources WHERE {where} RETURNING chunk_count)"
            " SELECT count(*), coalesce(sum(chunk_count), 0) FROM d"
        ).format(where=sql.SQL(" AND ").join(clauses))
        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
        assert row is not None
        return int(row[0]), int(row[1])

    def prune(self, collection: str, keep_uris: Sequence[str]) -> int:
        """Delete sources of ``collection`` that were not seen in the latest full ingest."""
        with self._connection() as conn:
            cur = conn.execute(
                "DELETE FROM sources WHERE collection = %s AND NOT (uri = ANY(%s))",
                (collection, list(keep_uris)),
            )
        return cur.rowcount
