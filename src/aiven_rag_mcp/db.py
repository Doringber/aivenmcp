"""Connection pool and schema migrations."""

from __future__ import annotations

import logging
from importlib import resources

import psycopg
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool, PoolTimeout

from aiven_rag_mcp.errors import ConfigurationError, StorageError

logger = logging.getLogger(__name__)

_MIGRATION_LOCK_ID = 7_412_001


def _configure_connection(conn: psycopg.Connection) -> None:
    register_vector(conn)


def create_pool(database_url: str, *, max_size: int, open_timeout: float = 30.0) -> ConnectionPool:
    """Open a pool; waits up to ``open_timeout`` seconds for the database to accept connections."""
    _ensure_extension(database_url, open_timeout)
    pool = ConnectionPool(
        database_url,
        min_size=1,
        max_size=max_size,
        configure=_configure_connection,
        # Validate on checkout: after a database restart every pooled connection is dead,
        # and without this the next call on each one fails.
        check=ConnectionPool.check_connection,
        timeout=10.0,
        open=False,
        name="rag",
    )
    try:
        pool.open(wait=True, timeout=open_timeout)
    except (psycopg.Error, PoolTimeout) as exc:
        raise StorageError(f"Cannot connect to PostgreSQL: {exc}") from exc
    return pool


def _ensure_extension(database_url: str, timeout: float) -> None:
    """``register_vector`` needs the type to exist, so create the extension before pooling."""
    try:
        with psycopg.connect(database_url, connect_timeout=int(timeout), autocommit=True) as conn:
            conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
            try:
                conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            finally:
                conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
    except psycopg.OperationalError as exc:
        raise StorageError(f"Cannot connect to PostgreSQL: {exc}") from exc


def _migration_files() -> list[tuple[int, str, str]]:
    folder = resources.files("aiven_rag_mcp") / "migrations"
    found = []
    for entry in folder.iterdir():
        if entry.name.endswith(".sql"):
            version = int(entry.name.split("_", 1)[0])
            found.append((version, entry.name, entry.read_text(encoding="utf-8")))
    return sorted(found)


def migrate(pool: ConnectionPool, *, embedding_model: str, embedding_dim: int) -> list[str]:
    """Apply pending migrations and pin the embedding model. Safe to run concurrently."""
    applied: list[str] = []
    with pool.connection() as conn, conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_MIGRATION_LOCK_ID,))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            " version integer PRIMARY KEY, name text NOT NULL,"
            " applied_at timestamptz NOT NULL DEFAULT now())"
        )
        done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
        for version, name, sql in _migration_files():
            if version in done:
                continue
            logger.info("Applying migration %s", name)
            conn.execute(sql.replace("{embedding_dim}", str(embedding_dim)).encode())
            conn.execute(
                "INSERT INTO schema_migrations (version, name) VALUES (%s, %s)", (version, name)
            )
            applied.append(name)
        _pin_embedding_model(conn, embedding_model, embedding_dim)
    return applied


def _pin_embedding_model(conn: psycopg.Connection, model: str, dim: int) -> None:
    rows: dict[str, str] = dict(
        conn.execute(
            "SELECT key, value FROM rag_meta WHERE key IN ('embedding_model', 'embedding_dim')"
        ).fetchall()
    )
    if not rows:
        conn.execute(
            "INSERT INTO rag_meta (key, value)"
            " VALUES ('embedding_model', %s), ('embedding_dim', %s)",
            (model, str(dim)),
        )
        return
    if rows.get("embedding_model") != model or rows.get("embedding_dim") != str(dim):
        raise ConfigurationError(
            f"The database was built with embedding model {rows.get('embedding_model')} "
            f"({rows.get('embedding_dim')} dims) but the server is configured with {model} "
            f"({dim} dims). Set RAG_EMBEDDING_MODEL back, or reset the data volume "
            "(docker compose down -v) and re-ingest."
        )


def pgvector_version(pool: ConnectionPool) -> tuple[int, ...]:
    with pool.connection() as conn:
        row = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
    if row is None:
        raise StorageError("pgvector extension is not installed")
    return tuple(int(part) for part in str(row[0]).split("."))
