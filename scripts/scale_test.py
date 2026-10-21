"""How does the vector side scale past the size of devportal?

Builds synthetic chunk tables of N rows in a scratch schema, each row a real devportal
embedding plus Gaussian-ish noise (so the data keeps the real clustering instead of
being uniform random). Then measures load time, HNSW build time, sizes, kNN latency
and recall@10 of the HNSW index against an exact scan.

Only the vector part of search is measured; full-text and reranking are not included.

Usage:
    DATABASE_URL=postgresql://rag:rag@localhost:5432/rag \
        python scripts/scale_test.py --rows 100000 300000 1000000
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import psycopg

SCHEMA = "rag_scale"


def timed(conn: psycopg.Connection[Any], sql: str, params: Any = None) -> float:
    started = time.perf_counter()
    conn.execute(sql, params)
    return time.perf_counter() - started


def prepare_base(conn: psycopg.Connection[Any]) -> tuple[int, int]:
    conn.execute(
        f"CREATE TABLE {SCHEMA}.base AS SELECT row_number() OVER () AS rn, embedding FROM chunks"
    )
    conn.execute(f"CREATE UNIQUE INDEX ON {SCHEMA}.base (rn)")
    row = conn.execute(
        f"SELECT count(*), max(vector_dims(embedding)) FROM {SCHEMA}.base"
    ).fetchone()
    assert row is not None and row[0] > 0, "ingest some documents first"
    return int(row[0]), int(row[1])


def measure(
    conn: psycopg.Connection[Any], rows: int, queries: int, ef_search: int
) -> dict[str, Any]:
    base_rows, dim = prepare_base(conn)
    conn.execute(f"CREATE UNLOGGED TABLE {SCHEMA}.items (id bigserial, embedding vector({dim}))")
    # Uniform noise in [-0.04, 0.04] per dimension: cosine ~0.9 to the source chunk.
    load = timed(
        conn,
        f"""
        INSERT INTO {SCHEMA}.items (embedding)
        SELECT l2_normalize(b.embedding + n.noise)
        FROM generate_series(1, %(rows)s) AS g(i)
        JOIN {SCHEMA}.base b ON b.rn = g.i %% %(base)s + 1
        CROSS JOIN LATERAL (
            -- The reference to g.i makes the subquery correlated: fresh noise per row.
            SELECT ARRAY(
                SELECT (random() - 0.5) * 0.08 FROM generate_series(1, {dim}) WHERE g.i > 0
            )::vector AS noise
        ) n
        """,
        {"rows": rows, "base": base_rows},
    )
    conn.execute("SET maintenance_work_mem = '900MB'")
    conn.execute("SET max_parallel_maintenance_workers = 3")
    build = timed(conn, f"CREATE INDEX ON {SCHEMA}.items USING hnsw (embedding vector_cosine_ops)")
    sizes = conn.execute(
        f"SELECT pg_table_size('{SCHEMA}.items'), pg_indexes_size('{SCHEMA}.items')"
    ).fetchone()
    assert sizes is not None

    probes = conn.execute(
        f"SELECT embedding FROM {SCHEMA}.items ORDER BY random() LIMIT %s", (queries,)
    ).fetchall()
    knn = f"SELECT id FROM {SCHEMA}.items ORDER BY embedding <=> %s LIMIT 10"
    latencies: list[float] = []
    recalls: list[float] = []
    for (probe,) in probes:
        conn.execute(f"SET hnsw.ef_search = {ef_search}")
        started = time.perf_counter()
        approx = {r[0] for r in conn.execute(knn, (probe,)).fetchall()}
        latencies.append((time.perf_counter() - started) * 1000)
        with conn.transaction():
            conn.execute("SET LOCAL enable_indexscan = off")
            exact = {r[0] for r in conn.execute(knn, (probe,)).fetchall()}
        recalls.append(len(approx & exact) / len(exact))
    latencies.sort()
    return {
        "rows": rows,
        "load_s": round(load, 1),
        "hnsw_build_s": round(build, 1),
        "table_mb": round(sizes[0] / 2**20),
        "index_mb": round(sizes[1] / 2**20),
        "knn_p50_ms": round(statistics.median(latencies), 2),
        "knn_p95_ms": round(latencies[int(0.95 * (len(latencies) - 1))], 2),
        "recall_at_10": round(statistics.mean(recalls), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, nargs="+", default=[100_000, 300_000, 1_000_000])
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    url = os.environ.get("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
    results = []
    with psycopg.connect(url, autocommit=True) as conn:
        for rows in args.rows:
            conn.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
            conn.execute(f"CREATE SCHEMA {SCHEMA}")
            try:
                result = measure(conn, rows, args.queries, args.ef_search)
            finally:
                conn.execute(f"DROP SCHEMA {SCHEMA} CASCADE")
            results.append(result)
            print(json.dumps(result), flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
