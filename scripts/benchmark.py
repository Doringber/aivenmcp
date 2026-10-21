"""Measure search latency and throughput through a real MCP client connection.

Every worker opens its own MCP session (like separate Cursor / Claude windows) and
sends the eval questions in a loop. Reports latency percentiles per concurrency level.

Usage:
    python scripts/benchmark.py .cursor/mcp.json --concurrency 1 4 8 16 --requests 64
    python scripts/benchmark.py .cursor/mcp.json --output bench/results.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Union

from mcp import Client, StdioServerParameters

sys.path.insert(0, str(Path(__file__).parent))
from smoke_test import load_target

Target = Union[str, StdioServerParameters]


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[index]


async def worker(target: Target, queries: list[str], count: int, offset: int) -> list[float]:
    latencies: list[float] = []
    async with Client(target, read_timeout_seconds=300) as client:
        await client.call_tool("search", {"query": "warm up", "k": 1})
        for i in range(count):
            started = time.perf_counter()
            result = await client.call_tool(
                "search", {"query": queries[(offset + i) % len(queries)], "k": 5}
            )
            if result.is_error:
                raise RuntimeError(f"search failed: {result.content}")
            latencies.append((time.perf_counter() - started) * 1000)
    return latencies


async def run_level(
    target: Target, queries: list[str], concurrency: int, requests: int
) -> dict[str, Any]:
    per_worker = max(1, requests // concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(worker(target, queries, per_worker, w * per_worker) for w in range(concurrency))
    )
    wall = time.perf_counter() - started
    latencies = [value for chunk in results for value in chunk]
    return {
        "concurrency": concurrency,
        "requests": len(latencies),
        "p50_ms": round(statistics.median(latencies), 1),
        "p95_ms": round(percentile(latencies, 95), 1),
        "p99_ms": round(percentile(latencies, 99), 1),
        "max_ms": round(max(latencies), 1),
        # Includes session setup and one warm-up call per worker.
        "throughput_rps": round(len(latencies) / wall, 2),
    }


async def main_async(args: argparse.Namespace) -> None:
    target = load_target(args.config, args.server)
    queries = [q["question"] for q in json.loads(args.questions.read_text())["questions"]]
    rows = []
    print(f"{'conc':>4} {'reqs':>5} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'req/s':>7}")
    for level in args.concurrency:
        row = await run_level(target, queries, level, args.requests)
        rows.append(row)
        print(
            f"{row['concurrency']:>4} {row['requests']:>5} {row['p50_ms']:>8} "
            f"{row['p95_ms']:>8} {row['p99_ms']:>8} {row['max_ms']:>8} "
            f"{row['throughput_rps']:>7}"
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="MCP client config (.cursor/mcp.json)")
    parser.add_argument("--server")
    parser.add_argument("--questions", type=Path, default=Path("eval/devportal_questions.json"))
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--requests", type=int, default=64, help="Requests per level")
    parser.add_argument("--output", type=Path)
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
