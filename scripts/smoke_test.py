"""Connect to the server the way an MCP client does and run one search.

Usage:
    python scripts/smoke_test.py stdio            # docker exec -i aiven-rag-mcp rag-mcp serve
    python scripts/smoke_test.py http             # http://localhost:8000/mcp
    python scripts/smoke_test.py http --query "..."
"""

from __future__ import annotations

import argparse
import asyncio
import time

from mcp import Client, StdioServerParameters

STDIO = StdioServerParameters(
    command="docker", args=["exec", "-i", "aiven-rag-mcp", "rag-mcp", "serve"]
)
HTTP_URL = "http://localhost:8000/mcp"


async def run(transport: str, query: str) -> None:
    target = STDIO if transport == "stdio" else HTTP_URL
    started = time.perf_counter()
    async with Client(target, read_timeout_seconds=120) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        print(f"connected over {transport} in {time.perf_counter() - started:.2f}s, tools={tools}")
        listing = await client.call_tool("list_sources", {"limit": 1})
        print("indexed:", listing.structured_content["collections"])
        started = time.perf_counter()
        result = await client.call_tool("search", {"query": query, "k": 3})
        print(f"search took {(time.perf_counter() - started) * 1000:.0f} ms")
        for hit in result.structured_content["results"]:
            print(f"  [{hit['rank']}] {hit['heading']}\n      {hit['web_url'] or hit['source_uri']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("transport", choices=["stdio", "http"])
    parser.add_argument(
        "--query", default="How many connections does a PostgreSQL Startup-8 plan allow?"
    )
    args = parser.parse_args()
    asyncio.run(run(args.transport, args.query))


if __name__ == "__main__":
    main()
