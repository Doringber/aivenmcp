"""Connect using a shipped MCP client config, exactly as Cursor / Claude Code would.

Usage:
    python scripts/smoke_test.py .mcp.json            # Claude Code config (stdio, docker exec)
    python scripts/smoke_test.py .cursor/mcp.json     # Cursor config (streamable HTTP)
    python scripts/smoke_test.py .mcp.json --query "..."
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional, Union

from mcp import Client, StdioServerParameters


def load_target(config: Path, server: Optional[str]) -> Union[str, StdioServerParameters]:
    servers: dict[str, dict[str, Any]] = json.loads(config.read_text())["mcpServers"]
    entry = servers[server] if server else next(iter(servers.values()))
    if "url" in entry:
        return str(entry["url"])
    return StdioServerParameters(
        command=entry["command"], args=entry.get("args", []), env=entry.get("env") or None
    )


async def run(target: Union[str, StdioServerParameters], query: str) -> None:
    label = target if isinstance(target, str) else " ".join([target.command, *target.args])
    started = time.perf_counter()
    async with Client(target, read_timeout_seconds=120) as client:
        tools = sorted(t.name for t in (await client.list_tools()).tools)
        print(f"connected to {label} in {time.perf_counter() - started:.2f}s, tools={tools}")
        listing = await client.call_tool("list_sources", {"limit": 1})
        print("indexed:", listing.structured_content["collections"])
        started = time.perf_counter()
        result = await client.call_tool("search", {"query": query, "k": 3})
        print(f"search took {(time.perf_counter() - started) * 1000:.0f} ms")
        for hit in result.structured_content["results"]:
            print(f"  [{hit['rank']}] {hit['heading']}")
            print(f"      {hit['web_url'] or hit['source_uri']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="MCP client config (.mcp.json, .cursor/mcp.json)")
    parser.add_argument("--server", help="Server name inside the config (default: the first)")
    parser.add_argument(
        "--query", default="How many connections does a PostgreSQL Startup-8 plan allow?"
    )
    args = parser.parse_args()
    asyncio.run(run(load_target(args.config, args.server), args.query))


if __name__ == "__main__":
    main()
