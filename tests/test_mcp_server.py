"""The MCP contract: what Claude/Cursor see (tool names, schemas, annotations, errors)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from mcp import Client, StdioServerParameters

from aiven_rag_mcp.errors import StorageError
from aiven_rag_mcp.server import create_server
from aiven_rag_mcp.service import RagService

EXPECTED_TOOLS = {"ingest", "search", "list_sources", "delete_source"}


def unavailable_service() -> RagService:
    raise StorageError("Cannot connect to PostgreSQL: connection refused")


@pytest.mark.anyio
async def test_tools_are_listed_with_safety_annotations() -> None:
    async with Client(create_server(unavailable_service)) as client:
        tools = {tool.name: tool for tool in (await client.list_tools()).tools}

    assert set(tools) == EXPECTED_TOOLS
    assert tools["search"].annotations.read_only_hint is True
    assert tools["list_sources"].annotations.read_only_hint is True
    assert tools["delete_source"].annotations.destructive_hint is True
    assert tools["ingest"].annotations.destructive_hint is False
    assert {"text", "path", "url"} <= set(tools["ingest"].input_schema["properties"])
    assert tools["search"].input_schema["required"] == ["query"]
    assert tools["search"].output_schema is not None


@pytest.mark.anyio
async def test_database_outage_is_a_readable_tool_error_not_a_crash() -> None:
    async with Client(create_server(unavailable_service)) as client:
        result = await client.call_tool("search", {"query": "max connections"})
        assert result.is_error
        assert "Cannot connect to PostgreSQL" in result.content[0].text
        assert (await client.list_tools()).tools


@pytest.mark.anyio
async def test_invalid_arguments_are_rejected_by_schema() -> None:
    async with Client(create_server(unavailable_service)) as client:
        result = await client.call_tool("search", {"query": "x", "k": 1000})
    assert result.is_error


@pytest.mark.anyio
@pytest.mark.integration
async def test_ingest_then_search_over_mcp(service: RagService, corpus: Path) -> None:
    async with Client(create_server(lambda: service)) as client:
        ingest = await client.call_tool("ingest", {"path": str(corpus), "collection": "docs"})
        assert not ingest.is_error
        assert ingest.structured_content["added"] == 3

        missing = await client.call_tool("ingest", {})
        assert missing.is_error
        assert "exactly one" in missing.content[0].text

        search = await client.call_tool("search", {"query": "min_insync_replicas", "k": 2})
        top = search.structured_content["results"][0]
        assert top["source_uri"].endswith("kafka/reference/advanced-params.rst")
        assert "min_insync_replicas" in top["content"]

        listing = await client.call_tool("list_sources", {})
        assert listing.structured_content["total_sources"] == 3

        deleted = await client.call_tool("delete_source", {"collection": "docs"})
        assert deleted.structured_content["deleted_sources"] == 3


@pytest.mark.anyio
async def test_stdio_transport_starts_without_database_and_keeps_stdout_clean() -> None:
    """Claude Code launches the server over stdio; any stray print would break the protocol."""
    env = {
        **os.environ,
        "DATABASE_URL": "postgresql://nobody:nothing@127.0.0.1:1/none",
        "RAG_RERANKER_MODEL": "",
        "RAG_LOG_LEVEL": "DEBUG",
    }
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "aiven_rag_mcp.cli", "serve"], env=env
    )
    async with Client(params, read_timeout_seconds=30) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == EXPECTED_TOOLS
        result = await client.call_tool("list_sources", {})
        assert result.is_error
        assert "PostgreSQL" in result.content[0].text
