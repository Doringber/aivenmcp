"""MCP tool layer. Thin: validates arguments, runs service calls off the event loop."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Annotated, Literal, Optional, TypeVar

import anyio
import anyio.from_thread
import anyio.to_thread
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from aiven_rag_mcp import __version__
from aiven_rag_mcp.errors import RagError
from aiven_rag_mcp.models import DeleteReport, IngestReport, SearchResponse, SourceListing
from aiven_rag_mcp.service import RagService

logger = logging.getLogger(__name__)
T = TypeVar("T")

INSTRUCTIONS = """\
Retrieval over an indexed documentation knowledge base (by default the Aiven
documentation, github.com/aiven/devportal).

- Call `search` before answering questions about the indexed content, and answer
  from the returned chunks. Cite the `web_url` (or `source_uri`) of the chunks used.
- If the first results do not contain the answer, search again with different
  wording, the exact parameter or product name, or a `collection` / `uri_prefix` filter.
- If nothing relevant comes back, say so instead of guessing.
- Use `list_sources` to see what is indexed and `ingest` to add more content.
"""


async def _run(fn: Callable[[], T]) -> T:
    try:
        return await anyio.to_thread.run_sync(fn)
    except RagError as exc:
        raise ToolError(str(exc)) from exc


def create_server(service: Callable[[], RagService]) -> MCPServer:
    mcp: MCPServer = MCPServer(
        name="aiven-docs-rag",
        title="Aiven docs RAG",
        version=__version__,
        instructions=INSTRUCTIONS,
    )

    @mcp.tool(
        title="Search the knowledge base",
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def search(
        query: Annotated[
            str,
            Field(
                description="Natural-language question or keywords. Include exact identifiers "
                "(parameter names, CLI commands, error messages) when you know them."
            ),
        ],
        k: Annotated[int, Field(ge=1, le=20, description="Number of chunks to return.")] = 5,
        collection: Annotated[
            Optional[str],
            Field(description="Only search this collection, e.g. 'github.com/aiven/devportal'."),
        ] = None,
        uri_prefix: Annotated[
            Optional[str],
            Field(
                description="Only search sources whose URI starts with this prefix, e.g. "
                "'github.com/aiven/devportal/docs/products/postgresql/'."
            ),
        ] = None,
        mode: Annotated[
            Literal["hybrid", "vector", "keyword"],
            Field(description="'hybrid' (default) fuses semantic and keyword search."),
        ] = "hybrid",
    ) -> SearchResponse:
        """Return the chunks most relevant to a query, each with its source and section heading.

        Use this to answer questions from the indexed documentation. Results are ranked by
        relevance; each result includes `content`, `heading` (document title and section path),
        `source_uri` and `web_url` for citation.
        """
        return await _run(
            lambda: service().search(
                query, k=k, mode=mode, collection=collection, uri_prefix=uri_prefix
            )
        )

    @mcp.tool(
        title="Ingest content",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=True
        ),
    )
    async def ingest(
        ctx: Context,
        text: Annotated[Optional[str], Field(description="Raw text or Markdown to store.")] = None,
        path: Annotated[
            Optional[str],
            Field(
                description="File or directory path on the server's filesystem (inside Docker, "
                "the host's ./data folder is mounted at /data). Directories are scanned "
                "recursively for .md, .mdx, .rst and .txt files."
            ),
        ] = None,
        url: Annotated[
            Optional[str],
            Field(
                description="http(s) URL of a page or raw file. A GitHub repository URL "
                "(https://github.com/owner/repo or .../tree/<branch>/<subdir>) is cloned and "
                "all its Markdown/RST files are ingested."
            ),
        ] = None,
        title: Annotated[Optional[str], Field(description="Title for a single document.")] = None,
        source_id: Annotated[
            Optional[str],
            Field(description="Stable id for `text` input, so re-ingesting replaces it."),
        ] = None,
        collection: Annotated[
            Optional[str], Field(description="Group name used for filtering and deletion.")
        ] = None,
        prune: Annotated[
            bool,
            Field(
                description="For directories and repositories: delete previously ingested "
                "sources of the same collection that no longer exist."
            ),
        ] = False,
    ) -> IngestReport:
        """Store text, a file/directory, a URL, or a whole git repository for retrieval.

        Provide exactly one of `text`, `path` or `url`. Documents are split on headings,
        embedded locally and stored. Re-ingesting unchanged content is a no-op, so this is
        safe to repeat; changed documents are replaced atomically.
        """

        def progress(done: int, message: str) -> None:
            try:
                anyio.from_thread.run(ctx.report_progress, float(done), None, message)
            except Exception as exc:
                logger.debug("Progress notification failed: %s", exc)

        return await _run(
            lambda: service().ingest(
                text=text,
                path=path,
                url=url,
                title=title,
                source_id=source_id,
                collection=collection,
                prune=prune,
                progress=progress,
            )
        )

    @mcp.tool(
        title="List ingested sources",
        annotations=ToolAnnotations(
            read_only_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def list_sources(
        collection: Annotated[Optional[str], Field(description="Only this collection.")] = None,
        uri_prefix: Annotated[
            Optional[str], Field(description="Only sources whose URI starts with this.")
        ] = None,
        limit: Annotated[int, Field(ge=1, le=500)] = 50,
        offset: Annotated[int, Field(ge=0)] = 0,
    ) -> SourceListing:
        """List what has been ingested: per-collection totals plus a page of sources."""
        return await _run(
            lambda: service().list_sources(
                collection=collection, uri_prefix=uri_prefix, limit=limit, offset=offset
            )
        )

    @mcp.tool(
        title="Delete sources",
        annotations=ToolAnnotations(
            read_only_hint=False, destructive_hint=True, idempotent_hint=True, open_world_hint=False
        ),
    )
    async def delete_source(
        uri: Annotated[Optional[str], Field(description="Exact source URI to delete.")] = None,
        collection: Annotated[
            Optional[str], Field(description="Delete every source in this collection.")
        ] = None,
        uri_prefix: Annotated[
            Optional[str], Field(description="Delete every source whose URI starts with this.")
        ] = None,
    ) -> DeleteReport:
        """Delete sources and their chunks. At least one filter is required; filters combine."""
        return await _run(
            lambda: service().delete_source(uri=uri, collection=collection, uri_prefix=uri_prefix)
        )

    return mcp
