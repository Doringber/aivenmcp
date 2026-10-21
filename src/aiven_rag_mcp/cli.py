"""Command line entry point: ``rag-mcp serve | ingest | search | sources | delete | migrate``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Optional

from aiven_rag_mcp.config import Settings, get_settings
from aiven_rag_mcp.errors import RagError
from aiven_rag_mcp.models import SearchMode

logger = logging.getLogger("aiven_rag_mcp")


def _configure_logging(level: str) -> None:
    # stdout carries the MCP JSON-RPC stream in stdio mode; logs must go to stderr.
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-mcp", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Run the MCP server")
    serve.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="Extra Host header values accepted over HTTP (DNS-rebinding protection).",
    )

    ingest = sub.add_parser("ingest", help="Ingest a path, URL or git repository URL")
    ingest.add_argument("target", help="File/directory path, http(s) URL or GitHub repo URL")
    ingest.add_argument("--collection")
    ingest.add_argument("--title")
    ingest.add_argument("--prune", action="store_true", help="Delete sources that disappeared")

    search = sub.add_parser("search", help="Search from the command line")
    search.add_argument("query")
    search.add_argument("-k", type=int, default=5)
    search.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    search.add_argument("--collection")
    search.add_argument("--uri-prefix")
    search.add_argument("--json", action="store_true")

    sources = sub.add_parser("sources", help="List ingested sources")
    sources.add_argument("--collection")
    sources.add_argument("--uri-prefix")
    sources.add_argument("--limit", type=int, default=50)

    delete = sub.add_parser("delete", help="Delete sources")
    delete.add_argument("--uri")
    delete.add_argument("--collection")
    delete.add_argument("--uri-prefix")

    evaluate = sub.add_parser("eval", help="Measure retrieval quality on a labeled question set")
    evaluate.add_argument(
        "questions", type=Path, help="JSON file, e.g. eval/devportal_questions.json"
    )
    evaluate.add_argument(
        "--mode", action="append", choices=["hybrid", "vector", "keyword"], default=None
    )
    evaluate.add_argument("--collection")
    evaluate.add_argument("--output", type=Path, help="Write per-question results as JSON")
    evaluate.add_argument("--verbose", action="store_true", help="Print misses")

    sub.add_parser("migrate", help="Create or upgrade the database schema")
    sub.add_parser("download-model", help="Download the embedding model into the cache")
    return parser


def _serve(args: argparse.Namespace, settings: Settings) -> None:
    from mcp.server.transport_security import TransportSecuritySettings

    from aiven_rag_mcp.app import LazyService
    from aiven_rag_mcp.server import create_server

    lazy = LazyService(settings)
    server = create_server(lazy.get)
    lazy.warm_up_in_background()
    if args.transport == "stdio":
        server.run("stdio")
        return
    hosts = ["127.0.0.1:*", "localhost:*", "[::1]:*", *args.allowed_host]
    security = TransportSecuritySettings(
        allowed_hosts=hosts,
        allowed_origins=[f"http://{h}" for h in hosts],
    )
    logger.info("Streamable HTTP on http://%s:%d/mcp", args.host, args.port)
    server.run("streamable-http", host=args.host, port=args.port, transport_security=security)


def _run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "serve":
        _serve(args, settings)
        return 0
    if args.command == "download-model":
        from aiven_rag_mcp.embeddings import download_models

        for model in download_models(settings):
            print(f"Model ready: {model}")
        return 0

    from aiven_rag_mcp.app import build_service

    service = build_service(settings)
    match args.command:
        case "migrate":
            print("Schema is up to date")
        case "ingest":
            is_url = args.target.startswith(("http://", "https://"))
            report = service.ingest(
                url=args.target if is_url else None,
                path=None if is_url else args.target,
                collection=args.collection,
                title=args.title,
                prune=args.prune,
                progress=lambda done, message: logger.info(message),
            )
            print(report.model_dump_json(indent=2))
            return 1 if report.failed else 0
        case "search":
            response = service.search(
                args.query,
                k=args.k,
                mode=args.mode,
                collection=args.collection,
                uri_prefix=args.uri_prefix,
            )
            if args.json:
                print(response.model_dump_json(indent=2))
            else:
                for hit in response.results:
                    print(f"[{hit.rank}] {hit.score:.4f}  {hit.heading}")
                    print(f"    {hit.web_url or hit.source_uri}")
                    print("    " + hit.content[:300].replace("\n", " ") + "\n")
        case "sources":
            listing = service.list_sources(
                collection=args.collection, uri_prefix=args.uri_prefix, limit=args.limit
            )
            print(listing.model_dump_json(indent=2))
        case "eval":
            from aiven_rag_mcp.evaluation import evaluate, format_table, load_questions

            questions = load_questions(args.questions)
            configs: list[tuple[SearchMode, bool]] = (
                [(mode, False) for mode in args.mode]
                if args.mode
                else [("vector", False), ("keyword", False), ("hybrid", False), ("hybrid", True)]
            )
            summaries = [
                evaluate(service, questions, mode, rerank=rerank, collection=args.collection)
                for mode, rerank in configs
            ]
            print(format_table(summaries))
            if args.verbose:
                for summary in summaries:
                    for result in summary.results:
                        if result.rank is None or result.rank > 3:
                            print(
                                f"  [{summary.mode}] {result.id}: rank={result.rank} "
                                f"top={result.top_source}"
                            )
            if args.output:
                args.output.write_text(
                    json.dumps([s.model_dump() for s in summaries], indent=2), encoding="utf-8"
                )
        case "delete":
            print(
                service.delete_source(
                    uri=args.uri, collection=args.collection, uri_prefix=args.uri_prefix
                ).model_dump_json()
            )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    settings = get_settings()
    _configure_logging(settings.log_level)
    try:
        return _run(args, settings)
    except RagError as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
