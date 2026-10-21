# Agent guide

RAG MCP server over the Aiven documentation. Python package `aiven_rag_mcp` in
`src/`, PostgreSQL + pgvector for storage, a local ONNX embedding model
(fastembed). See `README.md` for usage and `docs/PLAN.md` for design decisions.

## Layout

| Path | What |
|---|---|
| `src/aiven_rag_mcp/server.py` | MCP tools (`search`, `ingest`, `list_sources`, `delete_source`). Thin. |
| `src/aiven_rag_mcp/service.py` | `RagService`: ingest pipeline, search, rerank. |
| `src/aiven_rag_mcp/loaders.py` | text / file / directory / URL / git repo to documents; RST includes. |
| `src/aiven_rag_mcp/chunking.py` | Heading-aware Markdown and RST chunker. |
| `src/aiven_rag_mcp/store.py`, `db.py`, `migrations/` | SQL, hybrid search (HNSW + full text, RRF), migrations. |
| `tests/` | pytest. Fakes in `tests/support/`. |
| `eval/devportal_questions.json` | Labeled retrieval questions for `rag-mcp eval`. |

## Commands

```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/ruff check . && .venv/bin/ruff format --check . && .venv/bin/mypy src
TEST_DATABASE_URL=postgresql://rag:rag@localhost:5432/rag_test .venv/bin/pytest
docker compose up -d --build          # Postgres + server (HTTP on localhost:8000/mcp)
python scripts/smoke_test.py .mcp.json # end-to-end check through a real MCP client
```

Without `TEST_DATABASE_URL` the database tests are skipped, not failed.

## Rules

- stdout is the MCP stream in stdio mode: never `print` from library code.
- Expected failures raise `RagError` subclasses with user-readable messages.
- Schema changes need a new migration file; chunking changes need a
  `CHUNKER_VERSION` bump.
- Use `Optional[X]` (not `X | None`) and full type hints; `mypy --strict` must pass.
- If you change retrieval, run `rag-mcp eval eval/devportal_questions.json` before
  and after and report both tables.
