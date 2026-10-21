# Plan: RAG MCP server over the Aiven docs

This document breaks the assignment into components, records the best practice
chosen for each one, and lists the assumptions made. It is the working plan the
code in this repository follows.

## 1. What is being asked

| Deliverable | Must have | Nice to have |
|---|---|---|
| MCP server (stdio, Claude Code) | `ingest`, `search`, `list_sources` | `delete_source`, metadata filters |
| Knowledge base | Whole `aiven/devportal` repo (Markdown + RST) | Re-ingest only what changed |
| Embeddings | Local open-source model inside the container, no API key | Hosted provider via env var |
| Storage | PostgreSQL + pgvector in docker compose, persistent volume | |
| Packaging | Dockerfile + docker-compose.yml, one command | |
| README | Running in Claude in under 10 minutes, MCP snippet, env vars, example session | |
| Tests | Critical paths, with a stated reason for each | Retrieval-quality eval |
| Slide deck | 6 slides: solution, load (measured), where it breaks, next level, monitoring, 3 tradeoffs | |
| Extra (our own) | Cursor support: project `.cursor/mcp.json`, rules, install link | Streamable HTTP transport |

How it is graded: runs from the README on the first try, retrieval quality on
their own documents, code clarity, honesty about limits, quality of reasoning.

## 2. Component breakdown

```
            ┌──────────── MCP client (Claude Code / Claude Desktop / Cursor) ───────────┐
            │  stdio:  docker exec -i aiven-rag-mcp rag-mcp serve                       │
            │  http:   http://localhost:8000/mcp  (Cursor / remote clients)             │
            └──────────────────────────────┬────────────────────────────────────────────┘
                                           │ tools: ingest / search / list_sources / delete_source
                         ┌─────────────────▼─────────────────┐
                         │ server.py   MCP tool layer (thin) │
                         └─────────────────┬─────────────────┘
                         ┌─────────────────▼─────────────────┐
                         │ service.py  RagService            │  orchestration, idempotency, errors
                         └──┬──────────┬───────────┬─────────┘
             ┌──────────────▼┐  ┌──────▼──────┐  ┌─▼────────────────────────┐
             │ loaders.py    │  │ chunking.py │  │ embeddings.py            │
             │ text/file/dir │  │ MD + RST    │  │ fastembed (ONNX, CPU)    │
             │ URL/git repo  │  │ heading-    │  │ bge-small-en-v1.5, 384-d │
             └───────────────┘  │ aware       │  └──────────────────────────┘
                                └─────────────┘
                         ┌───────────────────────────────────┐
                         │ store.py  + db.py (migrations)    │  psycopg 3 pool
                         │ PostgreSQL 16 + pgvector 0.8      │  HNSW + GIN(tsvector), RRF fusion
                         └───────────────────────────────────┘
```

| # | Component | Responsibility | Key decisions |
|---|---|---|---|
| 1 | Config | Typed settings from env vars | pydantic-settings; every knob has a working default |
| 2 | Loaders | Turn `text` / `path` / `url` into documents | Explicit tool arguments, not guessing. Git repo URLs are shallow-cloned. HTML is converted to Markdown-like text with the standard library |
| 3 | Chunking | Split documents under the model's 512-token limit | Heading-aware for Markdown and RST. Packs small sibling sections and splits large ones with overlap. Prepends `title > section > subsection` to every chunk before embedding |
| 4 | Embeddings | Text to vector | Local ONNX model via fastembed (no PyTorch, small image). Model baked into the image at build time |
| 5 | Storage | Sources, chunks, vectors, full-text index | Two tables. HNSW cosine index, generated `tsvector` + GIN index. Versioned SQL migrations applied automatically under an advisory lock |
| 6 | Retrieval | Query to ranked chunks | Hybrid: vector kNN + Postgres full-text, fused with Reciprocal Rank Fusion. Optional filters by collection and URI prefix |
| 7 | Service | Ingest pipeline and search orchestration | Content-hash idempotency, per-document error isolation, one transaction per document, optional prune of deleted files |
| 8 | MCP layer | Tool schemas, descriptions, transports | Official MCP Python SDK. stdio for Claude, Streamable HTTP for Cursor/remote. Logs only to stderr |
| 9 | Packaging | One command to run | `docker compose up -d` starts Postgres and the server. Claude/Cursor attach with `docker exec -i` |
| 10 | Tests + eval | Prove the critical paths | Unit tests for chunking and loaders, integration tests against real pgvector, MCP contract test, retrieval eval on devportal questions |
| 11 | Measurements | Real numbers for the deck | `scripts/benchmark.py` for ingest throughput, search latency and concurrency |
| 12 | Cursor support | First-class use from Cursor | `.cursor/mcp.json`, `.cursor/rules/*.mdc`, one-click install link, `AGENTS.md` |

## 3. Best practices applied (and why)

### MCP server
- **Never write to stdout in stdio mode.** stdout is the JSON-RPC channel; any
  print corrupts the stream. All logging goes to stderr.
- **Few, well-described tools with explicit arguments.** The model chooses tools
  from their descriptions, so the descriptions say when to use each tool and
  what comes back. `ingest` takes `text`, `path` or `url` as separate arguments
  instead of one ambiguous string.
- **Tool annotations** (`readOnlyHint`, `destructiveHint`, `idempotentHint`) so
  clients can auto-approve safe tools and confirm destructive ones.
- **Structured output** (Pydantic models) plus readable text, so both humans and
  agents can use the result.
- **Return citations.** Every hit carries the source URI, a web URL when known,
  and the heading path, so answers can be checked.
- **Bounded responses.** `k` is capped and `list_sources` is paginated so a tool
  result cannot flood the context window (devportal alone has 845 sources).
- **Progress notifications** during long ingests so clients do not time out.
- **Fast startup.** The model loads lazily, so the MCP handshake is not blocked
  by model loading.
- **Errors as tool errors, not crashes.** One bad file fails that file only and
  is reported in the ingest result.

### RAG quality
- **Structure-aware chunking.** Docs are written in sections, and the answer to
  "what is parameter X" is usually one section. Splitting on headings keeps a
  parameter and its description together.
- **Contextual chunk headers.** Embedding `Title > Section > Subsection` with the
  text helps short sections like `max_connections` that say little on their own.
- **Hybrid search.** Dense vectors handle paraphrases; keyword search handles
  exact identifiers (`pg_stat_statements`, `kafka.auto_create_topics_enable`)
  that small embedding models handle poorly. RRF fusion needs no score
  calibration between the two.
- **Query instruction prefix** for BGE models on the query side only, as the
  model card recommends.
- **Measure retrieval, do not guess.** A small labeled question set
  (`eval/devportal_questions.json`) reports hit@1, hit@5 and MRR for vector-only,
  keyword-only and hybrid search.

### PostgreSQL / pgvector
- HNSW index (better recall/latency than IVFFlat and no training step) with
  cosine distance on normalized embeddings.
- `hnsw.iterative_scan = relaxed_order` (pgvector >= 0.8) so filtered searches
  still return `k` results.
- Generated `tsvector` column with weights (heading > body), so it cannot drift
  from the content.
- `ON DELETE CASCADE` from sources to chunks; replacing a document is one
  transaction (delete old chunks, insert new), so readers never see half a
  document.
- Migrations are versioned SQL files, applied at startup inside
  `pg_advisory_xact_lock`, because several stdio server processes can start at
  the same time.
- The embedding model name and dimension are stored in the database. Starting
  with a different model fails with a clear message instead of mixing vector
  spaces.

### Packaging
- Slim Python base image, non-root user, model weights downloaded at build time
  (runtime works offline with `HF_HUB_OFFLINE=1`).
- Compose healthcheck on Postgres; the server waits for it.
- Named volume for data persistence.

## 4. Assumptions (also listed in the README)
1. "Whole repo" means every `.md`, `.mdx`, `.rst` and `.txt` file in
   `aiven/devportal`. RST `include::` directives are expanded inline (confined
   to the repo, depth 3). Fragments that are included elsewhere are not indexed
   on their own, so the page that renders them is what gets returned. This
   raised hybrid MRR from 0.78 to about 0.82 on the eval set.
2. Markup is kept mostly as-is. RST and Markdown syntax costs a few tokens but
   keeps code blocks and tables readable in answers.
3. English-only content, so Postgres full-text uses the `english` config.
4. The MCP client is trusted (it runs on the user's machine), but file ingestion
   can still be limited to allowed roots with `RAG_INGEST_ALLOWED_PATHS`.
5. Single Postgres and single server process are enough for this scope, as the
   assignment says. Scaling paths go in the deck.

## 5. Work order
1. Core library: config, loaders, chunking, embeddings, store, service.
2. MCP server + CLI.
3. Docker + compose.
4. Cursor support files.
5. Tests (unit, integration, MCP contract) + retrieval eval.
6. Ingest the full devportal repo, run the eval and the benchmarks.
7. README and slide deck with the measured numbers.
