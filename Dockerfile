# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    RAG_MODEL_CACHE_DIR=/opt/models

# git is needed to ingest repository URLs (shallow clone).
RUN apt-get update \
    && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 app

WORKDIR /app

# 1. Dependencies (cached until requirements.txt changes).
COPY requirements.txt ./
RUN pip install -r requirements.txt

# 2. Model weights baked into the image, so the container needs no network or API key
#    at runtime. Only the modules needed to download are copied, so this layer survives
#    changes to the rest of the code.
COPY src/aiven_rag_mcp/__init__.py src/aiven_rag_mcp/config.py src/aiven_rag_mcp/errors.py \
     src/aiven_rag_mcp/embeddings.py /tmp/bootstrap/aiven_rag_mcp/
RUN PYTHONPATH=/tmp/bootstrap python -c \
        "from aiven_rag_mcp.embeddings import download_models; print(download_models())" \
    && rm -rf /tmp/bootstrap \
    && chown -R app:app /opt/models

# 3. Application code.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-deps .
COPY eval ./eval

ENV HF_HUB_OFFLINE=1 \
    RAG_INGEST_ALLOWED_PATHS=/data
USER app
EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8000), 2)"

# The long-running process serves MCP over Streamable HTTP (for Cursor / remote clients).
# Claude Code and Claude Desktop attach over stdio with: docker exec -i aiven-rag-mcp rag-mcp serve
CMD ["rag-mcp", "serve", "--transport", "http", "--host", "0.0.0.0", "--port", "8000"]
