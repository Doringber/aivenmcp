CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS rag_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    id           bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    uri          text        NOT NULL UNIQUE,
    collection   text        NOT NULL,
    kind         text        NOT NULL,
    title        text        NOT NULL,
    web_url      text,
    content_hash text        NOT NULL,
    char_count   integer     NOT NULL,
    chunk_count  integer     NOT NULL,
    metadata     jsonb       NOT NULL DEFAULT '{}'::jsonb,
    created_at   timestamptz NOT NULL DEFAULT now(),
    updated_at   timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sources_collection_idx ON sources (collection);
CREATE INDEX IF NOT EXISTS sources_uri_pattern_idx ON sources (uri text_pattern_ops);

CREATE TABLE IF NOT EXISTS chunks (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    source_id   bigint  NOT NULL REFERENCES sources (id) ON DELETE CASCADE,
    ordinal     integer NOT NULL,
    heading     text    NOT NULL,
    content     text    NOT NULL,
    token_count integer NOT NULL,
    embedding   vector({embedding_dim}) NOT NULL,
    tsv         tsvector GENERATED ALWAYS AS (
                    setweight(to_tsvector('english', heading), 'A')
                    || setweight(to_tsvector('english', content), 'B')
                ) STORED,
    UNIQUE (source_id, ordinal)
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin (tsv);
