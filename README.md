# DocMCP

An MCP server providing semantic document search capabilities over PDF, EPUB, Markdown, HTML, and plain-text files. Runs locally in Docker and exposes tools via the MCP Streamable HTTP transport.

## Quick Start

### 1. Configure document directories

Edit `docker-compose.yml` to mount your document directories:

```yaml
services:
  docmcp:
    volumes:
      - /home/user/documents:/data/docs/documents:ro
      - /home/user/notes:/data/docs/notes:ro
      - docmcp-index:/data/index
```

### 2. Build and run

```bash
docker compose up --build
```

The server starts on port **8808** and begins indexing documents in the background.

### 3. Connect an MCP client

Point any MCP client at `http://<your-host>:8808/mcp`.

## MCP Tools

| Tool | Description |
|------|-------------|
| `search_documents` | Semantic search across indexed documents, optionally scoped with a `document_filter` glob (e.g. `python*.pdf`) |
| `get_context` | Expand a search hit into the surrounding passage (neighboring chunks merged, overlap removed) |
| `get_document` | Retrieve document text, windowed by page range (PDFs) and/or offset + max_chars, with a `truncated` flag |
| `list_documents` | List all indexed documents with metadata (documents with `chunk_count` 0 had no extractable text, e.g. scanned PDFs needing OCR) |
| `reindex` | Trigger incremental or full re-indexing in the background; returns immediately |
| `index_status` | Get index statistics, live indexing progress (N of M files), and the last run's summary |

The intended query loop for grounded answers: `search_documents` to find
relevant passages, `get_context` on the best hits for fuller quotes, and
`get_document` with a page range when a whole section is needed.

Indexing runs in the background (at startup and via `reindex`); search and the
other tools stay available throughout, and a full rebuild keeps serving the old
index until the new one is swapped in.

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCMCP_HOST` | `0.0.0.0` | Bind host |
| `DOCMCP_PORT` | `8808` | Listen port |
| `DOCMCP_DOC_DIRS` | `/data/docs` | Comma-separated document directories |
| `DOCMCP_INDEX_DIR` | `/data/index` | Persistent index storage |
| `DOCMCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `DOCMCP_CHUNK_SIZE` | `1000` | Chunk size in characters (keep within the embedding model's window — see below) |
| `DOCMCP_CHUNK_OVERLAP` | `200` | Chunk overlap in characters (must be < chunk size) |
| `DOCMCP_LOG_LEVEL` | `INFO` | Logging level |
| `DOCMCP_AUTH_TOKEN` | _(unset)_ | If set, require `Authorization: Bearer <token>` on all requests. When unset the server is unauthenticated — only expose it on a trusted network. |

### Choosing an embedding model

Embedding models silently truncate text past their max sequence length, so
chunks longer than the model's window are only partially searchable (the
server logs a warning when the configured chunk size exceeds it). The default
`all-MiniLM-L6-v2` encodes 256 tokens (≈1,000 characters), matching the
default chunk size. If you want larger chunks or better retrieval quality,
set `DOCMCP_EMBEDDING_MODEL` to something like `BAAI/bge-small-en-v1.5`
(512 tokens) and run `reindex(full=true)` to re-embed. Changing the model
always requires a full reindex. The downloaded model is cached in the
`docmcp-hf-cache` volume so it survives container recreation.

## Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run server locally
uv run python -m docmcp.server
```
