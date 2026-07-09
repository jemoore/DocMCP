# DocMCP

An MCP server providing semantic document search capabilities over PDF, Markdown, and HTML files. Runs locally in Docker and exposes tools via the MCP Streamable HTTP transport.

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
| `search_documents` | Semantic search across all indexed documents |
| `get_document` | Retrieve full text of a specific document |
| `list_documents` | List all indexed documents with metadata |
| `reindex` | Trigger incremental or full re-indexing |
| `index_status` | Get index statistics |

## Configuration

All settings are via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCMCP_HOST` | `0.0.0.0` | Bind host |
| `DOCMCP_PORT` | `8808` | Listen port |
| `DOCMCP_DOC_DIRS` | `/data/docs` | Comma-separated document directories |
| `DOCMCP_INDEX_DIR` | `/data/index` | Persistent index storage |
| `DOCMCP_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformer model |
| `DOCMCP_CHUNK_SIZE` | `1000` | Chunk size in characters |
| `DOCMCP_CHUNK_OVERLAP` | `200` | Chunk overlap in characters (must be < chunk size) |
| `DOCMCP_LOG_LEVEL` | `INFO` | Logging level |
| `DOCMCP_AUTH_TOKEN` | _(unset)_ | If set, require `Authorization: Bearer <token>` on all requests. When unset the server is unauthenticated — only expose it on a trusted network. |

## Development

```bash
# Install dependencies
uv sync --all-extras

# Run tests
uv run pytest

# Run server locally
uv run python -m docmcp.server
```
