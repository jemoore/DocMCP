from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastmcp import FastMCP

from docmcp.config import Config
from docmcp.indexer import Indexer

logger = logging.getLogger(__name__)

config = Config.from_env()
config.setup_logging()

# The Indexer loads the embedding model and opens the vector store, so it is
# created lazily (on server startup / first tool call) rather than at import
# time. This keeps `import docmcp.server` cheap and side-effect free.
_indexer: Indexer | None = None


def get_indexer() -> Indexer:
    global _indexer
    if _indexer is None:
        _indexer = Indexer(config)
    return _indexer


def _build_auth():
    """Build a bearer-token auth provider when DOCMCP_AUTH_TOKEN is set.

    Without a token the HTTP server is unauthenticated, which is only safe on
    a trusted/loopback network. Setting the token gates every tool call behind
    an ``Authorization: Bearer <token>`` header.
    """
    if not config.auth_token:
        logger.warning(
            "DOCMCP_AUTH_TOKEN is not set — the server is UNAUTHENTICATED. "
            "Only expose it on a trusted network."
        )
        return None

    from fastmcp.server.auth.providers.jwt import StaticTokenVerifier

    return StaticTokenVerifier(tokens={config.auth_token: {"sub": "docmcp"}})


@asynccontextmanager
async def lifespan(app):
    """Start background incremental indexing on server startup."""
    logger.info("DocMCP starting up — launching background indexing...")
    indexer = get_indexer()
    thread = threading.Thread(target=indexer.index_incremental, daemon=True)
    thread.start()
    yield
    logger.info("DocMCP shutting down.")


mcp = FastMCP("DocMCP", lifespan=lifespan, auth=_build_auth())


@mcp.tool
def search_documents(query: str, limit: int = 10) -> list[dict]:
    """Perform semantic search across all indexed documents.

    Args:
        query: Natural-language search query.
        limit: Max results to return (default: 10, max: 50).

    Returns:
        A list of search results with document name, matching text, score, and metadata.
    """
    results = get_indexer().search(query, limit)
    return [asdict(r) for r in results]


@mcp.tool
def get_document(document_name: str) -> dict:
    """Retrieve the full text content of a specific indexed document.

    Args:
        document_name: Filename or relative path of the document.

    Returns:
        The full extracted text content plus metadata.
    """
    doc = get_indexer().get_document(document_name)
    return asdict(doc)


@mcp.tool
def list_documents(filter: str = "") -> list[dict]:
    """List all documents currently in the index.

    Args:
        filter: Glob pattern to filter results (e.g. '*.pdf'). Empty string returns all.

    Returns:
        A list of document info objects.
    """
    pattern = filter if filter else None
    docs = get_indexer().list_documents(pattern)
    return [asdict(d) for d in docs]


@mcp.tool
def reindex(full: bool = False) -> dict:
    """Trigger re-indexing of the document corpus.

    Args:
        full: If True, rebuild entire index. Default False (incremental).

    Returns:
        Summary with counts of added, updated, removed, unchanged, and errors.
    """
    indexer = get_indexer()
    if full:
        summary = indexer.index_full()
    else:
        summary = indexer.index_incremental()
    return asdict(summary)


@mcp.tool
def index_status() -> dict:
    """Report the current state of the index.

    Returns:
        Index statistics including document count, chunk count, size, and configuration.
    """
    status = get_indexer().get_status()
    return asdict(status)


if __name__ == "__main__":
    mcp.run(transport="http", host=config.host, port=config.port)
