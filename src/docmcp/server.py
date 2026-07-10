from __future__ import annotations

import logging
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
    get_indexer().start_reindex(full=False)
    yield
    logger.info("DocMCP shutting down.")


mcp = FastMCP("DocMCP", lifespan=lifespan, auth=_build_auth())


@mcp.tool
def search_documents(
    query: str, limit: int = 10, document_filter: str = ""
) -> list[dict]:
    """Perform semantic search across indexed documents.

    Args:
        query: Natural-language search query.
        limit: Max results to return (default: 10, max: 50).
        document_filter: Optional glob pattern to restrict the search to
            matching document names (e.g. 'python*.pdf' or a full document
            name from list_documents). Empty string searches everything.

    Returns:
        A list of search results with document name, matching text, score, and
        metadata (source path, page number, chunk index). Pass a result's
        document name and chunk index to get_context for the surrounding text.
    """
    results = get_indexer().search(query, limit, document_filter or None)
    return [asdict(r) for r in results]


@mcp.tool
def get_context(
    document_name: str, chunk_index: int, before: int = 2, after: int = 2
) -> dict:
    """Fetch the text surrounding a search hit for fuller context.

    Search results are short chunks; this returns the passage around one —
    the chunks from `chunk_index - before` through `chunk_index + after`,
    merged into one continuous text with the chunk overlap removed.

    Args:
        document_name: Document name from a search result.
        chunk_index: Chunk index from a search result's metadata.
        before: Chunks of preceding context to include (default 2).
        after: Chunks of following context to include (default 2).

    Returns:
        The merged passage plus the chunk and page range it covers.
    """
    ctx = get_indexer().get_context(document_name, chunk_index, before, after)
    return asdict(ctx)


@mcp.tool
def get_document(
    document_name: str,
    page_start: int | None = None,
    page_end: int | None = None,
    offset: int = 0,
    max_chars: int = 50000,
) -> dict:
    """Retrieve the text content of a specific indexed document.

    Whole books can be megabytes of text, so results are windowed. Use
    page_start/page_end to select a page range (PDFs only), and offset with
    the returned total_chars to page through longer texts. When the result is
    cut short, `truncated` is true.

    Args:
        document_name: Filename or relative path of the document.
        page_start: First page to include (PDFs only, 1-based, inclusive).
        page_end: Last page to include (PDFs only, inclusive).
        offset: Character offset into the selected text to start from.
        max_chars: Max characters to return (default 50000; <= 0 for no limit).

    Returns:
        The extracted text slice plus metadata (total_chars, offset, truncated,
        page_count, size_bytes, last_modified).
    """
    doc = get_indexer().get_document(
        document_name,
        page_start=page_start,
        page_end=page_end,
        offset=offset,
        max_chars=max_chars,
    )
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

    Indexing runs in the background (a large corpus can take hours); this
    returns immediately. Poll index_status for progress and, once finished,
    the last-run summary (added/updated/removed/unchanged/empty/errors).

    Args:
        full: If True, rebuild entire index. Default False (incremental).

    Returns:
        Whether a run was started, or a note that one is already in progress.
    """
    started = get_indexer().start_reindex(full=full)
    if started:
        return {
            "started": True,
            "message": f"{'Full' if full else 'Incremental'} indexing started "
            "in the background. Poll index_status for progress.",
        }
    return {
        "started": False,
        "message": "An indexing run is already in progress. "
        "Poll index_status for progress.",
    }


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
