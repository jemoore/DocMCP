from __future__ import annotations

from docmcp.models import SearchResult, SearchResultMetadata


def normalize_score(distance: float) -> float:
    """Convert a cosine distance (0..2) to a 0-1 relevance score (higher is
    better). The collection is built with hnsw:space=cosine to match."""
    return max(0.0, min(1.0, 1.0 - distance / 2.0))


def search(
    collection, query: str, limit: int = 10, where: dict | None = None
) -> list[SearchResult]:
    """Perform semantic search against a ChromaDB collection, optionally
    restricted by a Chroma `where` metadata filter."""
    limit = max(1, min(limit, 50))

    count = collection.count()
    if count == 0:
        return []

    n_results = min(limit, count)

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    search_results: list[SearchResult] = []
    if not results["ids"] or not results["ids"][0]:
        return search_results

    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i in range(len(ids)):
        meta = metadatas[i]
        page_num = meta.get("page_number")
        if page_num == -1:
            page_num = None

        search_results.append(
            SearchResult(
                document_name=meta.get("document_name", ""),
                chunk_text=documents[i],
                score=normalize_score(distances[i]),
                metadata=SearchResultMetadata(
                    source_path=meta.get("source_path", ""),
                    page_number=page_num,
                    chunk_index=meta.get("chunk_index", 0),
                ),
            )
        )

    return search_results
