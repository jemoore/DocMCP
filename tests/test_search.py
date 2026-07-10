from __future__ import annotations

from unittest.mock import MagicMock

from docmcp.search import normalize_score, search


class TestNormalizeScore:
    """Cosine distance is bounded to [0, 2]: 0 = identical, 1 = orthogonal,
    2 = opposite."""

    def test_zero_distance_gives_one(self) -> None:
        assert normalize_score(0.0) == 1.0

    def test_max_distance_gives_zero(self) -> None:
        assert normalize_score(2.0) == 0.0

    def test_orthogonal_gives_half(self) -> None:
        assert normalize_score(1.0) == 0.5

    def test_score_always_between_zero_and_one(self) -> None:
        # Includes out-of-range distances (float noise), which must clamp.
        for d in [-0.001, 0.0, 0.1, 0.5, 1.0, 1.9, 2.0, 2.001]:
            s = normalize_score(d)
            assert 0.0 <= s <= 1.0


class TestSearch:
    def test_empty_collection(self) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 0
        results = search(mock_collection, "test query", limit=10)
        assert results == []

    def test_returns_mapped_results(self) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 2
        mock_collection.query.return_value = {
            "ids": [["id1", "id2"]],
            "documents": [["chunk one text", "chunk two text"]],
            "metadatas": [
                [
                    {
                        "document_name": "doc.pdf",
                        "source_path": "/data/docs/doc.pdf",
                        "page_number": 1,
                        "chunk_index": 0,
                    },
                    {
                        "document_name": "doc.pdf",
                        "source_path": "/data/docs/doc.pdf",
                        "page_number": 2,
                        "chunk_index": 1,
                    },
                ]
            ],
            "distances": [[0.1, 0.5]],
        }

        results = search(mock_collection, "test", limit=5)
        assert len(results) == 2
        assert results[0].document_name == "doc.pdf"
        assert results[0].chunk_text == "chunk one text"
        assert results[0].score > results[1].score
        assert results[0].metadata.page_number == 1
        assert results[1].metadata.chunk_index == 1

    def test_clamps_limit(self) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        search(mock_collection, "test", limit=999)
        mock_collection.query.assert_called_once_with(
            query_texts=["test"], n_results=50, where=None
        )

    def test_page_number_negative_one_becomes_none(self) -> None:
        mock_collection = MagicMock()
        mock_collection.count.return_value = 1
        mock_collection.query.return_value = {
            "ids": [["id1"]],
            "documents": [["text"]],
            "metadatas": [
                [
                    {
                        "document_name": "readme.md",
                        "source_path": "/data/docs/readme.md",
                        "page_number": -1,
                        "chunk_index": 0,
                    }
                ]
            ],
            "distances": [[0.2]],
        }

        results = search(mock_collection, "test", limit=10)
        assert results[0].metadata.page_number is None
