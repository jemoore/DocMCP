"""End-to-end tests that build a real Indexer (ChromaDB + embedding model).

These are heavier than the unit tests since constructing an Indexer loads the
sentence-transformer model, but they cover behavior that only emerges from the
full parse -> chunk -> embed -> store -> retrieve pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pytest

from docmcp.config import Config
from docmcp.indexer import Indexer


@pytest.fixture
def make_indexer(tmp_path: Path) -> Callable[..., Indexer]:
    def _make(
        doc_dirs: list[Path], chunk_size: int = 200, chunk_overlap: int = 40
    ) -> Indexer:
        config = Config(
            doc_dirs=doc_dirs,
            index_dir=tmp_path / "index",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return Indexer(config)

    return _make


class TestFullTextRetrieval:
    def test_get_document_returns_clean_full_text(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        # The sentence repeats exactly 100 times. A chunk reconstruction would
        # duplicate the overlap regions and inflate that count; the stored
        # full-text blob must return it verbatim.
        body = "# Title\n\n" + "sentence number one. " * 100
        (docs / "note.md").write_text(body)

        ix = make_indexer([docs])
        ix.index_full()

        doc = ix.get_document("note.md")
        assert doc.document_name == "note.md"
        assert doc.content.count("sentence number one.") == 100

    def test_get_document_missing_raises(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("hello world " * 20)
        ix = make_indexer([docs])
        ix.index_full()

        with pytest.raises(ValueError, match="Document not found"):
            ix.get_document("nope.md")


class TestNameCollisions:
    def test_multi_dir_namespaces_identical_names(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        (a / "readme.md").write_text("alpha content here " * 30)
        (b / "readme.md").write_text("beta content here " * 30)

        ix = make_indexer([a, b])
        ix.index_full()

        names = sorted(d.document_name for d in ix.list_documents())
        assert names == ["a/readme.md", "b/readme.md"]
        assert "alpha" in ix.get_document("a/readme.md").content
        assert "beta" in ix.get_document("b/readme.md").content

    def test_ambiguous_name_raises(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("content here " * 30)
        ix = make_indexer([docs])
        ix.index_full()

        # Simulate a second file that resolved to the same document name by
        # injecting a chunk pointing at a different source path.
        ix._collection.add(
            ids=["other::chunk_0"],
            documents=["unrelated text"],
            metadatas=[
                {
                    "source_path": "/other/a.md",
                    "document_name": "a.md",
                    "chunk_index": 0,
                    "page_number": -1,
                }
            ],
        )

        with pytest.raises(ValueError, match="Ambiguous document name"):
            ix.get_document("a.md")


class TestIncrementalIndexing:
    def test_add_unchanged_update_remove_lifecycle(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "one.md"
        f.write_text("first document content " * 20)
        ix = make_indexer([docs])

        added = ix.index_incremental()
        assert added.added == 1
        assert added.updated == 0
        assert added.removed == 0

        unchanged = ix.index_incremental()
        assert unchanged.unchanged == 1
        assert unchanged.added == 0

        f.write_text("completely different words now " * 20)
        updated = ix.index_incremental()
        assert updated.updated == 1
        assert updated.added == 0

        f.unlink()
        removed = ix.index_incremental()
        assert removed.removed == 1
        assert ix.list_documents() == []

    def test_removed_document_deletes_fulltext_blob(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "x.md"
        f.write_text("some content to index " * 30)
        ix = make_indexer([docs])

        ix.index_incremental()
        assert len(list(ix._fulltext_dir.glob("*.txt"))) == 1

        f.unlink()
        ix.index_incremental()
        assert list(ix._fulltext_dir.glob("*.txt")) == []


class TestSearchIntegration:
    def test_search_ranks_relevant_document_first(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "python.md").write_text(
            "Python is a programming language used to write software and code. " * 10
        )
        (docs / "cooking.md").write_text(
            "Baking bread needs flour, water, yeast and salt baked in an oven. " * 10
        )
        ix = make_indexer([docs])
        ix.index_full()

        results = ix.search("how do I write software source code", limit=5)
        assert results
        assert results[0].document_name == "python.md"
        # Scores are normalized into (0, 1].
        assert all(0.0 < r.score <= 1.0 for r in results)
