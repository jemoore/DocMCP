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
        assert len(list(ix._fulltext_dir.glob("*.json"))) == 1

        f.unlink()
        ix.index_incremental()
        assert list(ix._fulltext_dir.glob("*.json")) == []


class TestEmptyDocuments:
    def test_empty_file_recorded_and_not_reprocessed(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        # Parses to no text (like a scanned/image-only PDF would).
        (docs / "empty.md").write_text("   \n\n   ")
        ix = make_indexer([docs])

        first = ix.index_incremental()
        assert first.empty == 1
        assert first.added == 0

        # Second pass must see the unchanged hash and skip it entirely —
        # previously empty files were re-parsed and counted "added" every run.
        second = ix.index_incremental()
        assert second.unchanged == 1
        assert second.empty == 0
        assert second.added == 0

        # Surfaced in listings with zero chunks so OCR-needed books are visible.
        listed = ix.list_documents()
        assert len(listed) == 1
        assert listed[0].chunk_count == 0

        with pytest.raises(ValueError, match="no extractable text"):
            ix.get_document("empty.md")


class TestGetDocumentWindowing:
    def test_offset_and_max_chars_truncate(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        body = "abcdefghij" * 100  # 1000 chars of indexable text
        (docs / "long.md").write_text(body)
        ix = make_indexer([docs])
        ix.index_full()

        full = ix.get_document("long.md")
        total = full.total_chars

        doc = ix.get_document("long.md", offset=0, max_chars=100)
        assert len(doc.content) == 100
        assert doc.truncated is True
        assert doc.total_chars == total

        # Continue from the offset; the pieces must tile the full text.
        rest = ix.get_document("long.md", offset=100, max_chars=0)
        assert doc.content + rest.content == full.content
        assert rest.truncated is False

    def test_page_range_selects_pdf_pages(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        import pymupdf

        docs = tmp_path / "docs"
        docs.mkdir()
        pdf = pymupdf.open()
        for n in range(1, 4):
            page = pdf.new_page()
            page.insert_text((72, 72), f"page {n} marker content here")
        pdf.save(docs / "book.pdf")
        pdf.close()

        ix = make_indexer([docs])
        ix.index_full()

        doc = ix.get_document("book.pdf", page_start=2, page_end=2)
        assert "page 2 marker" in doc.content
        assert "page 1 marker" not in doc.content
        assert "page 3 marker" not in doc.content
        assert doc.page_count == 3

    def test_page_range_rejected_for_non_pdf(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "note.md").write_text("plain markdown content " * 20)
        ix = make_indexer([docs])
        ix.index_full()

        with pytest.raises(ValueError, match="no page information"):
            ix.get_document("note.md", page_start=1, page_end=2)


class TestBatchedAdds:
    def test_document_larger_than_batch_size_indexes_fully(
        self,
        tmp_path: Path,
        make_indexer: Callable[..., Indexer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import docmcp.indexer as indexer_mod

        # Shrink the batch size so a modest document spans several batches,
        # standing in for a book that exceeds Chroma's real max batch size.
        monkeypatch.setattr(indexer_mod, "ADD_BATCH_SIZE", 5)

        docs = tmp_path / "docs"
        docs.mkdir()
        paragraphs = "\n\n".join(
            f"paragraph {i} with some distinct filler words here" for i in range(80)
        )
        (docs / "big.md").write_text(paragraphs)
        ix = make_indexer([docs])

        summary = ix.index_full()
        assert summary.added == 1
        assert not summary.errors

        listed = ix.list_documents()
        assert listed[0].chunk_count > 5  # spans multiple batches
        assert ix._collection.count() == listed[0].chunk_count


class TestBackgroundReindex:
    def test_start_reindex_runs_in_background(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        import time

        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("background indexing content " * 20)
        ix = make_indexer([docs])

        assert ix.start_reindex(full=False) is True
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            status = ix.get_status()
            if not status.indexing.in_progress and status.last_run is not None:
                break
            time.sleep(0.05)

        status = ix.get_status()
        assert status.indexing.in_progress is False
        assert status.last_run is not None
        assert status.last_run.added == 1
        assert status.indexing.files_total == 1
        assert status.indexing.files_processed == 1

    def test_start_reindex_refused_while_run_in_progress(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        ix = make_indexer([docs])

        # Simulate a run in progress by holding the run lock.
        assert ix._run_lock.acquire(blocking=False)
        try:
            assert ix.start_reindex(full=True) is False
        finally:
            ix._run_lock.release()


class TestFullRebuildKeepsIndexLive:
    def test_search_works_during_and_after_swap(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("searchable alpha content " * 20)
        ix = make_indexer([docs])
        ix.index_full()

        # A second full rebuild must leave search working (collection swap).
        ix.index_full()
        results = ix.search("alpha content", limit=3)
        assert results
        assert results[0].document_name == "a.md"

        # Stale full-text blobs are cleaned up after the swap.
        blobs = list(ix._fulltext_dir.iterdir())
        assert len(blobs) == 1


class TestHashFastPath:
    def test_unchanged_files_are_not_rehashed(
        self,
        tmp_path: Path,
        make_indexer: Callable[..., Indexer],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import docmcp.indexer as indexer_mod

        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "book.md"
        f.write_text("stable content that never changes " * 20)
        ix = make_indexer([docs])
        ix.index_incremental()

        calls = 0
        real_hash = indexer_mod.compute_hash

        def counting_hash(path: Path) -> str:
            nonlocal calls
            calls += 1
            return real_hash(path)

        monkeypatch.setattr(indexer_mod, "compute_hash", counting_hash)

        # Same size + mtime: the file must not be read at all.
        summary = ix.index_incremental()
        assert summary.unchanged == 1
        assert calls == 0

        # New mtime but same content: hashed once, then the refreshed stat
        # info lets the following pass skip the hash again.
        import os

        st = f.stat()
        os.utime(f, ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
        summary = ix.index_incremental()
        assert summary.unchanged == 1
        assert calls == 1

        summary = ix.index_incremental()
        assert summary.unchanged == 1
        assert calls == 1

    def test_changed_content_still_detected(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        f = docs / "book.md"
        f.write_text("original words in this file " * 20)
        ix = make_indexer([docs])
        ix.index_incremental()

        f.write_text("completely rewritten body text " * 20)
        summary = ix.index_incremental()
        assert summary.updated == 1


class TestDistanceMetric:
    def test_collection_uses_cosine_space(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("some indexable content " * 20)
        ix = make_indexer([docs])
        assert ix._collection.configuration_json["hnsw"]["space"] == "cosine"

        # The rebuilt-and-swapped collection must keep the cosine metric.
        ix.index_full()
        assert ix._collection.configuration_json["hnsw"]["space"] == "cosine"


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

    def test_document_filter_restricts_search(
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

        # Even for a software query, a filter scoped to the cooking doc must
        # only return chunks from it.
        results = ix.search(
            "how do I write software source code",
            limit=5,
            document_filter="cooking*",
        )
        assert results
        assert all(r.document_name == "cooking.md" for r in results)

        # Exact-name filters work too, and non-matching filters return empty.
        results = ix.search("bread", limit=5, document_filter="cooking.md")
        assert results
        assert ix.search("bread", limit=5, document_filter="nomatch*") == []


class TestGetContext:
    def test_context_window_merges_chunks(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        # Distinct numbered words so position in the merged text is checkable.
        body = " ".join(f"word{i:04d}" for i in range(400))
        (docs / "long.md").write_text(body)
        ix = make_indexer([docs])
        ix.index_full()

        chunk_count = ix.list_documents()[0].chunk_count
        assert chunk_count >= 5
        target = chunk_count // 2

        ctx = ix.get_context("long.md", target, before=1, after=1)
        assert ctx.first_chunk_index == target - 1
        assert ctx.last_chunk_index == target + 1
        assert ctx.chunk_index == target
        # Overlap between adjacent chunks must be de-duplicated: the merged
        # passage is a verbatim slice of the original document.
        assert ctx.text in body

    def test_window_clamped_at_document_edges(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        body = " ".join(f"word{i:04d}" for i in range(400))
        (docs / "long.md").write_text(body)
        ix = make_indexer([docs])
        ix.index_full()

        ctx = ix.get_context("long.md", 0, before=3, after=1)
        assert ctx.first_chunk_index == 0
        assert ctx.last_chunk_index == 1

    def test_missing_document_or_chunk_raises(
        self, tmp_path: Path, make_indexer: Callable[..., Indexer]
    ) -> None:
        docs = tmp_path / "docs"
        docs.mkdir()
        (docs / "a.md").write_text("short content " * 20)
        ix = make_indexer([docs])
        ix.index_full()

        with pytest.raises(ValueError, match="No chunks found"):
            ix.get_context("nope.md", 0)
        # Out of range entirely: the window matches nothing.
        with pytest.raises(ValueError, match="No chunks found"):
            ix.get_context("a.md", 9999)
        # Window overlaps real chunks but the requested chunk doesn't exist.
        chunk_count = ix.list_documents()[0].chunk_count
        with pytest.raises(ValueError, match="not found"):
            ix.get_context("a.md", chunk_count, before=1, after=0)
