from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from docmcp.parsers import SUPPORTED_EXTENSIONS, parse_file, read_text
from docmcp.parsers.html import extract_text as html_extract
from docmcp.parsers.markdown import extract_text as md_extract


class TestReadText:
    def test_reads_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_text("héllo wörld", encoding="utf-8")
        assert read_text(f) == "héllo wörld"

    def test_falls_back_on_invalid_utf8(self, tmp_path: Path) -> None:
        f = tmp_path / "b.txt"
        f.write_bytes(b"caf\xe9")  # latin-1 'é', invalid as UTF-8
        # Must not raise; the undecodable byte is replaced.
        assert read_text(f).startswith("caf")


class TestParsersHandleNonUtf8:
    def test_html_non_utf8_does_not_crash(self, tmp_path: Path) -> None:
        f = tmp_path / "x.html"
        f.write_bytes(b"<p>caf\xe9</p>")
        result = html_extract(f)
        assert result and "caf" in result[0][0]

    def test_markdown_non_utf8_does_not_crash(self, tmp_path: Path) -> None:
        f = tmp_path / "x.md"
        f.write_bytes(b"# t\xe9st heading")
        result = md_extract(f)
        assert result and "st heading" in result[0][0]


class TestMarkdownParser:
    def test_extracts_plain_text(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hello\n\nThis is a **test** document.\n")
        result = md_extract(md_file)
        assert len(result) == 1
        text, metadata = result[0]
        assert "Hello" in text
        assert "test" in text
        assert "<" not in text  # No HTML tags
        assert metadata == {}

    def test_empty_file_returns_empty(self, tmp_path: Path) -> None:
        md_file = tmp_path / "empty.md"
        md_file.write_text("")
        result = md_extract(md_file)
        assert result == []


class TestHtmlParser:
    def test_extracts_text_strips_tags(self, tmp_path: Path) -> None:
        html_file = tmp_path / "test.html"
        html_file.write_text(
            "<html><body><h1>Title</h1><p>Content here.</p></body></html>"
        )
        result = html_extract(html_file)
        assert len(result) == 1
        text, metadata = result[0]
        assert "Title" in text
        assert "Content here." in text
        assert "<" not in text
        assert metadata == {}

    def test_strips_scripts_and_styles(self, tmp_path: Path) -> None:
        html_file = tmp_path / "script.html"
        html_file.write_text(
            "<html><head><style>body{color:red}</style></head>"
            "<body><script>alert('x')</script><p>Visible</p></body></html>"
        )
        result = html_extract(html_file)
        assert len(result) == 1
        text, _ = result[0]
        assert "Visible" in text
        assert "alert" not in text
        assert "color:red" not in text

    def test_empty_html_returns_empty(self, tmp_path: Path) -> None:
        html_file = tmp_path / "empty.html"
        html_file.write_text("<html><body></body></html>")
        result = html_extract(html_file)
        assert result == []


class TestPdfParser:
    def test_extract_text_with_mock(self) -> None:
        """Test PDF parser using mocked pymupdf."""
        mock_page = MagicMock()
        mock_page.get_text.return_value = "Page one content"

        mock_doc = MagicMock()
        mock_doc.__iter__ = MagicMock(return_value=iter([mock_page]))
        mock_doc.__enter__ = MagicMock(return_value=mock_doc)
        mock_doc.__exit__ = MagicMock(return_value=False)

        with patch("docmcp.parsers.pdf.pymupdf") as mock_pymupdf:
            mock_pymupdf.open.return_value = mock_doc
            from docmcp.parsers.pdf import extract_text as pdf_extract

            result = pdf_extract(Path("/fake/doc.pdf"))

        assert len(result) == 1
        text, meta = result[0]
        assert text == "Page one content"
        assert meta["page_number"] == 1


class TestParseFileDispatch:
    def test_supported_extensions(self) -> None:
        assert ".pdf" in SUPPORTED_EXTENSIONS
        assert ".md" in SUPPORTED_EXTENSIONS
        assert ".markdown" in SUPPORTED_EXTENSIONS
        assert ".html" in SUPPORTED_EXTENSIONS
        assert ".htm" in SUPPORTED_EXTENSIONS

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            parse_file(txt_file)

    def test_dispatches_markdown(self, tmp_path: Path) -> None:
        md_file = tmp_path / "test.md"
        md_file.write_text("# Hi\n\nWorld\n")
        result = parse_file(md_file)
        assert len(result) == 1
        assert "Hi" in result[0][0]

    def test_dispatches_html(self, tmp_path: Path) -> None:
        html_file = tmp_path / "test.htm"
        html_file.write_text("<p>Hello</p>")
        result = parse_file(html_file)
        assert len(result) == 1
        assert "Hello" in result[0][0]
