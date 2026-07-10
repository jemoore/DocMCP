"""Document parsers package."""

from __future__ import annotations

from pathlib import Path
from typing import Callable


def read_text(path: Path) -> str:
    """Read a text file, tolerating non-UTF-8 content.

    Tries UTF-8 (the common case) and falls back to a lenient decode so a
    single oddly-encoded file does not abort indexing.
    """
    data = path.read_bytes()
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


from docmcp.parsers import html, markdown, pdf, text  # noqa: E402  (avoid circular import)

# Map file extensions to their parser functions
_PARSERS: dict[str, Callable[[Path], list[tuple[str, dict]]]] = {
    ".pdf": pdf.extract_text,
    ".epub": pdf.extract_text,  # pymupdf reads EPUB natively
    ".md": markdown.extract_text,
    ".markdown": markdown.extract_text,
    ".html": html.extract_text,
    ".htm": html.extract_text,
    ".txt": text.extract_text,
}

SUPPORTED_EXTENSIONS = set(_PARSERS.keys())


def parse_file(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a file using the appropriate parser.

    Raises ValueError if the file extension is not supported.
    """
    ext = path.suffix.lower()
    parser = _PARSERS.get(ext)
    if parser is None:
        raise ValueError(f"Unsupported file extension: {ext}")
    return parser(path)
