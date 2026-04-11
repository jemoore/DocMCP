"""Document parsers package."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from docmcp.parsers import html, markdown, pdf

# Map file extensions to their parser functions
_PARSERS: dict[str, Callable[[Path], list[tuple[str, dict]]]] = {
    ".pdf": pdf.extract_text,
    ".md": markdown.extract_text,
    ".markdown": markdown.extract_text,
    ".html": html.extract_text,
    ".htm": html.extract_text,
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
