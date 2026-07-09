from __future__ import annotations

from pathlib import Path

from bs4 import BeautifulSoup

from docmcp.parsers import read_text


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from an HTML file, stripping scripts and styles."""
    raw = read_text(path)
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script and style elements
    for element in soup(["script", "style"]):
        element.decompose()

    text = soup.get_text(separator="\n")
    if not text.strip():
        return []
    return [(text, {})]
