from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pymupdf

# Header/footer stripping only kicks in for documents long enough to make
# repetition statistically meaningful, and only for lines repeating on a
# solid majority of pages.
MIN_PAGES_FOR_STRIPPING = 5
REPEAT_THRESHOLD = 0.6


def extract_text(path: Path) -> list[tuple[str, dict]]:
    """Extract text from a PDF (or EPUB) file, returning (text, metadata)
    per page with repeating headers/footers removed."""
    pages: list[tuple[int, str]] = []
    doc = pymupdf.open(path)
    try:
        for page_num, page in enumerate(doc):
            pages.append((page_num + 1, page.get_text()))
    finally:
        doc.close()

    stripped = _strip_repeating_furniture([text for _, text in pages])
    return [
        (text, {"page_number": page_num})
        for (page_num, _), text in zip(pages, stripped)
        if text.strip()
    ]


def _normalize_line(line: str) -> str:
    """Normalize a line for repetition detection: page numbers vary per page,
    so digits are collapsed before comparing."""
    return re.sub(r"\d+", "#", line.strip()).lower()


def _strip_repeating_furniture(pages: list[str]) -> list[str]:
    """Remove running headers/footers (book title, chapter name, 'Page N')
    that repeat on most pages — they pollute every chunk's text and
    embedding, hurting retrieval quality on published books."""
    if len(pages) < MIN_PAGES_FOR_STRIPPING:
        return pages

    page_lines = [p.splitlines() for p in pages]
    first_counts: Counter[str] = Counter()
    last_counts: Counter[str] = Counter()
    for lines in page_lines:
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        if not nonempty:
            continue
        first_counts[_normalize_line(lines[nonempty[0]])] += 1
        last_counts[_normalize_line(lines[nonempty[-1]])] += 1

    threshold = max(3, int(len(pages) * REPEAT_THRESHOLD))
    strip_first = {k for k, c in first_counts.items() if c >= threshold and k}
    strip_last = {k for k, c in last_counts.items() if c >= threshold and k}
    if not strip_first and not strip_last:
        return pages

    result: list[str] = []
    for lines in page_lines:
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        drop: set[int] = set()
        if nonempty:
            if _normalize_line(lines[nonempty[0]]) in strip_first:
                drop.add(nonempty[0])
            if (
                nonempty[-1] not in drop
                and _normalize_line(lines[nonempty[-1]]) in strip_last
            ):
                drop.add(nonempty[-1])
        result.append(
            "\n".join(line for i, line in enumerate(lines) if i not in drop)
        )
    return result
