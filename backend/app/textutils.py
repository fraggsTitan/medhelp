"""Pure text helpers shared by ingestion (what gets indexed) and the engine (what gets used).

Two problems these solve, both measured on real guideline PDFs:

1. **Bibliography pages pollute retrieval.** In long guidelines roughly a quarter of all pages are
   reference lists (ACC/AHA 2019: 28 %, CDC STI 2021: 26 %, CDC opioids 2022: 22 %). Reference lists
   are packed with the same topic words as the real recommendations ("aspirin", "cardiovascular
   disease", "prevention"), so they rank highly for almost any question and push the genuine
   passages out of a small top-k. The LLM then sees only citations and answers "I do not know".
   Real content pages contain 0-1.3 citation markers per 1000 characters, reference pages 9+, so
   a simple density test separates them cleanly.

2. **Running headers/footers** ("MMWR / July 23, 2021 / Vol. 70 / No. 4 69 ...") are repeated on
   every page and add noise to every chunk's embedding.
"""
from __future__ import annotations

import os
import re
from collections import Counter

# Citation "markers" typical of reference lists: "et al", DOI/PubMed/PMID tags, "2014;64:959-68"
# volume/page patterns, "Available at", and the names of frequently cited journals.
_CITATION_PATTERNS = [
    re.compile(r"\bet al\b", re.I),
    re.compile(r"\bdoi\s*:|https?://doi\.org", re.I),
    re.compile(r"\[PubMed", re.I),
    re.compile(r"\b(?:19|20)\d{2}\s*;\s*\d{1,4}\s*(?:\(\s*\d+\s*\))?\s*:\s*[eE]?\d+"),
    re.compile(r"\b(?:19|20)\d{2}\s*[;:]\s*\d+\s*[\u2013-]\s*\d+"),
    re.compile(r"\bAvailable (?:at|from)\b|\baccessed\b", re.I),
    re.compile(r"\bPMID\b|\bPMCID\b"),
    re.compile(r"\bGeneva: World Health Organization\b|\bMMWR Morb Mortal Wkly Rep\b|\bN Engl J Med\b|\bLancet\b|\bJAMA\b"),
]

try:
    REFERENCE_DENSITY_THRESHOLD = float(os.getenv("REF_DENSITY_THRESHOLD", "5.0"))
except ValueError:
    REFERENCE_DENSITY_THRESHOLD = 5.0

MIN_CHARS_FOR_REFERENCE_TEST = 200  # tiny snippets give meaningless densities


def citation_density(text: str) -> float:
    """Citation markers per 1000 characters."""
    if not text:
        return 0.0
    hits = sum(len(p.findall(text)) for p in _CITATION_PATTERNS)
    return 1000.0 * hits / len(text)


def is_reference_text(text: str, threshold: float | None = None) -> bool:
    """True if *text* looks like a bibliography / reference list rather than guideline content."""
    if len(text) < MIN_CHARS_FOR_REFERENCE_TEST:
        return False
    return citation_density(text) >= (REFERENCE_DENSITY_THRESHOLD if threshold is None else threshold)


# --------------------------------------------------------------------------- #
# Repeated page headers / footers
# --------------------------------------------------------------------------- #
def _signature(line: str) -> str:
    """Normalise a line so 'MMWR / July 23, 2021 / Vol. 70 / No. 4 69' and '... 71' collide."""
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", line.strip().lower()))


def strip_repeated_lines(pages: list[str], min_pages: int = 4, min_fraction: float = 0.25, max_len: int = 160) -> list[str]:
    """Remove short lines that repeat (modulo digits) on many pages: running headers, footers, banners.

    A line is dropped when its normalised form occurs on at least ``max(min_pages, min_fraction * n_pages)``
    different pages. Long lines are never dropped, so genuine sentences that merely repeat are kept.
    """
    n_pages = len(pages)
    if n_pages < min_pages:
        return pages
    counts: Counter[str] = Counter()
    for page in pages:
        counts.update({_signature(line) for line in page.splitlines() if line.strip()})
    cutoff = max(min_pages, min_fraction * n_pages)
    repeated = {sig for sig, c in counts.items() if c >= cutoff and sig and len(sig) <= max_len}
    if not repeated:
        return pages
    cleaned: list[str] = []
    for page in pages:
        kept = [ln for ln in page.splitlines() if not (ln.strip() and _signature(ln) in repeated and len(ln) <= max_len)]
        cleaned.append("\n".join(kept))
    return cleaned


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token for English)."""
    return max(1, len(text) // 4)
