"""Citation verification (no LLM call, no network).

For every sentence of the answer that carries a ``[Document, Page N]`` citation we check, against the exact
passage that was cited:

* how many of the sentence's content words (stemmed, stop-words removed) appear in that passage (for a sentence
  with several citations: in the combined cited passages, and each citation must contribute something), and
* whether every *number* in the sentence (doses, thresholds, percentages) appears in the passage.

and we return the best-matching sentence of the passage as "evidence" so the UI can show the supporting quote.

This is a lexical sanity check, not a proof: it catches citations that point at the wrong passage and numbers
that were not in the source, but it cannot judge meaning. Statuses:

    supported         most content words found and all numbers present
    partial           some overlap, or a number is missing
    unsupported       little overlap with the cited passage
    source_not_found  the citation does not match any passage that was given to the model
"""
from __future__ import annotations

import re
from typing import Any, Optional

from app.lexical import tokenize

CITATION_RE = re.compile(r"\[([^\[\]]{2,200}?),\s*(?:Page|page|p\.?)\s*([^\[\]]{1,20}?)\]")
_NUM_RE = re.compile(r"(?<![\w.])\d+(?:[.,]\d+)?(?![\w])")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\[*(])")

SUPPORTED_AT = 0.6
PARTIAL_AT = 0.35
MIN_CLAIM_TOKENS = 3
MIN_OWN_COVERAGE = 0.15  # each of several citations must share at least this much with the sentence


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentence-like pieces (lines first, then sentence punctuation)."""
    out: list[str] = []
    for line in text.splitlines():
        line = _LIST_PREFIX_RE.sub("", line).strip()
        if not line:
            continue
        out.extend(p.strip() for p in _SENT_SPLIT_RE.split(line) if p.strip())
    return out


def _numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in _NUM_RE.findall(text)}


def _clean_claim(sentence: str) -> str:
    claim = CITATION_RE.sub(" ", sentence)
    claim = re.sub(r"\*\*[^*]{0,40}:\*\*", " ", claim)  # "**Strength of Recommendation:**" style labels
    return " ".join(claim.replace("*", " ").split())


def _find_source(doc: str, page: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    doc_n, page_n = doc.strip().lower(), page.strip().lower()
    matches = []
    for s in sources:
        name = str(s.get("document_name", "")).strip().lower()
        pages = {str(s.get("page_number", "")).strip().lower(), str(s.get("pdf_page", "")).strip().lower()}
        if page_n in pages and (name == doc_n or doc_n in name or name in doc_n):
            matches.append(s)
    return matches


def _best_evidence(claim_tokens: set[str], passage: str, limit: int = 320) -> str:
    best, best_score = "", -1.0
    for sent in split_sentences(passage) or [passage]:
        toks = set(tokenize(sent))
        if not toks:
            continue
        score = len(claim_tokens & toks) / (len(toks) ** 0.5)  # favour focused sentences over long ones
        if score > best_score:
            best, best_score = sent, score
    best = " ".join(best.split())
    return best if len(best) <= limit else best[: limit - 1].rstrip() + "…"


def verify_citations(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One result dict per (cited sentence, citation) pair; ``[]`` if the answer has no citations."""
    results: list[dict[str, Any]] = []
    for sentence in split_sentences(answer):
        cites = CITATION_RE.findall(sentence)
        if not cites:
            continue
        claim = _clean_claim(sentence)
        claim_tokens = set(tokenize(claim))
        if len(claim_tokens) < MIN_CLAIM_TOKENS:
            continue
        claim_numbers = _numbers(claim)
        seen: set[tuple[str, str]] = set()
        found: list[tuple[dict[str, Any], Optional[dict[str, Any]], float]] = []  # (entry, best source, own coverage)
        for doc, page in cites:
            key = (doc.strip().lower(), page.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            entry: dict[str, Any] = {
                "claim": claim if len(claim) <= 260 else claim[:259].rstrip() + "…",
                "document": doc.strip(),
                "page": page.strip(),
                "status": "source_not_found",
                "coverage": 0.0,
                "missing_numbers": [],
                "evidence": "",
            }
            best: Optional[dict[str, Any]] = None
            best_cov = -1.0
            for src in _find_source(doc, page, sources):
                cov = len(claim_tokens & set(tokenize(str(src.get("source_text", ""))))) / len(claim_tokens)
                if cov > best_cov:
                    best, best_cov = src, cov
            found.append((entry, best, max(best_cov, 0.0)))
            results.append(entry)

        # A sentence that cites several passages (typical when comparing publishers) is judged against the
        # *combined* cited text; each citation must still contribute something of its own.
        located = [(e, src, cov) for e, src, cov in found if src is not None]
        union_tokens: set[str] = set()
        union_numbers: set[str] = set()
        for _, src, _ in located:
            text = str(src.get("source_text", ""))
            union_tokens |= set(tokenize(text))
            union_numbers |= _numbers(text)
        union_cov = len(claim_tokens & union_tokens) / len(claim_tokens) if located else 0.0
        union_missing = sorted(n for n in claim_numbers if n not in union_numbers)
        for entry, src, own_cov in located:
            passage = str(src.get("source_text", ""))
            if len(located) == 1:
                coverage, missing = own_cov, sorted(n for n in claim_numbers if n not in _numbers(passage))
            else:
                coverage, missing = union_cov, union_missing
            if len(located) > 1 and own_cov < MIN_OWN_COVERAGE:
                status = "unsupported"  # this citation adds almost nothing to the sentence
            elif coverage >= SUPPORTED_AT and not missing:
                status = "supported"
            elif coverage >= PARTIAL_AT:
                status = "partial"
            else:
                status = "unsupported"
            entry.update(status=status, coverage=round(coverage, 2), missing_numbers=missing,
                         evidence=_best_evidence(claim_tokens, passage))
    return results


def verification_warning(results: list[dict[str, Any]]) -> Optional[str]:
    """One user-facing warning when some citations could not be matched to their passage."""
    bad = [r for r in results if r["status"] in ("unsupported", "source_not_found")]
    numbers = [r for r in results if r["status"] == "partial" and r["missing_numbers"]]
    parts = []
    if bad:
        parts.append(f"{len(bad)} cited statement(s) could not be matched to the passage they cite")
    if numbers:
        parts.append(f"{len(numbers)} statement(s) contain number(s) that do not appear in the cited passage")
    if not parts:
        return None
    return "Citation check: " + "; ".join(parts) + ". Please verify against the source before relying on them."


def summarize(results: list[dict[str, Any]]) -> dict[str, int]:
    out = {"supported": 0, "partial": 0, "unsupported": 0, "source_not_found": 0}
    for r in results:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out
