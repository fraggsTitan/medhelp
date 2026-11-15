"""Small in-memory BM25 keyword index used next to the vector search ("hybrid retrieval").

Why it exists
-------------
Embedding search is great at meaning but can miss exact clinical terms, drug names and
abbreviations ("ceftriaxone", "SGLT2", "COR 2a"). BM25 is the opposite: excellent at exact
terms, blind to paraphrase. Fusing both (see ``MedHelpEngine._retrieve``) retrieves the right
passage far more reliably than either alone.

The index is built from the chunks that are *already stored in Qdrant* (via ``scroll``), so no
extra files or ingestion step are needed. A few thousand chunks fit comfortably in memory and
a query takes a few milliseconds. No third-party dependencies.
"""
from __future__ import annotations

import json
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from app import textutils

logger = logging.getLogger("medhelp.lexical")

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Very common English words plus words that occur on almost every page of a guideline
# ("recommend", "guideline", ...) and therefore carry no signal.
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does doing for from had has have having how
    i if in into is it its may might more most must no not of on or other our over per should so some such
    than that the their them then there these they this those to under up use used using was we were what
    when where which while who whom why will with within would you your
    guideline guidelines recommend recommended recommends recommendation recommendations say says said
    according tell please explain describe mention mentioned about
    """.split()
)


def _stem(word: str) -> str:
    """Tiny suffix stripper so 'treated/treating/treatments' meet 'treat/treatment'."""
    n = len(word)
    if n > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if n > 4 and word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if n > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        word, n = word[:-1], n - 1
    if n > 5 and word.endswith("ing"):
        return word[:-3]
    if n > 4 and word.endswith("ed"):
        return word[:-2]
    return word


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stop-words, light stemming."""
    return [_stem(t) for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class LexicalHit:
    id: str
    text: str
    meta: dict[str, Any]
    score: float
    matched_terms: int


@dataclass
class LexicalIndex:
    """Okapi BM25 over (id, text, metadata) records."""

    k1: float = 1.5
    b: float = 0.75
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    metas: list[dict[str, Any]] = field(default_factory=list)
    _doc_len: list[int] = field(default_factory=list)
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=lambda: defaultdict(list))
    _avg_len: float = 0.0

    # ----- building ----------------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[tuple[str, str, dict[str, Any]]]) -> "LexicalIndex":
        index = cls()
        for rec_id, text, meta in records:
            tokens = tokenize(text)
            if not tokens:
                continue
            doc_idx = len(index.ids)
            index.ids.append(rec_id)
            index.texts.append(text)
            index.metas.append(meta)
            index._doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                index._postings[term].append((doc_idx, tf))
        index._avg_len = (sum(index._doc_len) / len(index._doc_len)) if index._doc_len else 0.0
        return index

    def __len__(self) -> int:
        return len(self.ids)

    # ----- searching ---------------------------------------------------------
    def search(self, query: str, k: int = 8, min_matched: Optional[int] = None) -> list[LexicalHit]:
        """Top-*k* records for *query*.

        ``min_matched`` = minimum number of distinct query terms a record must contain. By default it
        is 2 for multi-term queries (a single shared word is rarely meaningful) and 1 otherwise.
        """
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not self.ids:
            return []
        if min_matched is None:
            min_matched = 2 if len(terms) >= 3 else 1

        n_docs = len(self.ids)
        scores: dict[int, float] = defaultdict(float)
        matched: dict[int, int] = defaultdict(int)
        for term in terms:
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            for doc_idx, tf in postings:
                norm = 1.0 - self.b + self.b * (self._doc_len[doc_idx] / (self._avg_len or 1.0))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / (tf + self.k1 * norm)
                matched[doc_idx] += 1

        ranked = sorted(
            (d for d in scores if matched[d] >= min_matched), key=lambda d: scores[d], reverse=True
        )[:k]
        return [
            LexicalHit(self.ids[d], self.texts[d], self.metas[d], round(scores[d], 4), matched[d]) for d in ranked
        ]

    def document_names(self) -> list[str]:
        """Distinct ``document_name`` values present in the index (insertion order)."""
        seen: dict[str, None] = {}
        for meta in self.metas:
            name = meta.get("document_name")
            if name:
                seen.setdefault(str(name), None)
        return list(seen)


# --------------------------------------------------------------------------- #
# Loading chunks back out of Qdrant
# --------------------------------------------------------------------------- #
_META_KEYS = ("document_name", "page_label", "pdf_page", "source_url", "file_name", "publisher", "ingest_version")


def payload_to_record(point_id: Any, payload: dict[str, Any]) -> Optional[tuple[str, str, dict[str, Any]]]:
    """Convert one Qdrant payload (as written by LlamaIndex) to ``(id, text, metadata)``.

    LlamaIndex stores the node as JSON under ``_node_content``; metadata keys are also flattened
    at the top level. Both shapes are handled, and ``None`` is returned if no text can be found.
    """
    payload = payload or {}
    text = payload.get("text") or payload.get("document") or ""
    meta: dict[str, Any] = {k: payload[k] for k in _META_KEYS if payload.get(k) not in (None, "")}
    raw = payload.get("_node_content")
    if raw:
        try:
            node = json.loads(raw) if isinstance(raw, str) else dict(raw)
            text = text or node.get("text") or ""
            for key, value in (node.get("metadata") or {}).items():
                if key in _META_KEYS and value not in (None, "") and key not in meta:
                    meta[key] = value
        except (ValueError, TypeError):
            pass
    text = (text or "").strip()
    if not text:
        return None
    return str(point_id), text, meta


def load_index_from_qdrant(
    client: Any, collection: str, batch_size: int = 512, exclude_references: bool = True
) -> LexicalIndex:
    """Scroll through every point of *collection* and build a :class:`LexicalIndex`.

    Bibliography chunks are left out (``exclude_references``): they are full of topic words but never
    contain the answer, so they would only crowd real passages out of the keyword results.
    """
    records: list[tuple[str, str, dict[str, Any]]] = []
    skipped = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            rec = payload_to_record(point.id, point.payload or {})
            if not rec:
                continue
            if exclude_references and textutils.is_reference_text(rec[1]):
                skipped += 1
                continue
            records.append(rec)
        if offset is None or not points:
            break
    index = LexicalIndex.from_records(records)
    logger.info("Keyword index built from Qdrant: %d chunks (%d reference chunks skipped).", len(index), skipped)
    return index
