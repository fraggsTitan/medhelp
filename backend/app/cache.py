"""Semantic answer cache (feature: cache).

Repeated or near-identical questions skip retrieval-to-LLM entirely, which matters because Groq's free tier
allows only about two full answers per minute for the whole app.

Safety rules baked in:
* The cache key is the *standalone* question (follow-ups are rewritten first); the engine refuses to cache a
  follow-up whose rewrite failed, so "what about children?" from one chat can never be answered with another
  chat's context.
* Entries are scoped (single-guideline vs. each comparison set) so a comparison never answers a plain question.
* A hit needs cosine similarity >= ``threshold`` (default 0.97) or an identical normalised question.
  Only successful answers are stored. Entries expire; the cache is size-bounded (oldest evicted).
"""
from __future__ import annotations

import copy
import re
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

import numpy as np

_NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_question(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


class SemanticCache:
    def __init__(
        self,
        max_entries: int = 300,
        ttl_seconds: float = 43200,
        threshold: float = 0.97,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_entries, self.ttl, self.threshold, self._clock = max_entries, ttl_seconds, threshold, clock
        self._entries: "OrderedDict[int, dict[str, Any]]" = OrderedDict()
        self._next = 0
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def _sweep(self) -> None:
        now = self._clock()
        for key in [k for k, e in self._entries.items() if now - e["at"] > self.ttl]:
            del self._entries[key]

    @staticmethod
    def _unit(vector: Optional[list[float]]) -> Optional[np.ndarray]:
        if vector is None:
            return None
        v = np.asarray(vector, dtype=np.float32)
        n = float(np.linalg.norm(v))
        return v / n if n else None

    def get(self, scope: str, question: str, vector: Optional[list[float]] = None) -> Optional[dict[str, Any]]:
        norm, unit = normalize_question(question), self._unit(vector)
        with self._lock:
            self._sweep()
            best_key, best_sim = None, -1.0
            for key, e in self._entries.items():
                if e["scope"] != scope:
                    continue
                if e["norm"] == norm:
                    best_key, best_sim = key, 1.0
                    break
                if unit is not None and e["vec"] is not None:
                    sim = float(np.dot(unit, e["vec"]))
                    if sim > best_sim:
                        best_key, best_sim = key, sim
            if best_key is not None and best_sim >= self.threshold:
                self._entries.move_to_end(best_key)
                self.hits += 1
                result = copy.deepcopy(self._entries[best_key]["result"])
                result["_cache_similarity"] = round(best_sim, 4)
                return result
            self.misses += 1
            return None

    def put(self, scope: str, question: str, vector: Optional[list[float]], result: dict[str, Any]) -> None:
        with self._lock:
            self._sweep()
            norm = normalize_question(question)
            for key, e in list(self._entries.items()):  # replace an identical entry
                if e["scope"] == scope and e["norm"] == norm:
                    del self._entries[key]
            self._next += 1
            self._entries[self._next] = {
                "scope": scope, "norm": norm, "vec": self._unit(vector), "result": copy.deepcopy(result), "at": self._clock(),
            }
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {"entries": len(self._entries), "hits": self.hits, "misses": self.misses, "max_entries": self.max_entries}
