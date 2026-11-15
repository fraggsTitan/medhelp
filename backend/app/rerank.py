"""Optional passage re-ranking through an API (no model in RAM).

The engine fuses semantic and keyword hits, hands the best ``RERANK_CANDIDATES`` of them to the reranker, and
then packs the top ``TOP_K`` as before; so reranking only changes the *order*, never how many passages are used.
Fails soft: any error leaves the original order untouched.

Voyage's free allowance is 200M reranker tokens, but without a payment method on the account the rate limit is
only 3 requests/minute (one rerank call per question), so enable it only if that is enough for your traffic.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import requests

from app import config
from app.errors import MedHelpError

logger = logging.getLogger("medhelp.rerank")


class RerankError(MedHelpError):
    status_code = 502


class VoyageReranker:
    def __init__(self, api_key: str, model: str = "rerank-2.5-lite", base_url: str = "https://api.voyageai.com/v1",
                 timeout: float = 15, session: Optional[requests.Session] = None, max_chars: int = 3000) -> None:
        self.api_key, self.model, self.timeout, self.max_chars = api_key, model, timeout, max_chars
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def rerank(self, query: str, texts: list[str], top_n: Optional[int] = None) -> list[tuple[int, float]]:
        """``[(index_into_texts, relevance_score), ...]`` best first."""
        if not texts:
            return []
        try:
            resp = self.session.post(
                f"{self.base_url}/rerank",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json={"query": query, "documents": [t[: self.max_chars] for t in texts], "model": self.model,
                      "top_k": top_n or len(texts), "truncation": True},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise RerankError(f"Reranker unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise RerankError(f"Reranker returned HTTP {resp.status_code}: {resp.text[:120]}")
        try:
            data = resp.json()["data"]
            return [(int(d["index"]), float(d["relevance_score"])) for d in data if 0 <= int(d["index"]) < len(texts)]
        except (ValueError, KeyError, TypeError) as exc:
            raise RerankError("Unexpected reranker response.") from exc


def build_reranker() -> Optional[Any]:
    if config.RERANK_PROVIDER == "voyage":
        return VoyageReranker(config.VOYAGE_API_KEY, config.VOYAGE_RERANK_MODEL, timeout=config.RERANK_TIMEOUT)
    return None
