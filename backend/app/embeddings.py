"""Query-time embeddings.

The Qdrant index is built once, on your own machine, with ``BAAI/bge-large-en-v1.5`` (see ``ingestion.py``).
The deployed API only has to embed one short *question* per request, so it can do that through Cloudflare
Workers AI, which hosts the very same model: the vectors live in the same space and **no re-indexing is
needed**, and the API process does not need torch or a 1.3 GB model (that is what keeps RAM small).

Providers
---------
cloudflare  (default)  POST https://api.cloudflare.com/client/v4/accounts/<id>/ai/run/@cf/baai/bge-large-en-v1.5
local                  sentence-transformers in-process (needs ``pip install -r requirements-ingest.txt``)

BGE models are trained so that *queries* carry an instruction prefix and passages do not; ingestion embeds
passages without it, so every provider here prepends it to questions.

Run ``python -m app.embeddings --check`` once (on a machine that has the local model) to compare the
Cloudflare vectors with the local ones before you switch.
"""
from __future__ import annotations

import logging
import math
import threading
import time
from collections import OrderedDict
from typing import Any, Optional, Protocol

import requests

from app import config
from app.errors import EmbeddingError, RateLimitedError

logger = logging.getLogger("medhelp.embeddings")

# BGE retrieval models perform best when *queries* (not passages) carry this prefix.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def normalize(vector: list[float]) -> list[float]:
    """L2-normalise (Cosine distance in Qdrant is scale-free, but cache comparisons assume unit vectors)."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else list(vector)


def cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


class Embedder(Protocol):
    name: str

    def embed_query(self, text: str) -> list[float]: ...


# --------------------------------------------------------------------------- #
# Cloudflare Workers AI
# --------------------------------------------------------------------------- #
class CloudflareEmbedder:
    """Embeds questions through Cloudflare Workers AI's REST API (free allowance: 10,000 neurons/day)."""

    def __init__(
        self,
        account_id: str,
        api_token: str,
        model: str = "@cf/baai/bge-large-en-v1.5",
        instruction: str = BGE_QUERY_INSTRUCTION,
        base_url: str = "https://api.cloudflare.com/client/v4",
        timeout: float = 20,
        session: Optional[requests.Session] = None,
        retries: int = 1,
    ) -> None:
        self.account_id, self.api_token, self.model = account_id, api_token, model
        self.instruction, self.timeout, self.retries = instruction, timeout, retries
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()
        self.name = f"cloudflare:{model}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/accounts/{self.account_id}/ai/run/{self.model}"

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        last: Optional[Exception] = None
        for attempt in range(self.retries + 1):
            try:
                resp = self.session.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self.api_token}", "Content-Type": "application/json"},
                    json={"text": texts},
                    timeout=self.timeout,
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last = exc
                time.sleep(0.4 * (attempt + 1))
                continue
            if resp.status_code == 429:
                raise RateLimitedError(
                    "The embedding service's free allowance for today is used up or it is rate-limiting. "
                    "Please try again later.",
                    retry_after=_retry_after(resp),
                )
            if resp.status_code in (401, 403):
                raise EmbeddingError("Cloudflare rejected the credentials. Check CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN.")
            if resp.status_code >= 500:
                last = EmbeddingError(f"Cloudflare returned HTTP {resp.status_code}.")
                time.sleep(0.4 * (attempt + 1))
                continue
            if resp.status_code != 200:
                raise EmbeddingError(f"Embedding request failed (HTTP {resp.status_code}): {resp.text[:160]}")
            try:
                data = resp.json()["result"]["data"]
            except (ValueError, KeyError, TypeError) as exc:
                raise EmbeddingError("Unexpected response from the embedding service.") from exc
            if not isinstance(data, list) or len(data) != len(texts) or not all(isinstance(v, list) and v for v in data):
                raise EmbeddingError("The embedding service returned an unexpected number of vectors.")
            return [normalize([float(x) for x in v]) for v in data]
        raise EmbeddingError(f"The embedding service is unreachable: {last}")

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([self.instruction + text])[0]


def _retry_after(resp: Any) -> Optional[float]:
    try:
        return float(resp.headers.get("Retry-After"))
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Local model (fallback / ingestion machine)
# --------------------------------------------------------------------------- #
class LocalEmbedder:
    """sentence-transformers in-process. Heavy: only import it where you have the RAM."""

    def __init__(self, model_name: Optional[str] = None, instruction: str = BGE_QUERY_INSTRUCTION) -> None:
        self.model_name = model_name or config.EMBED_MODEL_NAME
        self.instruction = instruction
        self.name = f"local:{self.model_name}"
        self._model: Any = None
        self._lock = threading.Lock()

    def _load(self) -> Any:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:  # pragma: no cover - depends on the install
                    raise EmbeddingError(
                        "EMBED_PROVIDER=local needs sentence-transformers (pip install -r requirements-ingest.txt)."
                    ) from exc
                logger.info("Loading local embedding model %s ...", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                self._model.max_seq_length = 512
            return self._model

    def embed_query(self, text: str) -> list[float]:
        vec = self._load().encode(self.instruction + text, normalize_embeddings=True)
        return [float(x) for x in vec]


# --------------------------------------------------------------------------- #
# Wrapper with a tiny LRU cache and the factory
# --------------------------------------------------------------------------- #
class CachedEmbedder:
    """Remembers the last few question vectors (retries, repeated questions, cache + retrieval sharing)."""

    def __init__(self, inner: Any, size: int = 128) -> None:
        self.inner, self.size = inner, size
        self.name = getattr(inner, "name", "embedder")
        self._cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._lock = threading.Lock()

    def embed_query(self, text: str) -> list[float]:
        with self._lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return list(self._cache[text])
        vec = self.inner.embed_query(text)
        with self._lock:
            self._cache[text] = vec
            while len(self._cache) > self.size:
                self._cache.popitem(last=False)
        return list(vec)


def build_embedder() -> CachedEmbedder:
    provider = config.EMBED_PROVIDER
    if provider == "cloudflare":
        if not (config.CLOUDFLARE_ACCOUNT_ID and config.CLOUDFLARE_API_TOKEN):
            raise EmbeddingError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN must be set (or use EMBED_PROVIDER=local).")
        inner: Any = CloudflareEmbedder(
            config.CLOUDFLARE_ACCOUNT_ID,
            config.CLOUDFLARE_API_TOKEN,
            model=config.CLOUDFLARE_EMBED_MODEL,
            timeout=config.EMBED_TIMEOUT,
        )
    elif provider == "local":
        inner = LocalEmbedder()
    else:
        raise EmbeddingError(f"Unknown EMBED_PROVIDER '{provider}' (use 'cloudflare' or 'local').")
    return CachedEmbedder(inner)


# --------------------------------------------------------------------------- #
# Parity check: are the Cloudflare vectors the same as the ones that built the index?
# --------------------------------------------------------------------------- #
SAMPLE_QUERIES = [
    "What does WHO recommend as first-line drug treatment for hypertension in adults?",
    "HbA1c target for most adults with diabetes",
    "Inhaled corticosteroid for mild asthma",
    "When should patients with chronic kidney disease be referred to a nephrologist?",
    "Anticoagulation for atrial fibrillation stroke prevention",
]


def parity_report(reference: Any, candidate: Any, queries: Optional[list[str]] = None) -> dict[str, Any]:
    """Cosine similarity between two embedders on sample questions (1.0 = identical direction)."""
    scores = [round(cosine(reference.embed_query(q), candidate.embed_query(q)), 5) for q in (queries or SAMPLE_QUERIES)]
    low = min(scores)
    return {
        "scores": scores,
        "min": low,
        "verdict": "OK - safe to use the API embedder with your existing index" if low >= 0.99
        else "MISMATCH - the API vectors differ from the index model; re-ingest with the same provider or use EMBED_PROVIDER=local",
    }


def main(argv: Optional[list[str]] = None) -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Compare Cloudflare query vectors with the local bge model.")
    parser.add_argument("--check", action="store_true", required=True)
    parser.parse_args(argv)
    cf = CloudflareEmbedder(config.CLOUDFLARE_ACCOUNT_ID, config.CLOUDFLARE_API_TOKEN, model=config.CLOUDFLARE_EMBED_MODEL)
    report = parity_report(LocalEmbedder(), cf)
    print(json.dumps(report, indent=2))
    return 0 if report["min"] >= 0.99 else 1


if __name__ == "__main__":
    raise SystemExit(main())
