"""Query-time access to Qdrant without LlamaIndex.

The index was written by LlamaIndex (ingestion), which stores each chunk as a point whose payload holds the
chunk text (``text`` / ``_node_content``) plus flattened metadata (``document_name``, ``publisher``,
``page_label`` ...). Here we only *read* it: embed the question, ask Qdrant for the nearest points, and wrap
them in tiny objects shaped like LlamaIndex's ``NodeWithScore`` so the engine's retrieval code is unchanged.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Iterator, Optional

from app import config
from app.errors import RetrievalError
from app.lexical import payload_to_record

logger = logging.getLogger("medhelp.qdrant")

# Bump when chunking / cleaning changes (ingestion stamps every chunk with it; the engine warns on old indexes).
INGEST_VERSION = "2"

_client: Any = None
_client_lock = threading.Lock()


def get_qdrant_client() -> Any:
    """Process-wide Qdrant client (created on first use)."""
    global _client
    with _client_lock:
        if _client is None:
            from qdrant_client import QdrantClient

            config.validate(need_llm=False)
            _client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY or None, timeout=60)
        return _client


def collection_point_count(client: Any, collection: Optional[str] = None) -> int:
    """Number of vectors stored in the collection (0 if it does not exist)."""
    name = collection or config.QDRANT_COLLECTION
    if not client.collection_exists(name):
        return 0
    return int(client.count(collection_name=name, exact=True).count)


def make_filter(publisher: Optional[str] = None) -> Any:
    """Qdrant payload filter restricting results to one publisher (None = no filter)."""
    if not publisher:
        return None
    from qdrant_client import models

    return models.Filter(must=[models.FieldCondition(key="publisher", match=models.MatchValue(value=publisher))])


@dataclass
class NodeLite:
    """Duck-typed stand-in for a LlamaIndex node (what ``engine._unpack`` needs)."""

    node_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_content(self, *args: Any, **kwargs: Any) -> str:
        return self.text


@dataclass
class ScoredNode:
    node: NodeLite
    score: Optional[float]


class QdrantRetriever:
    """Embed a question and return the nearest chunks."""

    def __init__(self, client: Any, collection: str, embedder: Any, limit: int) -> None:
        self.client, self.collection, self.embedder, self.limit = client, collection, embedder, limit

    def retrieve(self, question: str, publisher: Optional[str] = None) -> list[ScoredNode]:
        return self.retrieve_vector(self.embedder.embed_query(question), publisher)

    def retrieve_vector(self, vector: list[float], publisher: Optional[str] = None) -> list[ScoredNode]:
        """Nearest chunks to *vector*; ``publisher`` restricts the search to one publisher (compare mode)."""
        try:
            flt = make_filter(publisher)
            if hasattr(self.client, "query_points"):
                result = self.client.query_points(
                    collection_name=self.collection, query=vector, limit=self.limit, query_filter=flt, with_payload=True
                )
                points = list(result.points)
            else:  # very old qdrant-client
                points = list(
                    self.client.search(
                        collection_name=self.collection, query_vector=vector, limit=self.limit, query_filter=flt, with_payload=True
                    )
                )
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError(f"Retrieval from the vector database failed: {str(exc)[:200]}") from exc

        nodes: list[ScoredNode] = []
        for point in points:
            rec = payload_to_record(point.id, point.payload or {})
            if not rec:
                continue
            node_id, text, meta = rec
            nodes.append(ScoredNode(NodeLite(node_id, text, meta), float(point.score) if point.score is not None else None))
        return nodes


def iter_chunk_records(client: Any, collection: str, batch_size: int = 512) -> Iterator[tuple[str, str, dict[str, Any]]]:
    """Scroll through every point and yield ``(id, text, metadata)`` (no vectors are transferred)."""
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection, limit=batch_size, offset=offset, with_payload=True, with_vectors=False
        )
        for point in points:
            rec = payload_to_record(point.id, point.payload or {})
            if rec:
                yield rec
        if offset is None or not points:
            break
