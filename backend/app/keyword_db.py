"""Keyword (BM25) search on SQLite FTS5 instead of a Python dictionary.

Why: the old in-memory index held every chunk's postings as Python tuples (roughly 100-200 MB for ~11,000
chunks) and had to be rebuilt from Qdrant on every cold start (tens of seconds of pure-Python tokenising,
minutes on a 0.1 vCPU free host). A SQLite file lives on disk, needs almost no RAM and opens instantly.

How it fits together
--------------------
* ``python -m app.ingestion`` (on your machine) ends by writing ``data/keyword.sqlite.gz`` from the chunks in
  Qdrant. Commit/ship that file with the backend.
* At start-up the API un-gzips it to the temp directory (disk, not RAM) and opens it read-only.
* The file records how many points Qdrant held when it was built. If Qdrant now has a different number (you
  re-ingested but did not ship a new file) the backend rebuilds a fresh copy from Qdrant in the background and
  swaps it in; until then answers use semantic search only.

Text is tokenised with the same tokenizer/stemmer as the old index (``lexical.tokenize``) and stored
pre-tokenised in a contentless FTS5 table; the readable text and metadata live in an ordinary table.
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from app import config, textutils
from app.lexical import LexicalHit, tokenize

logger = logging.getLogger("medhelp.keyword_db")

SCHEMA_VERSION = "1"

_SCHEMA = """
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE chunks(id INTEGER PRIMARY KEY, uid TEXT, text TEXT, doc TEXT, publisher TEXT, meta TEXT);
CREATE VIRTUAL TABLE fts USING fts5(tokens, content='', tokenize='unicode61');
"""


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #
def build_keyword_db(
    records: Iterable[tuple[str, str, dict[str, Any]]],
    out_path: Path | str,
    qdrant_points: Optional[int] = None,
    exclude_references: bool = True,
) -> dict[str, Any]:
    """Write a keyword database to *out_path* (``.gz`` => gzip-compressed). Returns a small summary."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix="medhelp-kwbuild-"))
    db_file = work_dir / "keyword.sqlite"
    try:
        conn = sqlite3.connect(str(db_file))
        conn.execute("PRAGMA journal_mode=OFF")
        conn.execute("PRAGMA synchronous=OFF")
        conn.executescript(_SCHEMA)
        count = skipped_refs = skipped_empty = 0
        rows: list[tuple] = []
        fts_rows: list[tuple] = []
        for uid, text, meta in records:
            if exclude_references and textutils.is_reference_text(text):
                skipped_refs += 1
                continue
            tokens = tokenize(text)
            if not tokens:
                skipped_empty += 1
                continue
            count += 1
            rows.append((count, str(uid), text, str(meta.get("document_name") or meta.get("file_name") or ""),
                         str(meta.get("publisher") or ""), json.dumps(meta, ensure_ascii=False)))
            fts_rows.append((count, " ".join(tokens)))
            if len(rows) >= 1000:
                _flush(conn, rows, fts_rows)
        _flush(conn, rows, fts_rows)
        info = {
            "schema": SCHEMA_VERSION,
            "chunks": str(count),
            "qdrant_points": str(qdrant_points if qdrant_points is not None else ""),
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        conn.executemany("INSERT INTO meta(key, value) VALUES (?, ?)", list(info.items()))
        conn.execute("INSERT INTO fts(fts) VALUES('optimize')")
        conn.commit()
        conn.close()

        tmp_out = out_path.with_name(out_path.name + ".part")
        if out_path.suffix == ".gz":
            with open(db_file, "rb") as src, gzip.open(tmp_out, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copyfile(db_file, tmp_out)
        os.replace(tmp_out, out_path)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
    logger.info("Keyword database written to %s (%d chunks, %d reference chunks skipped).", out_path, count, skipped_refs)
    return {"path": str(out_path), "chunks": count, "skipped_reference_chunks": skipped_refs,
            "skipped_empty": skipped_empty, "bytes": out_path.stat().st_size, "qdrant_points": qdrant_points}


def _flush(conn: sqlite3.Connection, rows: list[tuple], fts_rows: list[tuple]) -> None:
    if rows:
        conn.executemany("INSERT INTO chunks(id, uid, text, doc, publisher, meta) VALUES (?,?,?,?,?,?)", rows)
        conn.executemany("INSERT INTO fts(rowid, tokens) VALUES (?, ?)", fts_rows)
        rows.clear()
        fts_rows.clear()


def build_from_qdrant(client: Any, collection: str, out_path: Path | str) -> dict[str, Any]:
    """Scroll all chunks out of Qdrant and write the keyword database."""
    from app.qdrant_store import collection_point_count, iter_chunk_records

    points = collection_point_count(client, collection)
    return build_keyword_db(iter_chunk_records(client, collection), out_path, qdrant_points=points)


# --------------------------------------------------------------------------- #
# Opening / searching
# --------------------------------------------------------------------------- #
def prepare_for_reading(path: Path | str) -> Path:
    """Return a plain ``.sqlite`` path: ``.gz`` files are un-gzipped once into the temp dir (disk, not RAM)."""
    path = Path(path)
    if path.suffix != ".gz":
        return path
    stat = path.stat()
    target = Path(tempfile.gettempdir()) / f"medhelp-kw-{stat.st_size}-{stat.st_mtime_ns}.sqlite"
    if not target.exists():
        tmp = target.with_name(target.name + f".{os.getpid()}.part")
        with gzip.open(path, "rb") as src, open(tmp, "wb") as dst:
            shutil.copyfileobj(src, dst)
        os.replace(tmp, target)
    return target


class KeywordDB:
    """Read-only BM25 search over the keyword database (thread-safe: one connection per thread)."""

    supports_filters = True

    def __init__(self, sqlite_path: Path | str) -> None:
        self.path = Path(sqlite_path)
        self._local = threading.local()
        self._info: Optional[dict[str, str]] = None
        self.info()  # fail fast if the file is not a valid keyword database

    @classmethod
    def open(cls, path: Path | str) -> "KeywordDB":
        return cls(prepare_for_reading(path))

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, check_same_thread=False)
            self._local.conn = conn
        return conn

    def info(self) -> dict[str, str]:
        if self._info is None:
            self._info = dict(self._conn().execute("SELECT key, value FROM meta").fetchall())
        return self._info

    def __len__(self) -> int:
        return int(self.info().get("chunks", "0") or 0)

    @property
    def qdrant_points(self) -> Optional[int]:
        value = self.info().get("qdrant_points", "")
        return int(value) if value.isdigit() else None

    def document_names(self) -> list[str]:
        return [r[0] for r in self._conn().execute("SELECT DISTINCT doc FROM chunks WHERE doc <> '' ORDER BY doc")]

    def search(
        self,
        query: str,
        k: int = 8,
        min_matched: Optional[int] = None,
        publisher: Optional[str] = None,
    ) -> list[LexicalHit]:
        """Top-*k* chunks for *query* (optionally one publisher only).

        ``min_matched`` = minimum number of distinct query terms a chunk must contain (default: 2 for queries
        with 3+ terms, else 1), the same rule the old in-memory index used.
        """
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms:
            return []
        if min_matched is None:
            min_matched = 2 if len(terms) >= 3 else 1
        match = " OR ".join(f'"{t}"' for t in terms)
        sql = "SELECT c.uid, c.text, c.meta, bm25(fts) FROM fts JOIN chunks c ON c.id = fts.rowid WHERE fts MATCH ?"
        params: list[Any] = [match]
        if publisher:
            sql += " AND c.publisher = ?"
            params.append(publisher)
        sql += " ORDER BY bm25(fts) LIMIT ?"
        params.append(max(k * 5, 40))
        hits: list[LexicalHit] = []
        for uid, text, meta_json, score in self._conn().execute(sql, params):
            present = set(tokenize(text))
            matched = sum(1 for t in terms if t in present)
            if matched < min_matched:
                continue
            try:
                meta = json.loads(meta_json)
            except ValueError:
                meta = {}
            hits.append(LexicalHit(uid, text, meta, round(-float(score), 4), matched))
            if len(hits) >= k:
                break
        return hits


def is_current(db: KeywordDB, qdrant_points: Optional[int]) -> bool:
    """True when the database was built from a collection with the same number of points as Qdrant has now."""
    if qdrant_points is None or db.qdrant_points is None:
        return True  # cannot tell: trust the file
    return db.qdrant_points == qdrant_points


# --------------------------------------------------------------------------- #
# CLI:  python -m app.keyword_db build | info
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build or inspect the SQLite keyword index.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="read every chunk from Qdrant and write the keyword database")
    b.add_argument("--out", default=str(config.KEYWORD_DB_PATH))
    i = sub.add_parser("info", help="show metadata of an existing keyword database")
    i.add_argument("--path", default=str(config.KEYWORD_DB_PATH))
    args = parser.parse_args(argv)
    logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(message)s")

    if args.cmd == "build":
        from app.qdrant_store import get_qdrant_client

        summary = build_from_qdrant(get_qdrant_client(), config.QDRANT_COLLECTION, args.out)
        print(json.dumps(summary, indent=2))
        return 0
    db = KeywordDB.open(args.path)
    print(json.dumps(db.info(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
