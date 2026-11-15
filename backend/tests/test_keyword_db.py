import gzip
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.keyword_db import KeywordDB, build_from_qdrant, build_keyword_db, is_current, prepare_for_reading
from tests.test_legacy_core import REF_PAGE

RECORDS = [
    ("u1", "Adults with hypertension should start drug treatment at systolic pressure of 140 mmHg or higher.",
     {"document_name": "WHO HTN", "publisher": "WHO", "page_label": "9"}),
    ("u2", "Thiazide diuretics, ACE inhibitors and calcium channel blockers are first-line agents for hypertension.",
     {"document_name": "ACC HTN", "publisher": "ACC", "page_label": "4"}),
    ("u3", "Inhaled corticosteroids are the cornerstone of asthma controller therapy.",
     {"document_name": "GINA", "publisher": "GINA", "page_label": "2"}),
    ("u4", REF_PAGE, {"document_name": "WHO HTN", "publisher": "WHO", "page_label": "60"}),
]


class KeywordDBTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def build(self, name="kw.sqlite", points=4):
        out = self.tmp / name
        summary = build_keyword_db(RECORDS, out, qdrant_points=points)
        return out, summary

    def test_build_skips_bibliography_and_searches_with_bm25(self):
        out, summary = self.build()
        self.assertEqual(summary["chunks"], 3)
        self.assertEqual(summary["skipped_reference_chunks"], 1)
        db = KeywordDB(out)
        self.assertEqual(len(db), 3)
        hits = db.search("first-line drug treatment for hypertension", k=5)
        self.assertEqual({h.id for h in hits}, {"u1", "u2"})
        self.assertTrue(all(h.score > 0 for h in hits))
        self.assertEqual(hits[0].meta["publisher"] in ("WHO", "ACC"), True)
        self.assertEqual(db.search("asthma corticosteroids")[0].id, "u3")
        self.assertEqual(db.search("zzzz qqqq"), [])
        self.assertEqual(db.search("the of and"), [])  # only stop-words

    def test_publisher_filter(self):
        out, _ = self.build()
        db = KeywordDB(out)
        hits = db.search("hypertension treatment", k=5, publisher="ACC")
        self.assertEqual([h.id for h in hits], ["u2"])
        self.assertEqual(db.search("hypertension treatment", publisher="Nobody"), [])
        self.assertTrue(db.supports_filters)

    def test_min_matched_rule_for_long_queries(self):
        out, _ = self.build()
        db = KeywordDB(out)
        # "asthma" alone matches u3 but a 4-term query needs >= 2 matching terms
        self.assertEqual(db.search("asthma kidney transplant rejection"), [])

    def test_gzip_roundtrip_and_staleness(self):
        out, _ = self.build("kw.sqlite.gz", points=11)
        self.assertEqual(gzip.open(out).read(15)[:6], b"SQLite")
        db = KeywordDB.open(out)
        self.assertEqual(db.qdrant_points, 11)
        self.assertTrue(is_current(db, 11))
        self.assertFalse(is_current(db, 12))
        self.assertTrue(is_current(db, None))  # cannot tell -> trust
        self.assertEqual(db.document_names(), ["ACC HTN", "GINA", "WHO HTN"])
        self.assertEqual(prepare_for_reading(out), prepare_for_reading(out))  # decompressed once, reused

    def test_threads_get_their_own_connections(self):
        import threading

        out, _ = self.build()
        db = KeywordDB(out)
        results = []
        threads = [threading.Thread(target=lambda: results.append(len(db.search("hypertension treatment")))) for _ in range(6)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.assertEqual(results, [2] * 6)

    def test_build_from_qdrant_uses_scroll_and_count(self):
        pts = [SimpleNamespace(id=uid, payload={"text": text, **meta}) for uid, text, meta in RECORDS]

        class FakeClient:
            def collection_exists(self, name):
                return True

            def count(self, **kw):
                return SimpleNamespace(count=len(pts))

            def scroll(self, **kw):
                return (pts, None) if kw["offset"] is None else ([], None)

        out = self.tmp / "q.sqlite.gz"
        summary = build_from_qdrant(FakeClient(), "c", out)
        self.assertEqual((summary["chunks"], summary["qdrant_points"]), (3, 4))
        self.assertEqual(KeywordDB.open(out).qdrant_points, 4)

    def test_not_a_keyword_database_fails_fast(self):
        bad = self.tmp / "bad.sqlite"
        bad.write_bytes(b"not a database")
        with self.assertRaises(Exception):
            KeywordDB(bad)


if __name__ == "__main__":
    unittest.main()
