"""Tests for the downloader (against a throw-away local HTTP server) and ingestion helpers."""
import http.server
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path

from pypdf import PdfWriter

from app import fetch_data, ingestion


def make_pdf_bytes(pages: int = 2) -> bytes:
    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


PDF = make_pdf_bytes()


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):  # noqa: N802
        if self.path == "/good.pdf":
            self._send(200, PDF, "application/pdf")
        elif self.path == "/html-as-pdf.pdf":  # 200 OK but actually an HTML block page
            self._send(200, b"<html>Access denied</html>" * 50, "text/html")
        else:
            self._send(404, b"nope", "text/plain")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # keep test output quiet
        pass


class FetchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.base = f"http://127.0.0.1:{cls.server.server_port}"
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()
        cls._orig_min = fetch_data.MIN_PDF_BYTES
        fetch_data.MIN_PDF_BYTES = 100  # our generated PDF is tiny

    @classmethod
    def tearDownClass(cls):
        fetch_data.MIN_PDF_BYTES = cls._orig_min
        cls.server.shutdown()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.session = fetch_data.build_session()
        self.addCleanup(self.session.close)

    def test_fallback_url_used_when_primary_returns_html(self):
        src = {"title": "T", "filename": "t.pdf",
               "urls": [f"{self.base}/html-as-pdf.pdf", f"{self.base}/good.pdf"]}
        r = fetch_data.download_source(self.session, src, self.dir)
        self.assertEqual(r["status"], "downloaded")
        self.assertTrue(r["url"].endswith("/good.pdf"))
        self.assertEqual(r["pages"], 2)
        self.assertEqual(len(r["errors"]), 1)  # the HTML block page was rejected
        self.assertTrue((self.dir / "t.pdf").read_bytes().startswith(b"%PDF-"))
        self.assertEqual(list(self.dir.glob("*.part")), [])

    def test_broken_link_is_reported_not_raised(self):
        src = {"title": "T", "filename": "x.pdf", "urls": [f"{self.base}/missing.pdf"]}
        r = fetch_data.download_source(self.session, src, self.dir)
        self.assertEqual(r["status"], "failed")
        self.assertIn("404", r["errors"][0])
        self.assertFalse((self.dir / "x.pdf").exists())
        self.assertEqual(list(self.dir.glob("*.part")), [])

    def test_unreachable_host_is_reported_not_raised(self):
        src = {"title": "T", "filename": "y.pdf", "urls": ["http://127.0.0.1:9/never.pdf"]}
        self.assertEqual(fetch_data.download_source(self.session, src, self.dir)["status"], "failed")

    def test_cached_file_is_not_redownloaded_and_corrupt_cache_is_replaced(self):
        src = {"title": "T", "filename": "c.pdf", "urls": [f"{self.base}/good.pdf"]}
        self.assertEqual(fetch_data.download_source(self.session, src, self.dir)["status"], "downloaded")
        self.assertEqual(fetch_data.download_source(self.session, src, self.dir)["status"], "cached")
        (self.dir / "c.pdf").write_bytes(b"corrupted" * 50)
        self.assertEqual(fetch_data.download_source(self.session, src, self.dir)["status"], "downloaded")

    def test_fetch_all_writes_manifest_and_supports_custom_sources(self):
        (self.dir / fetch_data.CUSTOM_SOURCES_NAME).write_text(json.dumps([
            {"title": "My Custom Guideline", "publisher": "Me", "url": f"{self.base}/good.pdf"},
            {"title": "malformed entry with no url"},
        ]))
        original = fetch_data.SOURCES
        fetch_data.SOURCES = []  # avoid touching the real internet
        try:
            results = fetch_data.fetch_all(self.dir)
        finally:
            fetch_data.SOURCES = original
        self.assertEqual([r["status"] for r in results], ["downloaded"])
        manifest = fetch_data.load_manifest(self.dir)
        self.assertEqual(manifest["my_custom_guideline.pdf"]["title"], "My Custom Guideline")
        self.assertEqual(manifest["my_custom_guideline.pdf"]["pages"], 2)

    def test_predefined_sources_are_well_formed(self):
        self.assertGreaterEqual(len(fetch_data.SOURCES), 3)
        names = set()
        for s in fetch_data.SOURCES:
            self.assertTrue(s["urls"] and all(u.startswith(("https://", "page:https://", "pmc:PMC", "doi:10.")) for u in s["urls"]))
            self.assertTrue(s["filename"].endswith(".pdf"))
            names.add(s["filename"])
        self.assertEqual(len(names), len(fetch_data.SOURCES))  # unique filenames


class IngestionHelperTests(unittest.TestCase):
    def test_node_ids_are_deterministic_and_distinct(self):
        a = ingestion.make_node_id("a.pdf", 3, 0)
        self.assertEqual(a, ingestion.make_node_id("a.pdf", 3, 0))
        self.assertNotEqual(a, ingestion.make_node_id("a.pdf", 3, 1))
        self.assertNotEqual(a, ingestion.make_node_id("a.pdf", 4, 0))
        self.assertNotEqual(a, ingestion.make_node_id("b.pdf", 3, 0))
        self.assertRegex(a, r"^[0-9a-f]{8}-[0-9a-f]{4}-5[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}$")  # valid UUID v5

    def test_pretty_title_and_clean_text(self):
        self.assertEqual(ingestion.pretty_title("who_hypertension-2021.pdf"), "Who Hypertension 2021")
        self.assertEqual(ingestion.clean_text("a\x00b"), "ab")


if __name__ == "__main__":
    unittest.main()
