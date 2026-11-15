"""HTTP layer. Needs ``pip install fastapi httpx`` (skipped otherwise): the logic itself is covered by test_service.py."""
import importlib.util
import json
import py_compile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_service import FakeEngine, make

HAVE_FASTAPI = bool(importlib.util.find_spec("fastapi")) and bool(importlib.util.find_spec("httpx"))


class ApiSourceTests(unittest.TestCase):
    def test_api_module_compiles(self):
        py_compile.compile(str(Path(__file__).resolve().parent.parent / "app" / "api.py"), doraise=True)


@unittest.skipUnless(HAVE_FASTAPI, "fastapi/httpx not installed")
class ApiTests(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient

        from app.api import create_app

        self.svc = make(FakeEngine())
        self.client = TestClient(create_app(self.svc))
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)

    def test_health_and_root(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")
        self.assertEqual(self.client.get("/").status_code, 200)

    def test_session_roundtrip(self):
        sid = self.client.post("/sessions").json()["session_id"]
        body = {"question": "what is the target?", "session_id": sid}
        r = self.client.post("/query", json=body)
        self.assertEqual(r.status_code, 200, r.text)
        self.assertEqual(r.json()["session_id"], sid)
        self.assertEqual(len(self.client.get(f"/sessions/{sid}").json()["turns"]), 2)
        self.assertTrue(self.client.delete(f"/sessions/{sid}").json()["deleted"])
        self.assertEqual(self.client.get(f"/sessions/{sid}").status_code, 404)

    def test_api_key_is_enforced_when_configured(self):
        from app import config

        with mock.patch.object(config, "API_KEY", "secret"):
            self.assertEqual(self.client.post("/sessions").status_code, 401)
            self.assertEqual(self.client.post("/sessions", headers={"X-API-Key": "secret"}).status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)  # health stays open for uptime checks

    def test_stream_endpoint_emits_sse(self):
        with self.client.stream("POST", "/query/stream", json={"question": "stream me please"}) as r:
            text = "".join(r.iter_text())
        self.assertIn("event: token", text)
        self.assertIn("event: final", text)
        final = [json.loads(l[6:]) for l in text.splitlines() if l.startswith("data: ")][-1]
        self.assertEqual(final["kind"], "answer")

    def test_validation_errors_use_422(self):
        self.assertEqual(self.client.post("/query", json={"question": "x"}).status_code, 422)
        self.assertEqual(self.client.post("/query", json={"question": "fine question", "session_id": "bad id!"}).status_code, 422)

    def test_transcribe_upload(self):
        class T:
            def transcribe(self, audio, filename, content_type):
                return "hello world"

        self.svc._transcriber = T()
        r = self.client.post("/transcribe", files={"file": ("q.wav", b"RIFF....", "audio/wav")})
        self.assertEqual(r.json()["text"], "hello world")


if __name__ == "__main__":
    unittest.main()
