"""Shared test helpers: a throw-away local HTTP server that plays Groq / Cloudflare / Voyage."""
from __future__ import annotations

import http.server
import json
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator


class Recorder:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []


@contextmanager
def serve(routes: dict[tuple[str, str], Callable[[dict[str, Any]], tuple[int, dict[str, str], bytes]]]) -> Iterator[tuple[str, Recorder]]:
    """Serve ``routes[(METHOD, path)](request) -> (status, headers, body)``; yields ``(base_url, recorder)``.

    ``request`` is ``{"path", "headers", "body"(bytes), "json"(or None)}``. A route may be a list of callables,
    used one after another (handy for "fail once, then succeed").
    """
    rec = Recorder()
    state = {k: 0 for k in routes}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:  # silence
            pass

        def _handle(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            try:
                parsed = json.loads(body) if body and "json" in (self.headers.get("Content-Type") or "") else None
            except ValueError:
                parsed = None
            req = {"path": self.path, "headers": dict(self.headers), "body": body, "json": parsed}
            rec.requests.append(req)
            handler = routes.get((self.command, self.path.split("?")[0]))
            if handler is None:
                status, headers, payload = 404, {}, b'{"error": "no route"}'
            else:
                if isinstance(handler, list):
                    key = (self.command, self.path.split("?")[0])
                    idx = min(state[key], len(handler) - 1)
                    state[key] += 1
                    handler = handler[idx]
                status, headers, payload = handler(req)
            self.send_response(status)
            for k, v in headers.items():
                self.send_header(k, v)
            if "Content-Type" not in headers:
                self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        do_GET = do_POST = do_DELETE = _handle

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", rec
    finally:
        server.shutdown()
        server.server_close()


def ok(obj: Any, status: int = 200, headers: dict[str, str] | None = None) -> tuple[int, dict[str, str], bytes]:
    return status, headers or {}, json.dumps(obj).encode()
