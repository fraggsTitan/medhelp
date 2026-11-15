"""HTTP client for the MedHelp backend (plain ``requests``; no Streamlit imports)."""
from __future__ import annotations

from typing import Any, Iterable, Iterator, Optional

import requests

from medhelp_ui.helpers import friendly_error


class ApiError(Exception):
    """A readable error from (or about) the backend."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


def parse_sse(lines: Iterable[str]) -> Iterator[tuple[str, Any]]:
    """Parse a Server-Sent-Events line stream into ``(event, data)`` pairs (data is JSON-decoded when possible)."""
    import json

    event, data_lines = "message", []
    for line in lines:
        if line is None:
            continue
        if line == "":  # blank line: dispatch
            if data_lines:
                raw = "\n".join(data_lines)
                try:
                    yield event, json.loads(raw)
                except ValueError:
                    yield event, raw
            event, data_lines = "message", []
        elif line.startswith(":"):
            continue  # comment / keep-alive
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    if data_lines:  # stream ended without a trailing blank line
        raw = "\n".join(data_lines)
        try:
            yield event, json.loads(raw)
        except ValueError:
            yield event, raw


class MedHelpClient:
    def __init__(self, base_url: str, api_key: str = "", timeout: float = 120, session: Optional[requests.Session] = None) -> None:
        self.base_url, self.api_key, self.timeout = base_url.rstrip("/"), api_key, timeout
        self.http = session or requests.Session()

    # ----- plumbing -------------------------------------------------------------
    def _headers(self) -> dict[str, str]:
        return {"X-API-Key": self.api_key} if self.api_key else {}

    def _error_from(self, resp: requests.Response) -> ApiError:
        try:
            detail: Any = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "; ".join(str(d.get("msg", d)) for d in detail)
        retry = resp.headers.get("Retry-After")
        retry_after = float(retry) if retry and retry.replace(".", "", 1).isdigit() else None
        message = friendly_error(resp.status_code, str(detail))
        if resp.status_code == 429 and retry_after and "retry" not in message.lower():
            message += f" (retry in about {int(retry_after)}s)"
        return ApiError(message, resp.status_code, retry_after)

    def _request(self, method: str, path: str, **kw: Any) -> requests.Response:
        try:
            return self.http.request(method, f"{self.base_url}{path}", headers=self._headers(), timeout=kw.pop("timeout", self.timeout), **kw)
        except requests.exceptions.ConnectionError as exc:
            raise ApiError(f"Cannot reach the MedHelp API at {self.base_url}. Is the backend running (free hosts take about a minute to wake up)?") from exc
        except requests.exceptions.Timeout as exc:
            raise ApiError("The request timed out. The backend may still be waking up - please try again.") from exc

    # ----- info -----------------------------------------------------------------
    def health(self) -> dict[str, Any]:
        try:
            resp = self._request("GET", "/health", timeout=10)
            resp.raise_for_status()
            return resp.json()
        except (ApiError, requests.RequestException, ValueError) as exc:
            return {"status": "unreachable", "error": str(exc)}

    def documents(self) -> dict[str, Any]:
        try:
            resp = self._request("GET", "/documents", timeout=15)
            if resp.status_code != 200:
                return {"documents": [], "publishers": []}
            data = resp.json()
            return {"documents": data.get("documents", []), "publishers": data.get("publishers", [])}
        except (ApiError, ValueError):
            return {"documents": [], "publishers": []}

    # ----- sessions -------------------------------------------------------------
    def get_session(self, session_id: str) -> Optional[list[dict[str, str]]]:
        try:
            resp = self._request("GET", f"/sessions/{session_id}", timeout=10)
            return resp.json().get("turns") if resp.status_code == 200 else None
        except (ApiError, ValueError):
            return None

    def delete_session(self, session_id: str) -> None:
        try:
            self._request("DELETE", f"/sessions/{session_id}", timeout=10)
        except ApiError:
            pass  # best effort: sessions expire by themselves

    # ----- questions ------------------------------------------------------------
    @staticmethod
    def _body(question: str, history: list[dict[str, str]], session_id: Optional[str], compare: Optional[list[str]]) -> dict[str, Any]:
        body: dict[str, Any] = {"question": question, "history": history}
        if session_id:
            body["session_id"] = session_id
        if compare:
            body["compare"] = compare
        return body

    def query(self, question: str, history: list[dict[str, str]], session_id: Optional[str] = None, compare: Optional[list[str]] = None) -> dict[str, Any]:
        resp = self._request("POST", "/query", json=self._body(question, history, session_id, compare))
        if resp.status_code != 200:
            raise self._error_from(resp)
        return resp.json()

    def stream_query(
        self, question: str, history: list[dict[str, str]], session_id: Optional[str] = None, compare: Optional[list[str]] = None
    ) -> Iterator[tuple[str, Any]]:
        """Yield ``(event, data)`` pairs: ``status``, ``token`` ..., ``final``. Raises :class:`ApiError` on failure."""
        resp = self._request("POST", "/query/stream", json=self._body(question, history, session_id, compare), stream=True)
        if resp.status_code != 200:
            err = self._error_from(resp)
            resp.close()
            raise err
        # Decode explicitly: requests would assume ISO-8859-1 for a charset-less text/event-stream and garble symbols.
        lines = (b.decode("utf-8", errors="replace") if isinstance(b, bytes) else b for b in resp.iter_lines())
        try:
            for event, data in parse_sse(lines):
                if event == "error":
                    detail = data.get("detail", "Unexpected error.") if isinstance(data, dict) else str(data)
                    status = data.get("status") if isinstance(data, dict) else None
                    retry = data.get("retry_after") if isinstance(data, dict) else None
                    msg = friendly_error(status, detail)
                    if status == 429 and retry:
                        msg += f" (retry in about {int(retry) + 1}s)"
                    raise ApiError(msg, status, retry)
                yield event, data
        except requests.exceptions.ChunkedEncodingError as exc:
            raise ApiError("The connection to the backend dropped while the answer was streaming. Please try again.") from exc
        finally:
            resp.close()

    # ----- voice ------------------------------------------------------------------
    def transcribe(self, audio: bytes, filename: str = "question.wav", content_type: str = "audio/wav", session_id: Optional[str] = None) -> str:
        resp = self._request(
            "POST", "/transcribe",
            files={"file": (filename, audio, content_type)},
            data={"session_id": session_id} if session_id else None,
            timeout=60,
        )
        if resp.status_code != 200:
            raise self._error_from(resp)
        return str(resp.json().get("text", ""))
