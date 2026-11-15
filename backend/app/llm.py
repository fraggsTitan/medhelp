"""Thin Groq client built on ``requests`` (Groq exposes an OpenAI-compatible HTTP API).

Why not LlamaIndex here? Importing ``llama-index`` costs a few hundred MB of RAM and the deployed API only
needs "send a prompt, get text back (optionally token by token)". LlamaIndex is still used, on your own
machine, for ingestion.

Errors are raised as plain ``RuntimeError`` whose text contains the HTTP status and Groq's message, so
``engine.classify_llm_error`` can turn them into friendly 429 / 413 / 401 / "model decommissioned" errors.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Iterator, Optional

import requests

from app import config

logger = logging.getLogger("medhelp.llm")


@dataclass
class Completion:
    text: str
    finish_reason: Optional[str] = None


class GroqLLM:
    """Deterministic (temperature 0) chat completion client."""

    def __init__(
        self,
        model: str,
        api_key: str,
        max_tokens: int = 2048,
        reasoning_effort: str = "",
        base_url: str = "https://api.groq.com/openai/v1",
        timeout: float = 60,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.model, self.api_key, self.max_tokens = model, api_key, max_tokens
        self.reasoning_effort = reasoning_effort if "gpt-oss" in model else ""
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.session = session or requests.Session()

    # ----- request plumbing ----------------------------------------------------
    def _payload(self, prompt: str, stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": stream,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        return body

    def _post(self, prompt: str, stream: bool) -> requests.Response:
        try:
            resp = self.session.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=self._payload(prompt, stream),
                timeout=self.timeout,
                stream=stream,
            )
        except requests.Timeout as exc:
            raise RuntimeError(f"Request to Groq timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Groq: {exc}") from exc
        if resp.status_code != 200:
            body = resp.text[:500]
            resp.close()
            raise RuntimeError(f"Error code: {resp.status_code} - {body}")
        return resp

    # ----- public API ----------------------------------------------------------
    def complete(self, prompt: str) -> Completion:
        resp = self._post(prompt, stream=False)
        try:
            choice = resp.json()["choices"][0]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response from Groq: {resp.text[:200]}") from exc
        message = choice.get("message") or {}
        return Completion(text=(message.get("content") or "").strip(), finish_reason=choice.get("finish_reason"))

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the answer text piece by piece (reasoning tokens are not included)."""
        resp = self._post(prompt, stream=True)
        try:
            # Decode the bytes ourselves: Groq's stream is "text/event-stream" with no charset, and `requests`
            # would then assume ISO-8859-1 and turn every non-ASCII character (en dashes, non-breaking hyphens,
            # curly apostrophes, >=) into mojibake such as "\u00e2\u0080\u0091".
            for raw_bytes in resp.iter_lines():
                raw = raw_bytes.decode("utf-8", errors="replace") if isinstance(raw_bytes, bytes) else raw_bytes
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    event = json.loads(data)
                except ValueError:
                    continue
                if isinstance(event, dict) and event.get("error"):
                    raise RuntimeError(f"Error code: 500 - {json.dumps(event['error'])[:300]}")
                for choice in event.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
        finally:
            resp.close()


def build_groq(model: str, max_tokens: Optional[int] = None) -> GroqLLM:
    """Create a Groq client from the environment (raises ConfigError when the key is missing)."""
    config.validate(need_qdrant=False)
    return GroqLLM(
        model=model,
        api_key=config.GROQ_API_KEY,
        max_tokens=max_tokens or config.GROQ_MAX_TOKENS,
        reasoning_effort=config.GROQ_REASONING_EFFORT,
        base_url=config.GROQ_BASE_URL,
        timeout=config.GROQ_TIMEOUT,
    )
