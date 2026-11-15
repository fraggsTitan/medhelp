"""Framework-independent application service: sessions + fair-use limits + the engine + voice.

``api.py`` is a thin FastAPI wrapper around this class, which keeps all behaviour unit-testable without
FastAPI installed.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Iterator, Optional

from app import config
from app.engine import MedHelpEngine, classify_llm_error
from app.errors import BadRequestError, IndexNotReadyError, MedHelpError, RateLimitedError
from app.limits import RateLimiter, RequestGate
from app.sessions import SessionStore, valid_session_id
from app.stt import ALLOWED_AUDIO_TYPES

logger = logging.getLogger("medhelp.service")

VERSION = "2.0.0"
MAX_QUESTION_CHARS = 1000
# Turns worth remembering: real answers and "I don't know"s. Guardrail refusals contain personal health details and
# small talk is noise, so neither is kept on the server or sent to the model as context later.
STORED_KINDS = {"answer", "no_answer"}


def read_rss_mb() -> Optional[float]:
    """Resident memory of this process in MB (Linux /proc; falls back to the peak from ``resource``)."""
    try:
        with open("/proc/self/status", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024, 1)
    except OSError:
        pass
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)
    except Exception:  # noqa: BLE001
        return None


class MedHelpService:
    def __init__(
        self,
        engine: Optional[MedHelpEngine] = None,
        sessions: Optional[SessionStore] = None,
        limiter: Optional[RateLimiter] = None,
        gate: Optional[RequestGate] = None,
        transcriber: Any = None,
        check_config: bool = True,
    ) -> None:
        self.engine = engine or MedHelpEngine()
        self.sessions = sessions or SessionStore(
            ttl_seconds=config.SESSION_TTL_MINUTES * 60,
            max_sessions=config.SESSION_MAX,
            max_turns=config.SESSION_MAX_TURNS,
            max_chars=config.SESSION_TURN_CHARS,
        )
        self.limiter = limiter or RateLimiter(per_minute=config.RATE_LIMIT_PER_MINUTE)
        self.gate = gate or RequestGate(config.MAX_CONCURRENT_QUERIES, config.QUEUE_WAIT_SECONDS)
        self._keepalive_limiter = RateLimiter(per_minute=6)
        self._transcriber = transcriber
        self._check_config = check_config
        self.started_at = time.time()

    # ----- sessions -----------------------------------------------------------
    def new_session(self) -> str:
        return self.sessions.create()

    def get_session(self, sid: str) -> Optional[list[dict[str, str]]]:
        if not valid_session_id(sid):
            raise BadRequestError("Invalid session id.")
        return self.sessions.history(sid)

    def delete_session(self, sid: str) -> bool:
        if not valid_session_id(sid):
            raise BadRequestError("Invalid session id.")
        return self.sessions.delete(sid)

    def _resolve_history(
        self, session_id: Optional[str], client_history: Optional[list[dict[str, str]]]
    ) -> tuple[Optional[str], list[dict[str, str]], bool]:
        """``(session_id, turns, recovered)``. Without a session id the call is stateless (client history is used)."""
        if not session_id:
            return None, list(client_history or []), False
        if not valid_session_id(session_id):
            raise BadRequestError("Invalid session id.")
        turns, recovered = self.sessions.get_or_seed(session_id, client_history)
        return session_id, turns, recovered

    # ----- questions ----------------------------------------------------------
    def _precheck(self, question: str, client_key: str) -> str:
        question = " ".join((question or "").split())
        if len(question) < 2:
            raise BadRequestError("Question is too short.")
        if len(question) > MAX_QUESTION_CHARS:
            raise BadRequestError(f"Question is too long (max {MAX_QUESTION_CHARS} characters).")
        if self._check_config:
            missing = config.missing_settings()
            if missing:
                raise IndexNotReadyError("The server is missing configuration: " + ", ".join(missing))
        self.limiter.check(client_key)
        return question

    def _finalize(self, result: dict[str, Any], sid: Optional[str], question: str, recovered: bool) -> dict[str, Any]:
        result["session_id"] = sid
        result["session_recovered"] = recovered
        if sid and result.get("kind") in STORED_KINDS:
            self.sessions.append(sid, "user", question)
            self.sessions.append(sid, "assistant", result.get("answer", ""))
        return result

    def answer(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        session_id: Optional[str] = None,
        compare: Optional[list[str]] = None,
        client_key: str = "anon",
    ) -> dict[str, Any]:
        question = self._precheck(question, session_id or client_key)
        sid, turns, recovered = self._resolve_history(session_id, history)
        with self.gate.slot():
            result = self.engine.query(question, turns, compare)
        return self._finalize(result, sid, question, recovered)

    def answer_stream(
        self,
        question: str,
        history: Optional[list[dict[str, str]]] = None,
        session_id: Optional[str] = None,
        compare: Optional[list[str]] = None,
        client_key: str = "anon",
    ) -> Iterator[tuple[str, Any]]:
        """Validate eagerly (so HTTP errors keep their status codes), then return the event generator."""
        question = self._precheck(question, session_id or client_key)
        sid, turns, recovered = self._resolve_history(session_id, history)
        return self._stream_events(question, turns, compare, sid, recovered)

    def _stream_events(
        self, question: str, turns: list[dict[str, str]], compare: Optional[list[str]], sid: Optional[str], recovered: bool
    ) -> Iterator[tuple[str, Any]]:
        with self.gate.slot():
            for event, data in self.engine.query_stream(question, turns, compare):
                if event == "final":
                    data = self._finalize(data, sid, question, recovered)
                yield event, data

    # ----- voice ---------------------------------------------------------------
    def transcribe(self, audio: bytes, filename: str = "question.wav", content_type: str = "audio/wav", client_key: str = "anon") -> dict[str, Any]:
        if not audio:
            raise BadRequestError("The recording is empty.")
        if len(audio) > config.MAX_AUDIO_BYTES:
            raise BadRequestError(f"The recording is too large (max {config.MAX_AUDIO_BYTES // 1000} KB). Please record a shorter question.")
        ctype = (content_type or "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_AUDIO_TYPES and not ctype.startswith("audio/"):
            raise BadRequestError(f"Unsupported audio type '{ctype}'.")
        if self._check_config and "GROQ_API_KEY" in config.missing_settings(need_qdrant=False, need_embed_api=False):
            raise IndexNotReadyError("The server is missing configuration: GROQ_API_KEY")
        self.limiter.check("stt:" + client_key)
        if self._transcriber is None:
            from app.stt import build_transcriber

            self._transcriber = build_transcriber()
        started = time.perf_counter()
        try:
            text = self._transcriber.transcribe(audio, filename, content_type)
        except MedHelpError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise classify_llm_error(exc) from exc
        return {"text": text, "elapsed_ms": int((time.perf_counter() - started) * 1000)}

    # ----- info ------------------------------------------------------------------
    def documents(self) -> dict[str, Any]:
        return {"documents": self.engine.documents(), "publishers": self.engine.publishers()}

    def health(self) -> dict[str, Any]:
        missing = config.missing_settings() if self._check_config else []
        index_ready = False if missing else self.engine.is_ready()
        status = "misconfigured" if missing else ("ok" if index_ready else "degraded")
        cache = getattr(self.engine, "_cache", None)
        return {
            "status": status,
            "version": VERSION,
            "index_ready": index_ready,
            "missing_settings": missing,
            "keyword_index": self.engine.lexical_status,
            "embed_provider": config.EMBED_PROVIDER,
            "rerank_provider": config.RERANK_PROVIDER,
            "model": config.GROQ_MODEL,
            "memory_mb": read_rss_mb(),
            "uptime_s": int(time.time() - self.started_at),
            "sessions": self.sessions.stats(),
            "queue": self.gate.stats(),
            "cache": cache.stats() if cache is not None else None,
        }

    def keepalive(self, client_key: str = "keepalive") -> dict[str, Any]:
        """Touch Qdrant (free clusters are suspended after a week without activity). Call it from a cron job."""
        self._keepalive_limiter.check(client_key)
        from app.qdrant_store import collection_point_count, get_qdrant_client

        try:
            return {"ok": True, "qdrant_points": collection_point_count(get_qdrant_client())}
        except Exception as exc:  # noqa: BLE001
            raise IndexNotReadyError(f"Qdrant is unreachable: {str(exc)[:160]}") from exc
