"""Server-side chat sessions (feature: per-session message history).

Each browser tab gets its own session id; the server keeps only the *text* of recent turns, so one user's
context can never leak into another's. Memory is bounded three ways: a time-to-live, a maximum number of
sessions (least-recently-used are evicted) and a maximum number/length of turns per session.

Sessions live in this process's memory. On a free host that sleeps or restarts they are lost; the frontend
therefore re-sends its own copy of the conversation when the server does not know a session id
(``SessionStore.get_or_seed``), so users never notice.
"""
from __future__ import annotations

import re
import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Optional

SESSION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def new_session_id() -> str:
    return secrets.token_urlsafe(18)


def valid_session_id(value: Any) -> bool:
    return isinstance(value, str) and bool(SESSION_ID_RE.match(value))


class SessionStore:
    def __init__(
        self,
        ttl_seconds: float = 7200,
        max_sessions: int = 500,
        max_turns: int = 20,
        max_chars: int = 4000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl, self.max_sessions, self.max_turns, self.max_chars = ttl_seconds, max_sessions, max_turns, max_chars
        self._clock = clock
        self._data: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.evicted = 0

    # ----- internals (call with the lock held) ---------------------------------
    def _sweep(self) -> None:
        now = self._clock()
        expired = [sid for sid, s in self._data.items() if now - s["touched"] > self.ttl]
        for sid in expired:
            del self._data[sid]
            self.evicted += 1
        while len(self._data) > self.max_sessions:
            self._data.popitem(last=False)
            self.evicted += 1

    def _clean_turns(self, turns: list[dict[str, str]]) -> list[dict[str, str]]:
        out = [
            {"role": t["role"], "content": str(t["content"])[: self.max_chars]}
            for t in turns
            if isinstance(t, dict) and t.get("role") in ("user", "assistant") and str(t.get("content", "")).strip()
        ]
        return out[-self.max_turns :]

    # ----- public API ----------------------------------------------------------
    def create(self) -> str:
        sid = new_session_id()
        with self._lock:
            self._data[sid] = {"turns": [], "touched": self._clock(), "created": time.time()}
            self._sweep()
        return sid

    def history(self, sid: str) -> Optional[list[dict[str, str]]]:
        """Turns of a live session, or None if the id is unknown/expired."""
        with self._lock:
            self._sweep()
            s = self._data.get(sid)
            if s is None:
                return None
            s["touched"] = self._clock()
            self._data.move_to_end(sid)
            return [dict(t) for t in s["turns"]]

    def get_or_seed(self, sid: str, client_turns: Optional[list[dict[str, str]]] = None) -> tuple[list[dict[str, str]], bool]:
        """History for *sid*; if the server does not know it, start it from what the client sent.

        Returns ``(turns, recovered)``. When the session exists the client's copy is ignored (the server is
        the source of truth); this also stops a client from injecting fake assistant turns into a live session.
        """
        with self._lock:
            self._sweep()
            s = self._data.get(sid)
            if s is not None:
                s["touched"] = self._clock()
                self._data.move_to_end(sid)
                return [dict(t) for t in s["turns"]], False
            turns = self._clean_turns(client_turns or [])
            self._data[sid] = {"turns": turns, "touched": self._clock(), "created": time.time()}
            self._sweep()
            return [dict(t) for t in turns], bool(turns)

    def append(self, sid: str, role: str, content: str) -> None:
        with self._lock:
            s = self._data.get(sid)
            if s is None:
                s = self._data[sid] = {"turns": [], "touched": self._clock(), "created": time.time()}
            s["turns"] = self._clean_turns(s["turns"] + [{"role": role, "content": content}])
            s["touched"] = self._clock()
            self._data.move_to_end(sid)
            self._sweep()

    def delete(self, sid: str) -> bool:
        with self._lock:
            return self._data.pop(sid, None) is not None

    def stats(self) -> dict[str, int]:
        with self._lock:
            self._sweep()
            return {"sessions": len(self._data), "evicted": self.evicted, "max_sessions": self.max_sessions}
