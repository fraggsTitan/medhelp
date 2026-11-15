"""Fair-use controls for a small free-tier deployment.

* :class:`RateLimiter`  - at most N questions per client per minute (a client = a session id, else an IP).
* :class:`RequestGate`  - at most M questions are *processed* at once; extra ones wait in a short queue and
  get a friendly 429 with ``Retry-After`` if they would wait longer than ``wait_seconds``.

Groq's free plan gives the whole app roughly 8K tokens/minute, so letting five people fire at once only
produces five rate-limit errors; queueing them turns that into a short wait instead.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Callable, Iterator

from app.errors import RateLimitedError


class RateLimiter:
    def __init__(self, per_minute: int = 6, window: float = 60.0, max_keys: int = 5000, clock: Callable[[], float] = time.monotonic):
        self.per_minute, self.window, self.max_keys, self._clock = per_minute, window, max_keys, clock
        self._hits: dict[str, deque] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Record one request for *key* or raise :class:`RateLimitedError`. ``per_minute <= 0`` disables the limit."""
        if self.per_minute <= 0:
            return
        now = self._clock()
        with self._lock:
            dq = self._hits.setdefault(key, deque())
            while dq and now - dq[0] > self.window:
                dq.popleft()
            if len(dq) >= self.per_minute:
                wait = max(1.0, self.window - (now - dq[0]))
                raise RateLimitedError(
                    f"You are asking questions faster than the free plan allows. Please wait about {int(wait) + 1} seconds.",
                    retry_after=wait,
                )
            dq.append(now)
            if len(self._hits) > self.max_keys:  # drop idle keys
                for k in [k for k, d in self._hits.items() if not d or now - d[-1] > self.window]:
                    del self._hits[k]


class RequestGate:
    def __init__(self, max_concurrent: int = 3, wait_seconds: float = 25.0):
        self.max_concurrent, self.wait_seconds = max(1, max_concurrent), wait_seconds
        self._sem = threading.BoundedSemaphore(self.max_concurrent)
        self._waiting = 0
        self._active = 0
        self._lock = threading.Lock()

    @contextmanager
    def slot(self) -> Iterator[None]:
        with self._lock:
            self._waiting += 1
        got = self._sem.acquire(timeout=self.wait_seconds)
        with self._lock:
            self._waiting -= 1
            if got:
                self._active += 1
        if not got:
            raise RateLimitedError(
                "The assistant is busy answering other questions. Please try again in a few seconds.",
                retry_after=min(10.0, self.wait_seconds),
            )
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
            self._sem.release()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"active": self._active, "waiting": self._waiting, "max_concurrent": self.max_concurrent}
