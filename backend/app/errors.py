"""Error types shared by every backend module (the API layer maps ``status_code`` onto HTTP)."""
from __future__ import annotations

from typing import Optional


class MedHelpError(Exception):
    status_code = 500
    retry_after: Optional[float] = None


class IndexNotReadyError(MedHelpError):
    status_code = 503


class RetrievalError(MedHelpError):
    status_code = 503


class RateLimitedError(MedHelpError):
    status_code = 429

    def __init__(self, message: str, retry_after: Optional[float] = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelUnavailableError(MedHelpError):
    status_code = 503


class LLMAuthError(MedHelpError):
    status_code = 502


class LLMError(MedHelpError):
    status_code = 502


class EmbeddingError(MedHelpError):
    """The query-embedding provider (Cloudflare / local model) failed."""

    status_code = 502


class BadRequestError(MedHelpError):
    status_code = 422


class UnauthorizedError(MedHelpError):
    status_code = 401
