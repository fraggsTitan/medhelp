"""Central configuration for the MedHelp backend.

Everything comes from environment variables (optionally loaded from ``backend/.env``). Nothing in this
module imports heavy libraries, so it is safe to import from anywhere (including unit tests).

Required secrets
----------------
GROQ_API_KEY                 Groq API key (answers, jargon, follow-ups, speech-to-text)
QDRANT_URL / QDRANT_API_KEY  Qdrant Cloud cluster
CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN   only when EMBED_PROVIDER=cloudflare (the default)

See ``.env.example`` for every setting.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent  # the backend/ folder
load_dotenv(PROJECT_ROOT / ".env")


def _env_str(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


# --------------------------------------------------------------------------- #
# Secrets / endpoints
# --------------------------------------------------------------------------- #
GROQ_API_KEY = _env_str("GROQ_API_KEY")
QDRANT_URL = _env_str("QDRANT_URL")
QDRANT_API_KEY = _env_str("QDRANT_API_KEY")
# Optional shared secret between the frontend and this API (sent as the X-API-Key header). Strongly
# recommended on a public deployment: it stops strangers from spending your Groq / Cloudflare quota.
API_KEY = _env_str("API_KEY")

# --------------------------------------------------------------------------- #
# LLM settings (Groq)
# --------------------------------------------------------------------------- #
GROQ_BASE_URL = _env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = _env_str("GROQ_MODEL", "openai/gpt-oss-120b")  # answers + comparisons
GROQ_FAST_MODEL = _env_str("GROQ_FAST_MODEL", "openai/gpt-oss-20b")  # jargon, follow-ups, query rewriting
GROQ_REASONING_EFFORT = _env_str("GROQ_REASONING_EFFORT", "low")  # gpt-oss only; blank = not sent
GROQ_MAX_TOKENS = _env_int("GROQ_MAX_TOKENS", 2048)
GROQ_HELPER_MAX_TOKENS = _env_int("GROQ_HELPER_MAX_TOKENS", 1024)
GROQ_TIMEOUT = _env_int("GROQ_TIMEOUT", 60)
ANSWER_STYLE = _env_str("ANSWER_STYLE", "helpful").lower()  # helpful | strict

# Speech-to-text (voice questions)
GROQ_STT_MODEL = _env_str("GROQ_STT_MODEL", "whisper-large-v3-turbo")
STT_LANGUAGE = _env_str("STT_LANGUAGE", "en")  # ISO-639-1; blank = auto-detect
MAX_AUDIO_BYTES = _env_int("MAX_AUDIO_BYTES", 2_000_000)

# --------------------------------------------------------------------------- #
# Query-time embeddings (the index itself is built on your own machine, see ingestion.py)
# --------------------------------------------------------------------------- #
# cloudflare = Cloudflare Workers AI hosting the SAME bge-large model that built your index (no re-index)
# local      = run sentence-transformers inside this process (needs ~2 GB RAM; install requirements-ingest.txt)
EMBED_PROVIDER = _env_str("EMBED_PROVIDER", "cloudflare").lower()
EMBED_MODEL_NAME = _env_str("EMBED_MODEL_NAME", "BAAI/bge-large-en-v1.5")  # local provider + ingestion
EMBED_BATCH_SIZE = _env_int("EMBED_BATCH_SIZE", 16)
CLOUDFLARE_ACCOUNT_ID = _env_str("CLOUDFLARE_ACCOUNT_ID")
CLOUDFLARE_API_TOKEN = _env_str("CLOUDFLARE_API_TOKEN")
CLOUDFLARE_EMBED_MODEL = _env_str("CLOUDFLARE_EMBED_MODEL", "@cf/baai/bge-large-en-v1.5")
EMBED_TIMEOUT = _env_int("EMBED_TIMEOUT", 20)

# NOTE: the default collection name keeps its original spelling on purpose. It is the name of the collection that
# already exists in your Qdrant cluster; renaming it here would make the app look for a collection that is not there.
QDRANT_COLLECTION = _env_str("QDRANT_COLLECTION", "medrag_guidelines")

# --------------------------------------------------------------------------- #
# Retrieval behaviour (defaults unchanged from the original project)
# --------------------------------------------------------------------------- #
DATA_DIR = Path(os.getenv("DATA_DIR") or (PROJECT_ROOT / "data"))
CHUNK_SIZE = _env_int("CHUNK_SIZE", 512)
CHUNK_OVERLAP = _env_int("CHUNK_OVERLAP", 50)
TOP_K = _env_int("TOP_K", 6)
RETRIEVE_CANDIDATES = _env_int("RETRIEVE_CANDIDATES", 16)
CONTEXT_CHAR_BUDGET = _env_int("CONTEXT_CHAR_BUDGET", 11000)
LEXICAL_SEARCH = _env_bool("LEXICAL_SEARCH", True)
LEXICAL_TOP_K = _env_int("LEXICAL_TOP_K", 8)
HISTORY_MESSAGES = _env_int("HISTORY_MESSAGES", 6)
MIN_SIMILARITY = _env_float("MIN_SIMILARITY", 0.25)

# Keyword index (SQLite FTS5). Built at ingestion time and shipped with the backend; if the file is missing
# or stale (point count differs from Qdrant) the backend rebuilds it from Qdrant in the background.
KEYWORD_DB_PATH = Path(os.getenv("KEYWORD_DB_PATH") or (DATA_DIR / "keyword.sqlite.gz"))

# --------------------------------------------------------------------------- #
# Reranker (optional, API based => no RAM cost)
# --------------------------------------------------------------------------- #
RERANK_PROVIDER = _env_str("RERANK_PROVIDER", "none").lower()  # none | voyage
VOYAGE_API_KEY = _env_str("VOYAGE_API_KEY")
VOYAGE_RERANK_MODEL = _env_str("VOYAGE_RERANK_MODEL", "rerank-2.5-lite")
RERANK_CANDIDATES = _env_int("RERANK_CANDIDATES", 12)  # how many fused passages are re-ordered
RERANK_TIMEOUT = _env_int("RERANK_TIMEOUT", 15)

# --------------------------------------------------------------------------- #
# New features
# --------------------------------------------------------------------------- #
FOLLOWUPS = _env_bool("FOLLOWUPS", True)  # suggested follow-up questions (fast model)
VERIFY_CITATIONS = _env_bool("VERIFY_CITATIONS", True)  # local check, no LLM call

SESSION_TTL_MINUTES = _env_int("SESSION_TTL_MINUTES", 120)
SESSION_MAX = _env_int("SESSION_MAX", 500)
SESSION_MAX_TURNS = _env_int("SESSION_MAX_TURNS", 20)
SESSION_TURN_CHARS = _env_int("SESSION_TURN_CHARS", 4000)

CACHE_ENABLED = _env_bool("CACHE_ENABLED", True)
CACHE_MAX_ENTRIES = _env_int("CACHE_MAX_ENTRIES", 300)
CACHE_TTL_MINUTES = _env_int("CACHE_TTL_MINUTES", 720)
CACHE_SIMILARITY = _env_float("CACHE_SIMILARITY", 0.97)

RATE_LIMIT_PER_MINUTE = _env_int("RATE_LIMIT_PER_MINUTE", 6)  # questions per client per minute
MAX_CONCURRENT_QUERIES = _env_int("MAX_CONCURRENT_QUERIES", 3)
QUEUE_WAIT_SECONDS = _env_int("QUEUE_WAIT_SECONDS", 25)

AUTO_INGEST = False  # removed: ingestion runs on your own machine (see README); kept so old .env files don't break
LOG_LEVEL = _env_str("LOG_LEVEL", "INFO").upper()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class ConfigError(RuntimeError):
    """Raised when mandatory configuration is missing."""


def is_local_qdrant() -> bool:
    url = QDRANT_URL.lower()
    return url.startswith(("http://localhost", "http://127.0.0.1", "localhost", "127.0.0.1"))


def missing_settings(need_llm: bool = True, need_qdrant: bool = True, need_embed_api: bool = True) -> list[str]:
    """Names of mandatory settings that are not set."""
    missing: list[str] = []
    if need_llm and not GROQ_API_KEY:
        missing.append("GROQ_API_KEY")
    if need_qdrant:
        if not QDRANT_URL:
            missing.append("QDRANT_URL")
        if not QDRANT_API_KEY and not is_local_qdrant():
            missing.append("QDRANT_API_KEY")
    if need_embed_api and EMBED_PROVIDER == "cloudflare":
        if not CLOUDFLARE_ACCOUNT_ID:
            missing.append("CLOUDFLARE_ACCOUNT_ID")
        if not CLOUDFLARE_API_TOKEN:
            missing.append("CLOUDFLARE_API_TOKEN")
    if RERANK_PROVIDER == "voyage" and not VOYAGE_API_KEY:
        missing.append("VOYAGE_API_KEY")
    return missing


def validate(need_llm: bool = True, need_qdrant: bool = True, need_embed_api: bool = False) -> None:
    """Raise :class:`ConfigError` with a helpful message if secrets are missing.

    ``need_embed_api`` defaults to False so ingestion (which embeds locally) and tests don't demand
    Cloudflare credentials; the API process checks them via :func:`missing_settings`.
    """
    missing = missing_settings(need_llm=need_llm, need_qdrant=need_qdrant, need_embed_api=need_embed_api)
    if missing:
        raise ConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill them in (or set them in your host's secret manager)."
        )
