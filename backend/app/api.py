"""FastAPI wrapper around :class:`app.service.MedHelpService`.

Run (from the backend/ folder)::

    uvicorn app.api:app --host 0.0.0.0 --port 8000 --workers 1

Keep ``--workers 1``: sessions, the answer cache and the keyword index live in this process's memory, so a second
worker would duplicate the RAM and split sessions between processes.
"""
from __future__ import annotations

import hmac
import json
import logging
import threading
from contextlib import asynccontextmanager
from typing import Any, Literal, Optional

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from app import config
from app.errors import MedHelpError
from app.service import VERSION, MedHelpService

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("medhelp.api")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=6000)


class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    history: list[Turn] = Field(default_factory=list, max_length=30)
    session_id: Optional[str] = Field(default=None, min_length=16, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    compare: Optional[list[str]] = Field(default=None, max_length=3, description="2-3 publisher names to compare side by side")


class SourceItem(BaseModel):
    source_text: str
    document_name: str
    publisher: Optional[str] = None
    page_number: str
    pdf_page: Optional[int] = None
    relevance_score: Optional[float] = None
    match_type: str = "semantic"
    source_url: Optional[str] = None
    evidence_grades: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    kind: str
    mode: str = "single"
    jargon_dictionary: dict[str, str] = Field(default_factory=dict)
    suggested_followups: list[str] = Field(default_factory=list)
    verification: list[dict[str, Any]] = Field(default_factory=list)
    sources: list[SourceItem] = Field(default_factory=list)
    guardrail_triggered: bool = False
    answered: bool = False
    cached: bool = False
    session_id: Optional[str] = None
    session_recovered: bool = False
    warnings: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
    disclaimer: str = ""
    elapsed_ms: int = 0


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """If API_KEY is configured every protected endpoint needs a matching X-API-Key header."""
    expected = config.API_KEY
    if expected and not hmac.compare_digest(x_api_key or "", expected):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")


def client_key(request: Request, session_id: Optional[str] = None) -> str:
    if session_id:
        return session_id
    forwarded = request.headers.get("x-forwarded-for", "").split(",")[0].strip()
    return forwarded or (request.client.host if request.client else "anon")


def sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _error_response(exc: MedHelpError) -> JSONResponse:
    headers = {"Retry-After": str(int(exc.retry_after) + 1)} if exc.retry_after else None
    return JSONResponse(status_code=exc.status_code, content={"detail": str(exc)}, headers=headers)


# --------------------------------------------------------------------------- #
# App factory
# --------------------------------------------------------------------------- #
def create_app(service: Optional[MedHelpService] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or MedHelpService()
        if service is None and not config.missing_settings():
            # Warm up in the background so the port opens immediately (free hosts kill slow starters).
            threading.Thread(target=_warm_up, args=(app.state.service,), name="warm-up", daemon=True).start()
        yield

    app = FastAPI(title="MedHelp API", version=VERSION, lifespan=lifespan)

    @app.exception_handler(MedHelpError)
    async def medhelp_error_handler(_: Request, exc: MedHelpError) -> JSONResponse:
        return _error_response(exc)

    @app.exception_handler(ValueError)
    async def value_error_handler(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    def svc(request: Request) -> MedHelpService:
        return request.app.state.service

    @app.get("/")
    def root() -> dict[str, str]:
        return {"name": "MedHelp API", "version": VERSION, "docs": "/docs"}

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        return svc(request).health()

    @app.get("/keepalive", dependencies=[Depends(require_key)])
    def keepalive(request: Request) -> dict[str, Any]:
        return svc(request).keepalive()

    @app.get("/documents", dependencies=[Depends(require_key)])
    def documents(request: Request) -> dict[str, Any]:
        return svc(request).documents()

    # ----- sessions -----
    @app.post("/sessions", dependencies=[Depends(require_key)])
    def create_session(request: Request) -> dict[str, str]:
        return {"session_id": svc(request).new_session()}

    @app.get("/sessions/{session_id}", dependencies=[Depends(require_key)])
    def get_session(session_id: str, request: Request) -> dict[str, Any]:
        turns = svc(request).get_session(session_id)
        if turns is None:
            raise HTTPException(status_code=404, detail="Unknown or expired session.")
        return {"session_id": session_id, "turns": turns}

    @app.delete("/sessions/{session_id}", dependencies=[Depends(require_key)])
    def delete_session(session_id: str, request: Request) -> dict[str, bool]:
        return {"deleted": svc(request).delete_session(session_id)}

    # ----- questions -----
    @app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_key)])
    def query(body: QueryRequest, request: Request) -> dict[str, Any]:
        return svc(request).answer(
            body.question,
            [t.model_dump() for t in body.history],
            body.session_id,
            body.compare,
            client_key(request, body.session_id),
        )

    @app.post("/query/stream", dependencies=[Depends(require_key)])
    def query_stream(body: QueryRequest, request: Request) -> StreamingResponse:
        # Validation / rate-limit errors are raised here, before streaming starts, so they keep their HTTP status.
        events = svc(request).answer_stream(
            body.question,
            [t.model_dump() for t in body.history],
            body.session_id,
            body.compare,
            client_key(request, body.session_id),
        )

        def generate():
            try:
                for event, data in events:
                    yield sse(event, data)
            except MedHelpError as exc:
                yield sse("error", {"status": exc.status_code, "detail": str(exc), "retry_after": exc.retry_after})
            except Exception:  # noqa: BLE001
                logger.exception("Unexpected error while streaming")
                yield sse("error", {"status": 500, "detail": "Unexpected server error."})

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # ----- voice -----
    @app.post("/transcribe", dependencies=[Depends(require_key)])
    def transcribe(request: Request, file: UploadFile = File(...), session_id: Optional[str] = Form(default=None)) -> dict[str, Any]:
        data = file.file.read(config.MAX_AUDIO_BYTES + 1)
        return svc(request).transcribe(
            data, file.filename or "question.wav", file.content_type or "audio/wav", client_key(request, session_id)
        )

    return app


def _warm_up(service: MedHelpService) -> None:
    try:
        service.engine.warm_up()
        logger.info("Warm-up finished (keyword index: %s).", service.engine.lexical_status)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Warm-up failed (the first request will retry): %s", exc)


app = create_app()
