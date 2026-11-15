"""Voice questions: audio in, text out, through Groq's hosted Whisper (free-tier limits apply).

The transcript is only *returned*; the front-end shows it in an editable box and the user decides whether to
send it, because speech recognition can mishear drug names and a wrong word silently changes a medical
question. Audio is processed in memory and never stored.
"""
from __future__ import annotations

import logging
from typing import Optional

import requests

from app import config

logger = logging.getLogger("medhelp.stt")

# Whisper accepts a short "prompt" that biases spelling towards expected vocabulary.
MEDICAL_PROMPT = (
    "Questions about clinical practice guidelines: hypertension, HbA1c, SGLT2 inhibitors, GLP-1, statins, "
    "KDIGO, GINA, GOLD COPD, atrial fibrillation, anticoagulation, antenatal care, tuberculosis, malaria, hepatitis B."
)

ALLOWED_AUDIO_TYPES = {
    "audio/wav", "audio/x-wav", "audio/wave", "audio/webm", "audio/ogg", "audio/mpeg", "audio/mp3",
    "audio/mp4", "audio/m4a", "audio/x-m4a", "video/webm", "application/octet-stream",
}


class Transcriber:
    def __init__(
        self,
        api_key: str,
        model: str = "whisper-large-v3-turbo",
        base_url: str = "https://api.groq.com/openai/v1",
        language: str = "en",
        timeout: float = 45,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.api_key, self.model, self.language, self.timeout = api_key, model, language, timeout
        self.base_url = base_url.rstrip("/")
        self.session = session or requests.Session()

    def transcribe(self, audio: bytes, filename: str = "question.wav", content_type: str = "audio/wav") -> str:
        data = {"model": self.model, "response_format": "json", "temperature": "0", "prompt": MEDICAL_PROMPT}
        if self.language:
            data["language"] = self.language
        try:
            resp = self.session.post(
                f"{self.base_url}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={"file": (filename, audio, content_type)},
                data=data,
                timeout=self.timeout,
            )
        except requests.Timeout as exc:
            raise RuntimeError(f"Request to Groq timed out after {self.timeout}s") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Could not reach Groq: {exc}") from exc
        if resp.status_code != 200:
            raise RuntimeError(f"Error code: {resp.status_code} - {resp.text[:300]}")
        try:
            text = str(resp.json().get("text", ""))
        except ValueError as exc:
            raise RuntimeError(f"Unexpected response from Groq: {resp.text[:200]}") from exc
        return " ".join(text.split())


def build_transcriber() -> Transcriber:
    config.validate(need_qdrant=False)
    return Transcriber(
        api_key=config.GROQ_API_KEY,
        model=config.GROQ_STT_MODEL,
        base_url=config.GROQ_BASE_URL,
        language=config.STT_LANGUAGE,
    )
