from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    port: int
    speech_ready: bool
    missing_env: List[str]
    recognizer: Optional[str] = None
    model: str
    language_code: str


class TranscriptResponse(BaseModel):
    text: str
