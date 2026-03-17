from __future__ import annotations

from typing import List, Literal, Optional

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
    id: str
    text: str
    language_code: Optional[str] = None
    model: Optional[str] = None
    speech_seconds: Optional[float] = None
    processing_ms: int
    delivery_status: Literal["sent", "skipped", "failed"]
    delivery_target: Optional[str] = None


class TranscriptTextResponse(BaseModel):
    text: str


class MessageResponse(BaseModel):
    message: str
