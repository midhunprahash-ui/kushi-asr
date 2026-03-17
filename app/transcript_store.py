from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import threading
from typing import Dict, Optional
from uuid import uuid4


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class StoredTranscript:
    transcript_id: str
    text: str
    language_code: Optional[str]
    model: Optional[str]
    speech_seconds: Optional[float]
    processing_ms: int
    expires_at: datetime
    created_at: datetime = field(default_factory=utc_now)
    delivery_status: str = "skipped"
    delivery_target: Optional[str] = None


class TranscriptStore:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._results: Dict[str, StoredTranscript] = {}
        self._lock = threading.Lock()

    def create(
        self,
        *,
        text: str,
        language_code: Optional[str],
        model: Optional[str],
        speech_seconds: Optional[float],
        processing_ms: int,
    ) -> StoredTranscript:
        self.cleanup_expired()

        result = StoredTranscript(
            transcript_id=uuid4().hex,
            text=text,
            language_code=language_code,
            model=model,
            speech_seconds=speech_seconds,
            processing_ms=processing_ms,
            expires_at=utc_now() + timedelta(seconds=self.ttl_seconds),
        )

        with self._lock:
            self._results[result.transcript_id] = result

        return result

    def get(self, transcript_id: str) -> Optional[StoredTranscript]:
        self.cleanup_expired()
        with self._lock:
            return self._results.get(transcript_id)

    def mark_delivery(
        self,
        transcript_id: str,
        *,
        status: str,
        target: Optional[str],
    ) -> Optional[StoredTranscript]:
        with self._lock:
            result = self._results.get(transcript_id)
            if result is None:
                return None

            result.delivery_status = status
            result.delivery_target = target
            return result

    def cleanup_expired(self) -> None:
        now = utc_now()

        with self._lock:
            expired_ids = [
                transcript_id
                for transcript_id, result in self._results.items()
                if now >= result.expires_at
            ]

            for transcript_id in expired_ids:
                self._results.pop(transcript_id, None)
