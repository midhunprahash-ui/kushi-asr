from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from queue import Queue
import threading
from typing import Any, Dict, List, Optional
from uuid import uuid4

from fastapi import WebSocket


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _swallow_future_exception(future: asyncio.Future[Any]) -> None:
    try:
        future.result()
    except Exception:
        return


@dataclass
class TranscriptSession:
    session_id: str
    language_code: str
    expires_at: datetime
    audio_queue: Queue = field(default_factory=Queue)
    created_at: datetime = field(default_factory=utc_now)
    websocket: Optional[WebSocket] = None
    loop: Optional[asyncio.AbstractEventLoop] = None
    streaming_thread: Optional[threading.Thread] = None
    final_segments: List[str] = field(default_factory=list)
    audio_closed: bool = False
    terminated: bool = False
    _state_lock: threading.Lock = field(default_factory=threading.Lock)

    def bind(self, websocket: WebSocket, loop: asyncio.AbstractEventLoop) -> None:
        self.websocket = websocket
        self.loop = loop

    def enqueue_audio(self, audio_chunk: bytes) -> None:
        if not audio_chunk or self.audio_closed:
            return
        self.audio_queue.put(audio_chunk)

    def close_audio_input(self) -> None:
        with self._state_lock:
            if self.audio_closed:
                return
            self.audio_closed = True
            self.audio_queue.put(None)

    def terminate(self) -> None:
        with self._state_lock:
            self.terminated = True

    def append_final(self, transcript: str) -> None:
        cleaned = transcript.strip()
        if cleaned:
            self.final_segments.append(cleaned)

    @property
    def full_transcript(self) -> str:
        return " ".join(self.final_segments).strip()

    def schedule_event(self, payload: Dict[str, Any]) -> None:
        if self.terminated or self.websocket is None or self.loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self.websocket.send_json(payload), self.loop)
        future.add_done_callback(_swallow_future_exception)

    def schedule_close(self, code: int = 1000, reason: str = "session complete") -> None:
        if self.terminated or self.websocket is None or self.loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self.websocket.close(code=code, reason=reason),
            self.loop,
        )
        future.add_done_callback(_swallow_future_exception)


class SessionManager:
    def __init__(self, ttl_seconds: int) -> None:
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, TranscriptSession] = {}
        self._lock = threading.Lock()

    def create_session(self, language_code: str) -> TranscriptSession:
        self.cleanup_expired()

        session = TranscriptSession(
            session_id=uuid4().hex,
            language_code=language_code,
            expires_at=utc_now() + timedelta(seconds=self.ttl_seconds),
        )

        with self._lock:
            self._sessions[session.session_id] = session

        return session

    def get_session(self, session_id: str) -> Optional[TranscriptSession]:
        self.cleanup_expired()
        with self._lock:
            return self._sessions.get(session_id)

    def remove_session(self, session_id: str) -> Optional[TranscriptSession]:
        with self._lock:
            return self._sessions.pop(session_id, None)

    def cleanup_expired(self) -> None:
        expired_sessions: List[TranscriptSession] = []
        now = utc_now()

        with self._lock:
            for session_id, session in list(self._sessions.items()):
                if session.websocket is None and now >= session.expires_at:
                    expired_sessions.append(self._sessions.pop(session_id))

        for session in expired_sessions:
            session.close_audio_input()
            session.terminate()
