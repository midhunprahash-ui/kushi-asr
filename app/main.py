from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.callbacks import TranscriptPostError, TranscriptPoster
from app.config import get_settings
from app.models import HealthResponse, TranscriptResponse
from app.speech import GoogleSpeechRecognizer, SpeechRecognitionError

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Kushi ASR",
        version="0.1.0",
        summary="Push-to-talk transcription via Google Speech-to-Text v2.",
    )
    app.state.speech_recognizer = GoogleSpeechRecognizer(settings)
    app.state.transcript_poster = TranscriptPoster(
        timeout_seconds=settings.asr_output_post_timeout_seconds,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    router = APIRouter(prefix=settings.api_prefix)

    @router.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        missing_env = settings.missing_google_env()
        return HealthResponse(
            status="ok" if not missing_env else "degraded",
            service=settings.app_name,
            port=settings.app_port,
            speech_ready=not missing_env,
            missing_env=missing_env,
            recognizer=settings.recognizer_path if settings.google_cloud_project else None,
            model=settings.asr_model,
            language_code=settings.asr_language_code,
        )

    @router.get("/player")
    async def player() -> FileResponse:
        return FileResponse(STATIC_DIR / "player.html")

    @router.post("/transcript", response_model=TranscriptResponse)
    async def create_transcript(
        audio: UploadFile = File(...),
        language_code: Optional[str] = Form(default=None),
        callback_url: Optional[str] = Form(default=None),
    ) -> TranscriptResponse:
        missing_env = settings.missing_google_env()
        if missing_env:
            raise HTTPException(
                status_code=503,
                detail={
                    "message": "Google Speech-to-Text configuration is incomplete.",
                    "missing_env": missing_env,
                },
            )

        audio_bytes = await audio.read(settings.asr_max_upload_bytes + 1)
        if not audio_bytes:
            raise HTTPException(status_code=400, detail={"message": "Audio file is required."})

        if len(audio_bytes) > settings.asr_max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail={
                    "message": "Audio file is too large for synchronous recognition.",
                    "max_upload_bytes": settings.asr_max_upload_bytes,
                },
            )

        requested_language = language_code or settings.asr_language_code

        try:
            result = await asyncio.to_thread(
                app.state.speech_recognizer.transcribe,
                audio_bytes,
                requested_language,
            )
        except SpeechRecognitionError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"message": exc.message, "detail": exc.detail},
            ) from exc

        downstream_url = callback_url or settings.asr_output_post_url
        if downstream_url:
            try:
                await app.state.transcript_poster.post_text(downstream_url, result["text"])
            except TranscriptPostError as exc:
                raise HTTPException(
                    status_code=502,
                    detail={"message": exc.message, "detail": exc.detail},
                ) from exc

        return TranscriptResponse(
            text=result["text"],
        )

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url=f"{settings.api_prefix}/player")

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
