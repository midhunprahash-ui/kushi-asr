from __future__ import annotations

import asyncio
import json
from pathlib import Path
import threading
import time
from typing import Optional

from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AnyHttpUrl, TypeAdapter, ValidationError
import uvicorn

from app.callbacks import TranscriptPostError, TranscriptPoster
from app.config import Settings, get_settings
from app.models import HealthResponse, TranscriptResponse, TranscriptTextResponse
from app.speech import GoogleSpeechRecognizer, SpeechRecognitionError
from app.session_manager import SessionManager, TranscriptSession
from app.transcript_store import StoredTranscript, TranscriptStore

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
CALLBACK_URL_ADAPTER = TypeAdapter(AnyHttpUrl)
CACHE_CONTROL_NO_STORE = "no-store, no-cache, must-revalidate"


class NoCacheStaticFiles(StaticFiles):
    def file_response(self, full_path, stat_result, scope, status_code=200):
        response = FileResponse(full_path, status_code=status_code, stat_result=stat_result)
        response.headers["Cache-Control"] = CACHE_CONTROL_NO_STORE
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


def validate_callback_url(value: str, *, source: str, status_code: int) -> str:
    try:
        return str(CALLBACK_URL_ADAPTER.validate_python(value))
    except ValidationError as exc:
        raise HTTPException(
            status_code=status_code,
            detail={
                "message": f"{source} must be a valid http or https URL.",
                "detail": exc.errors(),
            },
        ) from exc


def resolve_callback_url(callback_url: Optional[str], settings: Settings) -> Optional[str]:
    candidate = (callback_url or "").strip()
    if candidate:
        return validate_callback_url(candidate, source="callback_url", status_code=422)

    configured_url = (settings.asr_output_post_url or "").strip()
    if configured_url:
        return validate_callback_url(
            configured_url,
            source="ASR_OUTPUT_POST_URL",
            status_code=500,
        )

    return None


def extract_speech_seconds(result: dict) -> Optional[float]:
    speech_seconds = result.get("speech_seconds")
    if speech_seconds is not None:
        return speech_seconds

    return max(
        (
            segment.get("end_offset_seconds")
            for segment in result.get("segments", [])
            if segment.get("end_offset_seconds") is not None
        ),
        default=None,
    )


def build_transcript_response(result: StoredTranscript) -> TranscriptResponse:
    return TranscriptResponse(
        id=result.transcript_id,
        text=result.text,
        language_code=result.language_code,
        model=result.model,
        speech_seconds=result.speech_seconds,
        processing_ms=result.processing_ms,
        delivery_status=result.delivery_status,
        delivery_target=result.delivery_target,
    )


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
    app.state.session_manager = SessionManager(settings.asr_session_ttl_seconds)
    app.state.transcript_store = TranscriptStore(settings.asr_result_ttl_seconds)
    app.mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static")

    router = APIRouter(prefix=settings.api_prefix)

    async def deliver_transcript(
        text: str,
        callback_url: Optional[str],
    ) -> tuple[str, Optional[str]]:
        downstream_url = resolve_callback_url(callback_url, settings)
        if not downstream_url:
            return "skipped", None

        try:
            await app.state.transcript_poster.post_text(
                downstream_url,
                text,
                bearer_token=settings.asr_output_bearer_token,
            )
        except TranscriptPostError:
            return "failed", downstream_url

        return "sent", downstream_url

    def create_stored_transcript(result: dict, processing_ms: int) -> StoredTranscript:
        return app.state.transcript_store.create(
            text=result["text"],
            language_code=result.get("language_code"),
            model=result.get("model"),
            speech_seconds=extract_speech_seconds(result),
            processing_ms=processing_ms,
        )

    def apply_delivery_status(
        transcript: StoredTranscript,
        delivery_status: str,
        delivery_target: Optional[str],
    ) -> StoredTranscript:
        stored = app.state.transcript_store.mark_delivery(
            transcript.transcript_id,
            status=delivery_status,
            target=delivery_target,
        )
        return stored or transcript

    def iter_session_audio(session: TranscriptSession):
        while True:
            chunk = session.audio_queue.get()
            if chunk is None:
                break
            yield chunk

    def run_streaming_session(session: TranscriptSession) -> None:
        started_at = time.perf_counter()

        def on_partial(text: str) -> None:
            session.schedule_event({"type": "partial", "text": text})

        def on_final(text: str) -> None:
            session.append_final(text)
            session.schedule_event({"type": "final", "text": text})

        try:
            result = app.state.speech_recognizer.transcribe_streaming(
                iter_session_audio(session),
                language_code=session.language_code,
                source_sample_rate_hz=session.sample_rate_hz,
                on_partial=on_partial,
                on_final=on_final,
            )
            if session.terminated:
                return

            processing_ms = round((time.perf_counter() - started_at) * 1000)
            transcript = create_stored_transcript(result, processing_ms)
            delivery_status, delivery_target = asyncio.run(
                deliver_transcript(result["text"], session.callback_url)
            )
            transcript = apply_delivery_status(transcript, delivery_status, delivery_target)
            session.schedule_event(
                {
                    "type": "completed",
                    **build_transcript_response(transcript).model_dump(),
                }
            )
            session.schedule_close()
        except SpeechRecognitionError as exc:
            if not session.terminated:
                session.schedule_event({"type": "error", "message": exc.message})
                session.schedule_close(code=1011, reason="streaming recognition failed")
        except Exception:
            if not session.terminated:
                session.schedule_event(
                    {
                        "type": "error",
                        "message": "Unexpected streaming failure.",
                    }
                )
                session.schedule_close(code=1011, reason="streaming failure")
        finally:
            session.close_audio_input()
            session.terminate()
            app.state.session_manager.remove_session(session.session_id)

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
        return FileResponse(
            STATIC_DIR / "player.html",
            headers={
                "Cache-Control": CACHE_CONTROL_NO_STORE,
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )

    async def create_transcript_record(
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
            started_at = time.perf_counter()
            result = await asyncio.to_thread(
                app.state.speech_recognizer.transcribe,
                audio_bytes,
                requested_language,
            )
            processing_ms = round((time.perf_counter() - started_at) * 1000)
        except SpeechRecognitionError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail={"message": exc.message, "detail": exc.detail},
            ) from exc

        transcript = create_stored_transcript(result, processing_ms)
        delivery_status, delivery_target = await deliver_transcript(result["text"], callback_url)
        transcript = apply_delivery_status(transcript, delivery_status, delivery_target)

        return build_transcript_response(transcript)

    @router.post("/transcripts", response_model=TranscriptResponse)
    async def create_transcripts(
        audio: UploadFile = File(...),
        language_code: Optional[str] = Form(default=None),
        callback_url: Optional[str] = Form(default=None),
    ) -> TranscriptResponse:
        return await create_transcript_record(audio, language_code, callback_url)

    @router.post("/transcript", response_model=TranscriptResponse)
    async def create_transcript(
        audio: UploadFile = File(...),
        language_code: Optional[str] = Form(default=None),
        callback_url: Optional[str] = Form(default=None),
    ) -> TranscriptResponse:
        return await create_transcript_record(audio, language_code, callback_url)

    @router.get("/transcripts/{transcript_id}", response_model=TranscriptTextResponse)
    async def get_transcript(transcript_id: str) -> TranscriptTextResponse:
        transcript = app.state.transcript_store.get(transcript_id)
        if transcript is None:
            raise HTTPException(
                status_code=404,
                detail={"message": "Transcript not found."},
            )

        return TranscriptTextResponse(text=transcript.text)

    @router.websocket("/stream")
    async def stream_transcript(websocket: WebSocket) -> None:
        await websocket.accept()
        session: Optional[TranscriptSession] = None
        received_chunks = 0

        async def close_with_error(message: str, code: int = 1003) -> None:
            await websocket.send_json({"type": "error", "message": message})
            await websocket.close(code=code, reason=message[:123])

        try:
            initial_message = await websocket.receive()
            if initial_message.get("text") is None:
                await close_with_error("The first WebSocket message must be a JSON start payload.")
                return

            try:
                payload = json.loads(initial_message["text"])
            except json.JSONDecodeError:
                await close_with_error("The first WebSocket message must be valid JSON.")
                return

            if payload.get("type") != "start":
                await close_with_error("The first WebSocket message must have type=start.")
                return

            language_code = payload.get("language_code") or settings.asr_language_code
            callback_url = payload.get("callback_url")
            if callback_url and not isinstance(callback_url, str):
                await close_with_error("callback_url must be a string.")
                return

            normalized_callback_url = None
            if callback_url:
                try:
                    normalized_callback_url = validate_callback_url(
                        callback_url.strip(),
                        source="callback_url",
                        status_code=422,
                    )
                except HTTPException:
                    await close_with_error("callback_url must be a valid http or https URL.")
                    return

            sample_rate_hz = payload.get("sample_rate_hz")
            if not isinstance(sample_rate_hz, int) or sample_rate_hz <= 0:
                await close_with_error("sample_rate_hz must be a positive integer.")
                return

            session = app.state.session_manager.create_session(
                language_code=language_code,
                sample_rate_hz=sample_rate_hz,
                callback_url=normalized_callback_url,
            )
            session.bind(websocket, asyncio.get_running_loop())
            session.streaming_thread = threading.Thread(
                target=run_streaming_session,
                args=(session,),
                daemon=True,
            )
            session.streaming_thread.start()

            await websocket.send_json({"type": "session", "session_id": session.session_id})

            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    break

                if message.get("bytes") is not None:
                    session.enqueue_audio(message["bytes"])
                    received_chunks += 1
                    await websocket.send_json(
                        {
                            "type": "chunk",
                            "received_chunks": received_chunks,
                        }
                    )
                    continue

                control_text = message.get("text")
                if control_text is None:
                    continue

                try:
                    control_message = json.loads(control_text)
                except json.JSONDecodeError:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Control messages must be valid JSON.",
                        }
                    )
                    continue

                if control_message.get("type") == "stop":
                    session.close_audio_input()
                    continue

                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Unsupported control message.",
                    }
                )
        except WebSocketDisconnect:
            pass
        finally:
            if session is not None and not session.audio_closed:
                session.close_audio_input()
            if session is not None and not session.terminated:
                session.terminate()
                app.state.session_manager.remove_session(session.session_id)

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url=f"{settings.api_prefix}/player")

    app.include_router(router)
    return app


app = create_app()


if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.app_host, port=settings.app_port, reload=False)
