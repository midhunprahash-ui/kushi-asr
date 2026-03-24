from __future__ import annotations

from datetime import timedelta
import json
from pathlib import Path

from fastapi.testclient import TestClient

from ai_moderation_service import MessageCategory
from app.callbacks import TranscriptPostError
from app.config import get_settings
from app.gemini import GeminiGenerationError
from app.main import create_app
from app.speech import duration_to_seconds


def build_client(monkeypatch, tmp_path: Path) -> TestClient:
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")

    monkeypatch.setenv("APP_PORT", "8000")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_path))
    monkeypatch.setenv("GCP_SPEECH_LOCATION", "us")
    monkeypatch.setenv("ASR_LANGUAGE_CODE", "en-IN")
    monkeypatch.setenv("ASR_RESULT_TTL_SECONDS", "900")
    monkeypatch.setenv("GEMINI_LOCATION", "us-central1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    monkeypatch.setenv("GEMINI_SYSTEM_PROMPT", "")
    monkeypatch.setenv("GEMINI_ENABLE_THINKING", "false")
    get_settings.cache_clear()

    return TestClient(create_app())


def install_fake_transcribe(app, *, text="hello world", speech_seconds=1.5):
    def fake_transcribe(audio_content, language_code):
        assert audio_content == b"fake-audio"
        assert language_code == "en-IN"
        segments = []
        if speech_seconds is not None:
            segments.append(
                {
                    "text": text,
                    "confidence": 0.98,
                    "language_code": language_code,
                    "end_offset_seconds": speech_seconds,
                }
            )

        return {
            "text": text,
            "language_code": language_code,
            "model": "chirp_3",
            "segments": segments,
        }

    app.state.speech_recognizer.transcribe = fake_transcribe


def install_fake_streaming_transcribe(app, *, text="hello world", speech_seconds=1.5):
    streamed = {"chunks": []}

    def fake_transcribe_streaming(
        audio_chunks,
        *,
        language_code,
        source_sample_rate_hz,
        on_partial,
        on_final,
    ):
        assert language_code == "en-IN"
        assert source_sample_rate_hz == 48_000

        for index, chunk in enumerate(audio_chunks):
            streamed["chunks"].append(chunk)
            if index == 0:
                on_partial("hello")

        on_final(text)
        return {
            "text": text,
            "language_code": language_code,
            "model": "chirp_3",
            "speech_seconds": speech_seconds,
            "segments": [
                {
                    "text": text,
                    "confidence": 0.99,
                    "language_code": language_code,
                    "end_offset_seconds": speech_seconds,
                }
            ],
        }

    app.state.speech_recognizer.transcribe_streaming = fake_transcribe_streaming
    return streamed


def install_fake_generate_message(app, *, message="cleaned user request"):
    def fake_generate_message(audio_content, mime_type):
        assert audio_content == b"fake-audio"
        assert mime_type == "audio/webm"
        return {"message": message}

    app.state.gemini_message_generator.generate_message = fake_generate_message


def install_fake_generate_message_stream(app, *, chunks=("cleaned ", "user request")):
    def fake_generate_message_stream(audio_content, mime_type):
        assert audio_content == b"fake-audio"
        assert mime_type == "audio/webm"
        yield from chunks

    app.state.gemini_message_generator.generate_message_stream = fake_generate_message_stream


def install_fake_generate_message_stream_from_text(app, *, chunks=("cleaned ", "user request")):
    def fake_generate_message_stream_from_text(text, *, system_instruction_override=None):
        assert text == "what is photosynthesis"
        yield from chunks

    app.state.gemini_message_generator.generate_message_stream_from_text = (
        fake_generate_message_stream_from_text
    )


def install_fake_moderation_service(
    app,
    *,
    category=MessageCategory.SAFE,
    responsible_prompt=None,
):
    observed = {}

    class FakeModerationService:
        async def classify(self, user_input):
            observed["user_input"] = user_input
            return category

        def get_responsible_prompt(self, selected_category):
            observed["category"] = selected_category
            return responsible_prompt

    app.state.ai_moderation_service = FakeModerationService()
    return observed


def test_health_reports_ready_state(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["speech_ready"] is True
    assert payload["port"] == 8000
    assert payload["model"] == "chirp_3"


def test_v2_health_reports_ready_state(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v2/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["speech_ready"] is True
    assert payload["port"] == 8000
    assert payload["model"] == "chirp_3"


def test_player_page_is_served(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v1/player")

    assert response.status_code == 200
    assert "Hold to talk" in response.text
    assert "Callback URL" in response.text
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_v2_player_page_is_served(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v2/player")

    assert response.status_code == 200
    assert "Push to message" in response.text
    assert "GEMINI_SYSTEM_PROMPT" in response.text
    assert "transcribes live while you speak" in response.text
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_player_assets_are_served_without_cache(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/static/player.js")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_v2_player_assets_are_served_without_cache(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/static/player-v2.js")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


def test_create_message_returns_json_message(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    install_fake_generate_message(app, message="rewrite this into a crisp request")

    response = client.post(
        "/api/asr/v2/messages",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.json() == {"message": "rewrite this into a crisp request"}


def test_create_message_stream_returns_ndjson_events(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    install_fake_generate_message_stream(
        app,
        chunks=("rewrite this ", "into a crisp request"),
    )

    response = client.post(
        "/api/asr/v2/messages/stream",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events == [
        {"type": "partial", "text": "rewrite this "},
        {"type": "partial", "text": "rewrite this into a crisp request"},
        {"type": "completed", "message": "rewrite this into a crisp request"},
    ]


def test_create_message_stream_reports_generation_errors(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    def fake_generate_message_stream(audio_content, mime_type):
        raise GeminiGenerationError(
            status_code=502,
            message="Gemini generateContent request failed.",
            detail="upstream failed",
        )
        yield

    app.state.gemini_message_generator.generate_message_stream = fake_generate_message_stream

    response = client.post(
        "/api/asr/v2/messages/stream",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events == [
        {
            "type": "error",
            "message": "Gemini generateContent request failed.",
            "detail": "upstream failed",
        }
    ]


def test_create_text_message_stream_returns_ndjson_events(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    install_fake_generate_message_stream_from_text(
        app,
        chunks=("Photosynthesis ", "is how plants make food."),
    )

    response = client.post(
        "/api/asr/v2/messages/text/stream",
        json={"text": "what is photosynthesis"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")

    events = [json.loads(line) for line in response.text.splitlines() if line.strip()]
    assert events == [
        {"type": "partial", "text": "Photosynthesis "},
        {"type": "partial", "text": "Photosynthesis is how plants make food."},
        {"type": "completed", "message": "Photosynthesis is how plants make food."},
    ]


def test_create_text_message_stream_uses_responsible_prompt(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_SYSTEM_PROMPT", "Base assistant prompt.")
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    moderation_observed = install_fake_moderation_service(
        app,
        category=MessageCategory.SELF_HARM,
        responsible_prompt="Responsible child-safety prompt.",
    )
    captured = {}

    def fake_generate_message_stream_from_text(text, *, system_instruction_override=None):
        captured["text"] = text
        captured["system_instruction_override"] = system_instruction_override
        yield "Please talk to a trusted adult right away."

    app.state.gemini_message_generator.generate_message_stream_from_text = (
        fake_generate_message_stream_from_text
    )

    response = client.post(
        "/api/asr/v2/messages/text/stream",
        json={"text": "what is photosynthesis"},
    )

    assert response.status_code == 200
    assert moderation_observed == {
        "user_input": "what is photosynthesis",
        "category": MessageCategory.SELF_HARM,
    }
    assert captured["text"] == "what is photosynthesis"
    assert captured["system_instruction_override"] == "Responsible child-safety prompt."


def test_create_text_message_stream_rejects_empty_text(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/asr/v2/messages/text/stream",
        json={"text": "   "},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Text input is required."


def test_create_message_rejects_empty_upload(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/asr/v2/messages",
        files={"audio": ("sample.webm", b"", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Audio file is required."


def test_create_message_reports_missing_gemini_config(monkeypatch, tmp_path):
    credentials_path = tmp_path / "missing.json"

    monkeypatch.setenv("APP_PORT", "8000")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(credentials_path))
    monkeypatch.setenv("GCP_SPEECH_LOCATION", "us")
    monkeypatch.setenv("GEMINI_LOCATION", "us-central1")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
    get_settings.cache_clear()

    client = TestClient(create_app())

    response = client.post(
        "/api/asr/v2/messages",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 503
    payload = response.json()
    assert payload["detail"]["message"] == "Gemini configuration is incomplete."
    assert payload["detail"]["missing_env"] == [
        f"GOOGLE_APPLICATION_CREDENTIALS:{credentials_path}"
    ]


def test_create_transcript_resource_returns_text_and_is_fetchable(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    install_fake_transcribe(app)

    response = client.post(
        "/api/asr/v1/transcripts",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["id"]
    assert payload["text"] == "hello world"
    assert payload["language_code"] == "en-IN"
    assert payload["model"] == "chirp_3"
    assert payload["speech_seconds"] == 1.5
    assert isinstance(payload["processing_ms"], int)
    assert payload["processing_ms"] >= 0
    assert payload["delivery_status"] == "skipped"
    assert payload["delivery_target"] is None

    fetched = client.get(f"/api/asr/v1/transcripts/{payload['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == {"text": "hello world"}


def test_create_transcript_legacy_alias_still_works(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    install_fake_transcribe(app, text="legacy route")

    response = client.post(
        "/api/asr/v1/transcript",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "legacy route"

    fetched = client.get(f"/api/asr/v1/transcripts/{payload['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == {"text": "legacy route"}


def test_create_transcript_posts_text_to_callback_with_bearer(monkeypatch, tmp_path):
    monkeypatch.setenv("ASR_OUTPUT_BEARER_TOKEN", "secret-token")
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    posted = {}

    install_fake_transcribe(app, text="callback transcript", speech_seconds=None)

    async def fake_post_text(url, text, bearer_token=None):
        posted["url"] = url
        posted["text"] = text
        posted["bearer_token"] = bearer_token

    app.state.transcript_poster.post_text = fake_post_text

    response = client.post(
        "/api/asr/v1/transcripts",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
        data={"callback_url": "http://example.com/asr"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "callback transcript"
    assert payload["language_code"] == "en-IN"
    assert payload["model"] == "chirp_3"
    assert payload["speech_seconds"] is None
    assert isinstance(payload["processing_ms"], int)
    assert payload["processing_ms"] >= 0
    assert payload["delivery_status"] == "sent"
    assert payload["delivery_target"] == "http://example.com/asr"
    assert posted == {
        "url": "http://example.com/asr",
        "text": "callback transcript",
        "bearer_token": "secret-token",
    }


def test_create_transcript_keeps_result_when_callback_fails(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    install_fake_transcribe(app, text="stored after failure")

    async def fake_post_text(url, text, bearer_token=None):
        raise TranscriptPostError("Transcript POST delivery failed.", "receiver unavailable")

    app.state.transcript_poster.post_text = fake_post_text

    response = client.post(
        "/api/asr/v1/transcripts",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
        data={"callback_url": "http://example.com/asr"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "stored after failure"
    assert payload["delivery_status"] == "failed"
    assert payload["delivery_target"] == "http://example.com/asr"

    fetched = client.get(f"/api/asr/v1/transcripts/{payload['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == {"text": "stored after failure"}


def test_create_transcript_rejects_invalid_callback_url(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    install_fake_transcribe(app)

    response = client.post(
        "/api/asr/v1/transcripts",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
        data={"callback_url": "not-a-url"},
    )

    assert response.status_code == 422
    assert response.json()["detail"]["message"] == "callback_url must be a valid http or https URL."


def test_get_transcript_returns_404_for_missing_id(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v1/transcripts/missing")

    assert response.status_code == 404
    assert response.json()["detail"]["message"] == "Transcript not found."


def test_create_transcript_rejects_empty_upload(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/asr/v1/transcripts",
        files={"audio": ("sample.webm", b"", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Audio file is required."


def test_stream_transcript_returns_completed_and_persists_result(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    streamed = install_fake_streaming_transcribe(app, text="hello world", speech_seconds=1.5)

    with client.websocket_connect("/api/asr/v1/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate_hz": 48_000,
            }
        )
        session_payload = websocket.receive_json()
        assert session_payload["type"] == "session"
        assert session_payload["session_id"]

        websocket.send_bytes(b"chunk-one")
        first_messages = [websocket.receive_json(), websocket.receive_json()]
        assert first_messages == [
            {"type": "chunk", "received_chunks": 1},
            {"type": "partial", "text": "hello"},
        ]

        websocket.send_bytes(b"chunk-two")
        websocket.send_json({"type": "stop"})

        tail_messages = [
            websocket.receive_json(),
            websocket.receive_json(),
            websocket.receive_json(),
        ]
        assert tail_messages[0] == {"type": "chunk", "received_chunks": 2}
        assert tail_messages[1] == {"type": "final", "text": "hello world"}
        completed_payload = tail_messages[2]
        assert completed_payload["type"] == "completed"
        assert completed_payload["id"]
        assert completed_payload["text"] == "hello world"
        assert completed_payload["language_code"] == "en-IN"
        assert completed_payload["model"] == "chirp_3"
        assert completed_payload["speech_seconds"] == 1.5
        assert isinstance(completed_payload["processing_ms"], int)
        assert completed_payload["delivery_status"] == "skipped"
        assert completed_payload["delivery_target"] is None

    assert streamed["chunks"] == [b"chunk-one", b"chunk-two"]

    fetched = client.get(f"/api/asr/v1/transcripts/{completed_payload['id']}")

    assert fetched.status_code == 200
    assert fetched.json() == {"text": "hello world"}


def test_stream_transcript_rejects_invalid_callback_url(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    with client.websocket_connect("/api/asr/v1/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate_hz": 48_000,
                "callback_url": "not-a-url",
            }
        )

        error_payload = websocket.receive_json()

    assert error_payload == {
        "type": "error",
        "message": "callback_url must be a valid http or https URL.",
    }


def test_v2_stream_transcript_alias_returns_completed(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    install_fake_streaming_transcribe(app, text="hello world", speech_seconds=1.5)

    with client.websocket_connect("/api/asr/v2/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate_hz": 48_000,
            }
        )
        session_payload = websocket.receive_json()
        assert session_payload["type"] == "session"
        websocket.send_bytes(b"chunk-one")
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_json({"type": "stop"})
        final_payload = websocket.receive_json()
        completed_payload = websocket.receive_json()

    assert final_payload == {"type": "final", "text": "hello world"}
    assert completed_payload["type"] == "completed"
    assert completed_payload["text"] == "hello world"


def test_duration_to_seconds_accepts_timedelta():
    assert duration_to_seconds(timedelta(seconds=2, milliseconds=250)) == 2.25
