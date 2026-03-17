from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from app.callbacks import TranscriptPostError
from app.config import get_settings
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


def test_health_reports_ready_state(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/api/asr/v1/health")

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


def test_player_assets_are_served_without_cache(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.get("/static/player.js")

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store, no-cache, must-revalidate"


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


def test_duration_to_seconds_accepts_timedelta():
    assert duration_to_seconds(timedelta(seconds=2, milliseconds=250)) == 2.25
