from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

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
    get_settings.cache_clear()

    return TestClient(create_app())


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


def test_create_transcript_returns_text(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app

    def fake_transcribe(audio_content, language_code):
        assert audio_content == b"fake-audio"
        assert language_code == "en-US"
        return {
            "text": "hello world",
            "language_code": language_code,
            "model": "chirp_3",
            "segments": [
                {
                    "text": "hello world",
                    "confidence": 0.98,
                    "language_code": language_code,
                    "end_offset_seconds": 1.5,
                }
            ],
        }

    app.state.speech_recognizer.transcribe = fake_transcribe

    response = client.post(
        "/api/asr/v1/transcript",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["text"] == "hello world"
    assert list(payload.keys()) == ["text"]


def test_create_transcript_posts_text_to_callback(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)
    app = client.app
    posted = {}

    def fake_transcribe(audio_content, language_code):
        return {
            "text": "callback transcript",
            "language_code": language_code,
            "model": "chirp_3",
            "segments": [],
        }

    async def fake_post_text(url, text):
        posted["url"] = url
        posted["text"] = text

    app.state.speech_recognizer.transcribe = fake_transcribe
    app.state.transcript_poster.post_text = fake_post_text

    response = client.post(
        "/api/asr/v1/transcript",
        files={"audio": ("sample.webm", b"fake-audio", "audio/webm")},
        data={"callback_url": "http://example.com/asr"},
    )

    assert response.status_code == 200
    assert response.json() == {"text": "callback transcript"}
    assert posted == {
        "url": "http://example.com/asr",
        "text": "callback transcript",
    }


def test_create_transcript_rejects_empty_upload(monkeypatch, tmp_path):
    client = build_client(monkeypatch, tmp_path)

    response = client.post(
        "/api/asr/v1/transcript",
        files={"audio": ("sample.webm", b"", "audio/webm")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["message"] == "Audio file is required."


def test_duration_to_seconds_accepts_timedelta():
    assert duration_to_seconds(timedelta(seconds=2, milliseconds=250)) == 2.25
