from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.speech import GoogleSpeechRecognizer


def build_settings(credentials_path: Path) -> Settings:
    return Settings(
        APP_PORT=8000,
        GOOGLE_CLOUD_PROJECT="demo-project",
        GOOGLE_APPLICATION_CREDENTIALS=str(credentials_path),
        GCP_SPEECH_LOCATION="us",
        GCP_SPEECH_RECOGNIZER="_",
    )


def test_google_speech_recognizer_uses_configured_credentials(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    recognizer = GoogleSpeechRecognizer(build_settings(credentials_path))

    loaded = {}

    def fake_load_credentials_from_file(path, scopes=None):
        loaded["path"] = path
        loaded["scopes"] = scopes
        return object(), "demo-project"

    class FakeSpeechClient:
        def __init__(self, *, credentials=None, client_options=None):
            loaded["credentials"] = credentials
            loaded["api_endpoint"] = client_options.api_endpoint
            self.transport = type("Transport", (), {"close": lambda self: None})()

        def recognize(self, request):
            return type("Response", (), {"results": []})()

    monkeypatch.setattr("app.speech.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.speech.speech_v2.SpeechClient", FakeSpeechClient)

    result = recognizer.transcribe(b"fake-audio", "en-US")

    assert loaded["path"] == str(credentials_path)
    assert loaded["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert loaded["api_endpoint"] == "us-speech.googleapis.com"
    assert loaded["credentials"] is not None
    assert result["text"] == ""
