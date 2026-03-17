from __future__ import annotations

from datetime import timedelta
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
        def __init__(self, *, credentials=None, client_options=None, transport=None):
            loaded["credentials"] = credentials
            loaded["api_endpoint"] = client_options.api_endpoint
            loaded["transport"] = transport
            self.transport = type("Transport", (), {"close": lambda self: None})()

        def recognize(self, request):
            return type("Response", (), {"results": []})()

    monkeypatch.setattr("app.speech.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.speech.speech_v2.SpeechClient", FakeSpeechClient)

    result = recognizer.transcribe(b"fake-audio", "en-US")

    assert loaded["path"] == str(credentials_path)
    assert loaded["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert loaded["api_endpoint"] == "us-speech.googleapis.com"
    assert loaded["transport"] == "grpc"
    assert loaded["credentials"] is not None
    assert result["text"] == ""


def test_google_speech_recognizer_streaming_uses_grpc_and_sends_config_first(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    recognizer = GoogleSpeechRecognizer(build_settings(credentials_path))

    loaded = {}
    partials = []
    finals = []

    def fake_load_credentials_from_file(path, scopes=None):
        loaded["path"] = path
        loaded["scopes"] = scopes
        return object(), "demo-project"

    class Alternative:
        def __init__(self, transcript, confidence):
            self.transcript = transcript
            self.confidence = confidence

    class Result:
        def __init__(self, transcript, *, is_final, seconds, language_code="en-US", confidence=0.98):
            self.alternatives = [Alternative(transcript, confidence)]
            self.is_final = is_final
            self.language_code = language_code
            self.result_end_offset = timedelta(seconds=seconds)

    class Response:
        def __init__(self, *results):
            self.results = list(results)

    class FakeSpeechClient:
        def __init__(self, *, credentials=None, client_options=None, transport=None):
            loaded["credentials"] = credentials
            loaded["api_endpoint"] = client_options.api_endpoint
            loaded["transport"] = transport
            self.transport = type("Transport", (), {"close": lambda self: None})()

        def streaming_recognize(self, *, requests):
            loaded["requests"] = list(requests)
            return [
                Response(Result("hello", is_final=False, seconds=0.5)),
                Response(Result("hello world", is_final=True, seconds=1.5)),
            ]

    monkeypatch.setattr("app.speech.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.speech.speech_v2.SpeechClient", FakeSpeechClient)

    result = recognizer.transcribe_streaming(
        [b"chunk-1", b"chunk-2"],
        language_code="en-US",
        source_sample_rate_hz=16_000,
        on_partial=partials.append,
        on_final=finals.append,
    )

    requests = loaded["requests"]
    assert len(requests) == 3
    assert requests[0].recognizer == "projects/demo-project/locations/us/recognizers/_"
    assert requests[0].streaming_config.config.model == "chirp_3"
    assert requests[0].streaming_config.config.explicit_decoding_config.sample_rate_hertz == 16_000
    assert requests[0].streaming_config.streaming_features.interim_results is True
    assert requests[1].audio == b"chunk-1"
    assert requests[2].audio == b"chunk-2"

    assert loaded["path"] == str(credentials_path)
    assert loaded["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert loaded["api_endpoint"] == "us-speech.googleapis.com"
    assert loaded["transport"] == "grpc"
    assert partials == ["hello"]
    assert finals == ["hello world"]
    assert result["text"] == "hello world"
    assert result["language_code"] == "en-US"
    assert result["model"] == "chirp_3"
    assert result["speech_seconds"] == 1.5
