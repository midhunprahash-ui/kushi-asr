from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.gemini import VertexGeminiMessageGenerator


def build_settings(credentials_path: Path) -> Settings:
    return Settings(
        APP_PORT=8000,
        GOOGLE_CLOUD_PROJECT="demo-project",
        GOOGLE_APPLICATION_CREDENTIALS=str(credentials_path),
        GEMINI_LOCATION="us-central1",
        GEMINI_MODEL="gemini-2.5-flash",
        GEMINI_SYSTEM_PROMPT="Rewrite the spoken audio into a crisp message.",
    )


def test_vertex_gemini_generator_uses_vertex_credentials_and_returns_message(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(build_settings(credentials_path))

    loaded = {}

    def fake_load_credentials_from_file(path, scopes=None):
        loaded["path"] = path
        loaded["scopes"] = scopes
        return object(), "demo-project"

    class FakeResponse:
        parsed = {"message": "cleaned prompt"}
        text = '{"message":"cleaned prompt"}'

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            loaded["model"] = model
            loaded["contents"] = contents
            loaded["config"] = config
            return FakeResponse()

    class FakeClient:
        def __init__(
            self,
            *,
            vertexai=None,
            credentials=None,
            project=None,
            location=None,
            http_options=None,
        ):
            loaded["vertexai"] = vertexai
            loaded["credentials"] = credentials
            loaded["project"] = project
            loaded["location"] = location
            loaded["http_options"] = http_options
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    payload = generator.generate_message(b"fake-audio", "audio/webm")

    assert loaded["path"] == str(credentials_path)
    assert loaded["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert loaded["vertexai"] is True
    assert loaded["credentials"] is not None
    assert loaded["project"] == "demo-project"
    assert loaded["location"] == "us-central1"
    assert loaded["http_options"].api_version == "v1"
    assert loaded["model"] == "gemini-2.5-flash"
    assert loaded["contents"][0] == (
        "Use the attached audio as the user's input. Apply the system instruction to that "
        "spoken content and return only the final text output in the JSON field `message`."
    )
    assert loaded["contents"][1].inline_data.mime_type == "audio/webm"
    assert loaded["contents"][1].inline_data.data == b"fake-audio"
    assert loaded["config"].system_instruction == "Rewrite the spoken audio into a crisp message."
    assert loaded["config"].response_mime_type == "application/json"
    assert payload == {"message": "cleaned prompt"}


def test_vertex_gemini_generator_parses_json_text_when_parsed_payload_is_missing(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(build_settings(credentials_path))

    def fake_load_credentials_from_file(path, scopes=None):
        return object(), "demo-project"

    class FakeResponse:
        parsed = None
        text = '{"message":"json fallback"}'

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    payload = generator.generate_message(b"fake-audio", "audio/webm")

    assert payload == {"message": "json fallback"}
