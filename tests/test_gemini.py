from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.gemini import VertexGeminiMessageGenerator


def build_settings(credentials_path: Path, **overrides) -> Settings:
    values = {
        "APP_PORT": 8000,
        "GOOGLE_CLOUD_PROJECT": "demo-project",
        "GOOGLE_APPLICATION_CREDENTIALS": str(credentials_path),
        "GEMINI_LOCATION": "us-central1",
        "GEMINI_MODEL": "gemini-2.5-flash-lite",
        "GEMINI_SYSTEM_PROMPT": "Rewrite the spoken audio into a crisp message.",
        "GEMINI_ENABLE_THINKING": "false",
    }
    values.update(overrides)
    return Settings(**values)


def test_vertex_gemini_generator_reuses_vertex_client_and_returns_plain_text(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(build_settings(credentials_path))

    loaded = {"credential_calls": 0, "client_inits": 0}

    def fake_load_credentials_from_file(path, scopes=None):
        loaded["credential_calls"] += 1
        loaded["path"] = path
        loaded["scopes"] = scopes
        return object(), "demo-project"

    class FakeResponse:
        text = "cleaned prompt"

    class FakeStreamChunk:
        def __init__(self, text):
            self.text = text

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            loaded.setdefault("sync_models", []).append(model)
            loaded.setdefault("sync_contents_calls", []).append(contents)
            loaded.setdefault("sync_configs", []).append(config)
            return FakeResponse()

        def generate_content_stream(self, *, model, contents, config):
            loaded.setdefault("stream_models", []).append(model)
            loaded.setdefault("stream_contents_calls", []).append(contents)
            loaded.setdefault("stream_configs", []).append(config)
            return iter([FakeStreamChunk("cleaned "), FakeStreamChunk("prompt")])

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
            loaded["client_inits"] += 1
            loaded["vertexai"] = vertexai
            loaded["credentials"] = credentials
            loaded["project"] = project
            loaded["location"] = location
            loaded["http_options"] = http_options
            self.models = FakeModels()

        def close(self):
            loaded["closed"] = True

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    payload = generator.generate_message(b"fake-audio", "audio/webm")
    text_payload = generator.generate_message_from_text("what is photosynthesis?")
    chunks = list(generator.generate_message_stream(b"fake-audio", "audio/webm"))
    text_chunks = list(generator.generate_message_stream_from_text("what is photosynthesis?"))

    assert payload == {"message": "cleaned prompt"}
    assert text_payload == {"message": "cleaned prompt"}
    assert chunks == ["cleaned ", "prompt"]
    assert text_chunks == ["cleaned ", "prompt"]
    assert loaded["credential_calls"] == 1
    assert loaded["client_inits"] == 1
    assert loaded["path"] == str(credentials_path)
    assert loaded["scopes"] == ["https://www.googleapis.com/auth/cloud-platform"]
    assert loaded["vertexai"] is True
    assert loaded["credentials"] is not None
    assert loaded["project"] == "demo-project"
    assert loaded["location"] == "us-central1"
    assert loaded["http_options"].api_version == "v1"
    assert loaded["sync_models"] == ["gemini-2.5-flash-lite", "gemini-2.5-flash-lite"]
    assert loaded["stream_models"] == ["gemini-2.5-flash-lite", "gemini-2.5-flash-lite"]
    assert loaded["sync_contents_calls"][0][0] == (
        "Treat the attached audio as the user's spoken input. Apply the system instruction to "
        "that spoken content and return only the final answer text. Do not wrap the answer in "
        "JSON."
    )
    assert loaded["sync_contents_calls"][0][1].inline_data.mime_type == "audio/webm"
    assert loaded["sync_contents_calls"][0][1].inline_data.data == b"fake-audio"
    assert loaded["sync_contents_calls"][1] == ["what is photosynthesis?"]
    assert loaded["sync_configs"][0].system_instruction == "Rewrite the spoken audio into a crisp message."
    assert getattr(loaded["sync_configs"][0], "response_mime_type", None) is None
    assert loaded["stream_contents_calls"][1] == ["what is photosynthesis?"]
    assert loaded["stream_configs"][0].thinking_config is None


def test_vertex_gemini_generator_adds_low_thinking_when_enabled(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(
        build_settings(
            credentials_path,
            GEMINI_ENABLE_THINKING="true",
        )
    )

    captured = {}

    def fake_load_credentials_from_file(path, scopes=None):
        return object(), "demo-project"

    class FakeResponse:
        text = "thinking response"

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["config"] = config
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def close(self):
            return None

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    payload = generator.generate_message(b"fake-audio", "audio/webm")

    assert payload == {"message": "thinking response"}
    thinking_config = captured["config"].thinking_config
    assert thinking_config is not None

    if hasattr(thinking_config, "thinking_level"):
        assert thinking_config.thinking_level == "low"
    else:
        assert thinking_config.thinking_budget == generator.LOW_THINKING_BUDGET


def test_vertex_gemini_generator_uses_override_system_instruction(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(build_settings(credentials_path))

    captured = {}

    def fake_load_credentials_from_file(path, scopes=None):
        return object(), "demo-project"

    class FakeResponse:
        text = "override response"

    class FakeModels:
        def generate_content(self, *, model, contents, config):
            captured["config"] = config
            return FakeResponse()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def close(self):
            return None

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    payload = generator.generate_message_from_text(
        "what is photosynthesis?",
        system_instruction_override="Responsible override prompt.",
    )

    assert payload == {"message": "override response"}
    assert captured["config"].system_instruction == "Responsible override prompt."


def test_vertex_gemini_generator_rejects_empty_stream(monkeypatch, tmp_path):
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    generator = VertexGeminiMessageGenerator(build_settings(credentials_path))

    def fake_load_credentials_from_file(path, scopes=None):
        return object(), "demo-project"

    class FakeModels:
        def generate_content_stream(self, *, model, contents, config):
            return iter([type("Chunk", (), {"text": ""})(), type("Chunk", (), {"text": None})()])

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def close(self):
            return None

    monkeypatch.setattr("app.gemini.load_credentials_from_file", fake_load_credentials_from_file)
    monkeypatch.setattr("app.gemini.genai.Client", FakeClient)

    try:
        list(generator.generate_message_stream(b"fake-audio", "audio/webm"))
    except Exception as exc:
        assert exc.message == "Gemini returned an empty response."
        assert exc.detail == "No streamed text chunks were returned by the model."
    else:
        raise AssertionError("Expected generate_message_stream to fail on an empty stream.")
