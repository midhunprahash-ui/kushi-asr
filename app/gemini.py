from __future__ import annotations

from threading import Lock
from typing import Any, Dict, Iterator, Optional

from google import genai
from google.auth import load_credentials_from_file
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as genai_errors
from google.genai import types

from app.config import Settings
from app.speech import GOOGLE_CLOUD_SCOPE


class GeminiGenerationError(Exception):
    def __init__(self, status_code: int, message: str, detail: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail


class VertexGeminiMessageGenerator:
    AUDIO_INPUT_INSTRUCTION = (
        "Treat the attached audio as the user's spoken input. Apply the system instruction to "
        "that spoken content and return only the final answer text. Do not wrap the answer in "
        "JSON."
    )
    LOW_THINKING_BUDGET = 1024

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Optional[genai.Client] = None
        self._credentials: Optional[Any] = None
        self._client_lock = Lock()

    def close(self) -> None:
        client = self._client
        self._client = None
        if client is not None and hasattr(client, "close"):
            client.close()

    def _load_credentials(self) -> Any:
        credentials_path = self.settings.google_application_credentials
        if not credentials_path:
            raise DefaultCredentialsError("GOOGLE_APPLICATION_CREDENTIALS is not configured.")

        credentials, _ = load_credentials_from_file(
            credentials_path,
            scopes=[GOOGLE_CLOUD_SCOPE],
        )
        return credentials

    def _get_credentials(self) -> Any:
        if self._credentials is None:
            self._credentials = self._load_credentials()
        return self._credentials

    def _get_client(self) -> genai.Client:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    self._client = genai.Client(
                        vertexai=True,
                        credentials=self._get_credentials(),
                        project=self.settings.google_cloud_project,
                        location=self.settings.gemini_location,
                        http_options=types.HttpOptions(api_version="v1"),
                    )
        return self._client

    def _build_audio_request_contents(self, audio_content: bytes, mime_type: str) -> list[Any]:
        return [
            self.AUDIO_INPUT_INSTRUCTION,
            types.Part.from_bytes(
                data=audio_content,
                mime_type=mime_type,
            ),
        ]

    def _build_text_request_contents(self, text: str) -> list[Any]:
        return [text]

    def _build_thinking_config(self) -> Optional[types.ThinkingConfig]:
        if not self.settings.gemini_enable_thinking:
            return None

        fields = getattr(types.ThinkingConfig, "model_fields", {})
        if "thinking_level" in fields:
            return types.ThinkingConfig(thinking_level="low")

        # google-genai==1.33.0 exposes thinking_budget instead of thinking_level.
        return types.ThinkingConfig(thinking_budget=self.LOW_THINKING_BUDGET)

    def _build_generation_config(
        self,
        system_instruction_override: Optional[str] = None,
    ) -> types.GenerateContentConfig:
        config: Dict[str, Any] = {}
        if system_instruction_override is not None:
            system_instruction = system_instruction_override.strip()
        else:
            system_instruction = (self.settings.gemini_system_prompt or "").strip()
        if system_instruction:
            config["system_instruction"] = system_instruction

        thinking_config = self._build_thinking_config()
        if thinking_config is not None:
            config["thinking_config"] = thinking_config

        return types.GenerateContentConfig(**config)

    def _normalize_message(self, text: Optional[str]) -> str:
        message = (text or "").strip()
        if not message:
            raise GeminiGenerationError(
                status_code=502,
                message="Gemini returned an empty response.",
                detail="No text payload was returned by the model.",
            )
        return message

    def _generate_message_text(
        self,
        request_contents: list[Any],
        *,
        system_instruction_override: Optional[str] = None,
    ) -> str:
        try:
            response = self._get_client().models.generate_content(
                model=self.settings.gemini_model,
                contents=request_contents,
                config=self._build_generation_config(system_instruction_override),
            )
        except DefaultCredentialsError as exc:
            raise GeminiGenerationError(
                status_code=503,
                message="Google Cloud credentials are missing or invalid.",
                detail=str(exc),
            ) from exc
        except genai_errors.APIError as exc:
            raise GeminiGenerationError(
                status_code=502,
                message="Gemini generateContent request failed.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise GeminiGenerationError(
                status_code=500,
                message="Unexpected Gemini generation failure.",
                detail=str(exc),
            ) from exc

        return self._normalize_message(getattr(response, "text", None))

    def generate_message_text(
        self,
        audio_content: bytes,
        mime_type: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> str:
        return self._generate_message_text(
            self._build_audio_request_contents(audio_content, mime_type),
            system_instruction_override=system_instruction_override,
        )

    def generate_message_text_from_text(
        self,
        text: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> str:
        return self._generate_message_text(
            self._build_text_request_contents(text),
            system_instruction_override=system_instruction_override,
        )

    def generate_message(
        self,
        audio_content: bytes,
        mime_type: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> Dict[str, str]:
        return {
            "message": self.generate_message_text(
                audio_content,
                mime_type,
                system_instruction_override=system_instruction_override,
            )
        }

    def generate_message_from_text(
        self,
        text: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> Dict[str, str]:
        return {
            "message": self.generate_message_text_from_text(
                text,
                system_instruction_override=system_instruction_override,
            )
        }

    def _generate_message_stream(
        self,
        request_contents: list[Any],
        *,
        system_instruction_override: Optional[str] = None,
    ) -> Iterator[str]:
        response_stream = None
        emitted_text = False

        try:
            response_stream = self._get_client().models.generate_content_stream(
                model=self.settings.gemini_model,
                contents=request_contents,
                config=self._build_generation_config(system_instruction_override),
            )
            for chunk in response_stream:
                text = getattr(chunk, "text", None) or ""
                if not text:
                    continue
                emitted_text = True
                yield text
        except DefaultCredentialsError as exc:
            raise GeminiGenerationError(
                status_code=503,
                message="Google Cloud credentials are missing or invalid.",
                detail=str(exc),
            ) from exc
        except genai_errors.APIError as exc:
            raise GeminiGenerationError(
                status_code=502,
                message="Gemini generateContent request failed.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise GeminiGenerationError(
                status_code=500,
                message="Unexpected Gemini generation failure.",
                detail=str(exc),
            ) from exc
        finally:
            if response_stream is not None and hasattr(response_stream, "close"):
                response_stream.close()

        if not emitted_text:
            raise GeminiGenerationError(
                status_code=502,
                message="Gemini returned an empty response.",
                detail="No streamed text chunks were returned by the model.",
            )

    def generate_message_stream(
        self,
        audio_content: bytes,
        mime_type: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> Iterator[str]:
        return self._generate_message_stream(
            self._build_audio_request_contents(audio_content, mime_type),
            system_instruction_override=system_instruction_override,
        )

    def generate_message_stream_from_text(
        self,
        text: str,
        *,
        system_instruction_override: Optional[str] = None,
    ) -> Iterator[str]:
        return self._generate_message_stream(
            self._build_text_request_contents(text),
            system_instruction_override=system_instruction_override,
        )
