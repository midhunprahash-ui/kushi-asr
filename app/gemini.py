from __future__ import annotations

import json
from typing import Any, Dict

from google import genai
from google.auth import load_credentials_from_file
from google.auth.exceptions import DefaultCredentialsError
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import ValidationError

from app.config import Settings
from app.models import MessageResponse
from app.speech import GOOGLE_CLOUD_SCOPE


class GeminiGenerationError(Exception):
    def __init__(self, status_code: int, message: str, detail: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail


class VertexGeminiMessageGenerator:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def _load_credentials(self) -> Any:
        credentials_path = self.settings.google_application_credentials
        if not credentials_path:
            raise DefaultCredentialsError("GOOGLE_APPLICATION_CREDENTIALS is not configured.")

        credentials, _ = load_credentials_from_file(
            credentials_path,
            scopes=[GOOGLE_CLOUD_SCOPE],
        )
        return credentials

    def generate_message(self, audio_content: bytes, mime_type: str) -> Dict[str, str]:
        request_contents = [
            (
                "Use the attached audio as the user's input. Apply the system instruction to that "
                "spoken content and return only the final text output in the JSON field `message`."
            ),
            types.Part.from_bytes(
                data=audio_content,
                mime_type=mime_type,
            ),
        ]

        client = None
        try:
            client = genai.Client(
                vertexai=True,
                credentials=self._load_credentials(),
                project=self.settings.google_cloud_project,
                location=self.settings.gemini_location,
                http_options=types.HttpOptions(api_version="v1"),
            )
            response = client.models.generate_content(
                model=self.settings.gemini_model,
                contents=request_contents,
                config=types.GenerateContentConfig(
                    system_instruction=self.settings.gemini_system_prompt,
                    response_mime_type="application/json",
                    response_json_schema=MessageResponse.model_json_schema(),
                )
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
        finally:
            if client is not None and hasattr(client, "close"):
                client.close()

        payload = response.parsed
        if payload is None:
            response_text = (response.text or "").strip()
            if not response_text:
                raise GeminiGenerationError(
                    status_code=502,
                    message="Gemini returned an empty response.",
                    detail="No JSON payload was returned by the model.",
                )
            try:
                payload = json.loads(response_text)
            except json.JSONDecodeError as exc:
                raise GeminiGenerationError(
                    status_code=502,
                    message="Gemini returned an invalid JSON response.",
                    detail=str(exc),
                ) from exc

        try:
            message = MessageResponse.model_validate(payload)
        except ValidationError as exc:
            raise GeminiGenerationError(
                status_code=502,
                message="Gemini returned an unexpected response shape.",
                detail=str(exc),
            ) from exc

        return message.model_dump()
