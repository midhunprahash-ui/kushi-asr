from __future__ import annotations

from datetime import timedelta
from typing import Any, Dict, List, Optional

from google.api_core.client_options import ClientOptions
from google.api_core.exceptions import GoogleAPICallError, RetryError
from google.auth import load_credentials_from_file
from google.auth.exceptions import DefaultCredentialsError
from google.cloud import speech_v2
from google.cloud.speech_v2.types import cloud_speech

from app.config import Settings

GOOGLE_CLOUD_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def duration_to_seconds(duration: Any) -> Optional[float]:
    if duration is None:
        return None
    if isinstance(duration, timedelta):
        return duration.total_seconds()
    if hasattr(duration, "total_seconds"):
        return float(duration.total_seconds())
    return float(duration.seconds) + (float(duration.nanos) / 1_000_000_000)


class SpeechRecognitionError(Exception):
    def __init__(self, status_code: int, message: str, detail: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.detail = detail


class GoogleSpeechRecognizer:
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

    def transcribe(self, audio_content: bytes, language_code: str) -> Dict[str, Any]:
        client: Optional[speech_v2.SpeechClient] = None

        try:
            client = speech_v2.SpeechClient(
                credentials=self._load_credentials(),
                client_options=ClientOptions(api_endpoint=self.settings.speech_api_endpoint),
            )
            request = cloud_speech.RecognizeRequest(
                recognizer=self.settings.recognizer_path,
                config=cloud_speech.RecognitionConfig(
                    auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
                    language_codes=[language_code],
                    model=self.settings.asr_model,
                    features=cloud_speech.RecognitionFeatures(
                        enable_automatic_punctuation=True,
                    ),
                ),
                content=audio_content,
            )

            response = client.recognize(request=request)
            transcript_parts: List[str] = []
            detected_language = language_code
            segments: List[Dict[str, Any]] = []

            for result in response.results:
                if not result.alternatives:
                    continue

                alternative = result.alternatives[0]
                transcript = alternative.transcript.strip()
                if transcript:
                    transcript_parts.append(transcript)

                if result.language_code:
                    detected_language = result.language_code

                segments.append(
                    {
                        "text": transcript,
                        "confidence": float(alternative.confidence),
                        "language_code": result.language_code or detected_language,
                        "end_offset_seconds": duration_to_seconds(result.result_end_offset),
                    }
                )

            speech_seconds = max(
                (
                    segment["end_offset_seconds"]
                    for segment in segments
                    if segment["end_offset_seconds"] is not None
                ),
                default=None,
            )

            return {
                "text": " ".join(part for part in transcript_parts if part).strip(),
                "language_code": detected_language,
                "model": self.settings.asr_model,
                "segments": segments,
                "speech_seconds": speech_seconds,
            }
        except DefaultCredentialsError as exc:
            raise SpeechRecognitionError(
                status_code=503,
                message="Google Cloud credentials are missing or invalid.",
                detail=str(exc),
            ) from exc
        except (GoogleAPICallError, RetryError) as exc:
            raise SpeechRecognitionError(
                status_code=502,
                message="Google Speech-to-Text request failed.",
                detail=str(exc),
            ) from exc
        except Exception as exc:
            raise SpeechRecognitionError(
                status_code=500,
                message="Unexpected ASR failure.",
                detail=str(exc),
            ) from exc
        finally:
            if client is not None and hasattr(client.transport, "close"):
                client.transport.close()
