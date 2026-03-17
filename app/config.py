from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "kushi-asr"
    app_host: str = Field(default="0.0.0.0", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    api_prefix: str = "/api/asr/v1"

    google_cloud_project: Optional[str] = Field(default=None, alias="GOOGLE_CLOUD_PROJECT")
    google_application_credentials: Optional[str] = Field(
        default=None,
        alias="GOOGLE_APPLICATION_CREDENTIALS",
    )
    gcp_speech_location: str = Field(default="us", alias="GCP_SPEECH_LOCATION")
    gcp_speech_recognizer: str = Field(default="_", alias="GCP_SPEECH_RECOGNIZER")

    asr_language_code: str = Field(default="en-IN", alias="ASR_LANGUAGE_CODE")
    asr_model: str = Field(default="chirp_3", alias="ASR_MODEL")
    asr_max_upload_bytes: int = Field(default=10_485_760, alias="ASR_MAX_UPLOAD_BYTES")
    asr_session_ttl_seconds: int = Field(default=180, alias="ASR_SESSION_TTL_SECONDS")
    asr_target_sample_rate_hz: int = Field(default=16_000, alias="ASR_TARGET_SAMPLE_RATE_HZ")
    asr_enable_interim_results: bool = Field(default=True, alias="ASR_ENABLE_INTERIM_RESULTS")
    asr_output_post_url: Optional[str] = Field(default=None, alias="ASR_OUTPUT_POST_URL")
    asr_output_bearer_token: Optional[str] = Field(
        default=None,
        alias="ASR_OUTPUT_BEARER_TOKEN",
    )
    asr_output_post_timeout_seconds: float = Field(
        default=10.0,
        alias="ASR_OUTPUT_POST_TIMEOUT_SECONDS",
    )
    asr_result_ttl_seconds: int = Field(default=900, alias="ASR_RESULT_TTL_SECONDS")

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    @property
    def recognizer_path(self) -> str:
        return (
            f"projects/{self.google_cloud_project}/locations/"
            f"{self.gcp_speech_location}/recognizers/{self.gcp_speech_recognizer}"
        )

    @property
    def speech_api_endpoint(self) -> str:
        if self.gcp_speech_location == "global":
            return "speech.googleapis.com"
        return f"{self.gcp_speech_location}-speech.googleapis.com"

    def missing_google_env(self) -> List[str]:
        missing: List[str] = []

        if not self.google_cloud_project:
            missing.append("GOOGLE_CLOUD_PROJECT")

        credentials_path = self.google_application_credentials
        if not credentials_path:
            missing.append("GOOGLE_APPLICATION_CREDENTIALS")
        elif not Path(credentials_path).exists():
            missing.append(f"GOOGLE_APPLICATION_CREDENTIALS:{credentials_path}")

        return missing


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
