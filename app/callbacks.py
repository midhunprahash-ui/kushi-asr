from __future__ import annotations

import httpx


class TranscriptPostError(Exception):
    def __init__(self, message: str, detail: str) -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail


class TranscriptPoster:
    def __init__(self, timeout_seconds: float = 10.0) -> None:
        self.timeout_seconds = timeout_seconds

    async def post_text(self, url: str, text: str) -> None:
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json={"text": text})
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TranscriptPostError(
                message="Transcript POST delivery failed.",
                detail=str(exc),
            ) from exc
