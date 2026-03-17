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

    async def post_text(
        self,
        url: str,
        text: str,
        bearer_token: str | None = None,
    ) -> None:
        headers = {}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(url, json={"text": text}, headers=headers)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise TranscriptPostError(
                message="Transcript POST delivery failed.",
                detail=str(exc),
            ) from exc
