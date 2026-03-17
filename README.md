# Kushi ASR

Simple push-to-talk ASR microservice using Google Speech-to-Text v2 Chirp 3, FastAPI, and a built-in browser recorder.

## Endpoints

- `GET /api/asr/v1/health`
- `GET /api/asr/v1/player`
- `POST /api/asr/v1/transcripts`
- `GET /api/asr/v1/transcripts/{transcript_id}`
- `WS /api/asr/v1/stream`
- `POST /api/asr/v1/transcript` (legacy alias)
- `GET /api/asr/v2/player`
- `POST /api/asr/v2/messages`

`POST /api/asr/v1/transcripts` accepts one uploaded audio file, stores the result, returns transcript metadata for the built-in player UI, and can optionally POST `{"text":"..."}` to another service. The browser player prefers `WS /api/asr/v1/stream` for lower-latency hold-to-talk streaming and falls back to the upload endpoint if streaming capture is unavailable.

`POST /api/asr/v2/messages` accepts one uploaded audio file, sends it to Gemini `generateContent` on Vertex AI with the server-side `GEMINI_SYSTEM_PROMPT`, and returns only:

```json
{"message":"..."}
```

If a callback URL is provided, the service forwards the transcript as:

```json
{"text":"hello world"}
```

## Transcript API

Request:

- Content type: `multipart/form-data`
- File field: `audio`
- Optional form field: `language_code`
- Optional form field: `callback_url`

Response:

```json
{
  "id": "6f6d0bc6b8ad42fca2e30efde3d45f19",
  "text": "hello world",
  "language_code": "en-IN",
  "model": "chirp_3",
  "speech_seconds": 1.5,
  "processing_ms": 412,
  "delivery_status": "sent",
  "delivery_target": "https://your-other-service.example/post"
}
```

Example:

```bash
curl -X POST http://localhost:8000/api/asr/v1/transcripts \
  -F "audio=@/absolute/path/to/clip.webm"
```

Callback example:

```bash
curl -X POST http://localhost:8000/api/asr/v1/transcripts \
  -F "audio=@/absolute/path/to/clip.webm" \
  -F "callback_url=http://your-other-service:9000/input"
```

Fetch stored transcript by ID:

```bash
curl http://localhost:8000/api/asr/v1/transcripts/6f6d0bc6b8ad42fca2e30efde3d45f19
```

GET response:

```json
{"text":"hello world"}
```

## Message API

Request:

- Content type: `multipart/form-data`
- File field: `audio`

Response:

```json
{"message":"cleaned user message"}
```

Example:

```bash
curl -X POST http://localhost:8000/api/asr/v2/messages \
  -F "audio=@/absolute/path/to/clip.webm"
```

## Local Run

1. Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
./.venv/bin/pip install -r requirements-dev.txt
```

2. Set environment variables:

```bash
cp .env.example .env
```

3. Start the app:

```bash
export APP_PORT=8000
./.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "${APP_PORT}"
```

4. Open:

```text
http://localhost:8000/api/asr/v1/player
```

Gemini player:

```text
http://localhost:8000/api/asr/v2/player
```

## Docker

Build:

```bash
docker build -t kushi-asr:v2 .
```

Run:

```bash
docker run --rm \
  -p 8000:8000 \
  -e APP_PORT=8000 \
  -e GOOGLE_CLOUD_PROJECT=your-project-id \
  -e GOOGLE_APPLICATION_CREDENTIALS=/app/credentials/service-account.json \
  -e GCP_SPEECH_LOCATION=us \
  -e GCP_SPEECH_RECOGNIZER=_ \
  -v /absolute/path/to/service-account.json:/app/credentials/service-account.json:ro \
  kushi-asr:v2
```

Open the player at:

```text
http://localhost:8000/api/asr/v1/player
```

## Docker Compose

Build and start:

```bash
docker compose up --build
```

Compose expects these in `.env`:

- `GOOGLE_CLOUD_PROJECT`
- `GCP_SPEECH_LOCATION`
- `GCP_SPEECH_RECOGNIZER`
- `GOOGLE_APPLICATION_CREDENTIALS_HOST_PATH`
- `GEMINI_LOCATION`
- `GEMINI_MODEL`
- `GEMINI_SYSTEM_PROMPT`

The mounted host credential file is exposed inside the container as `/app/credentials/service-account.json`.

Run detached:

```bash
docker compose up --build -d
```

Stop:

```bash
docker compose down
```

## Notes

- Chirp 3 is configured with `ASR_MODEL=chirp_3`.
- The browser player streams PCM audio to FastAPI over WebSocket while you hold the button, then falls back to upload-on-release when streaming capture is unavailable.
- FastAPI uses Google Speech-to-Text v2 over the Python gRPC transport. Live hold-to-talk uses bidirectional `StreamingRecognize`; the upload fallback uses unary `Recognize`.
- The v2 message flow records from the mic, uploads the final clip on button release, and calls Gemini `generateContent` on Vertex AI with the server-side `GEMINI_SYSTEM_PROMPT`.
- `GEMINI_SYSTEM_PROMPT` is intentionally empty in `.env.example`; set it in `.env` to control how spoken audio is rewritten before the JSON message response is returned.
- Keep clips short; synchronous recognition is intended for brief local files rather than long recordings.
- To auto-forward UI transcripts, either set `ASR_OUTPUT_POST_URL` in `.env` or fill the callback URL field in the player UI.
- If `ASR_OUTPUT_BEARER_TOKEN` is set, outbound callback requests include `Authorization: Bearer <token>`.
- Stored transcripts expire after `ASR_RESULT_TTL_SECONDS` and are kept in memory only.
