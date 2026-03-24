# Kushi ASR

Single-version `v2` push-to-talk voice assistant built with FastAPI, Google Speech-to-Text v2, Gemini `generateContent`, and a built-in browser recorder.

## Endpoints

- `GET /`
- `GET /api/asr/v2/health`
- `GET /api/asr/v2/player`
- `WS /api/asr/v2/stream`
- `POST /api/asr/v2/messages`
- `POST /api/asr/v2/messages/stream`
- `POST /api/asr/v2/messages/text/stream`

`GET /` redirects to the `v2` player.

`POST /api/asr/v2/messages` accepts one uploaded audio file, sends it to Gemini `generateContent` on Vertex AI with the server-side `GEMINI_SYSTEM_PROMPT`, and returns only:

```json
{"message":"..."}
```

`POST /api/asr/v2/messages/stream` accepts the same upload but streams Gemini text back as newline-delimited JSON events while the answer is being generated. The built-in v2 player uses this route so you see partial output before the final JSON lands.

`POST /api/asr/v2/messages/text/stream` accepts a final transcript string and streams Gemini text back as newline-delimited JSON events. The built-in v2 player uses `WS /api/asr/v2/stream` while you hold the button, then calls this text route on release for lower answer latency.

When the optional AI moderation service is configured, the v2 text route classifies the final transcript first and appends the matching responsible system prompt before Gemini generates the answer.

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

Streaming example:

```bash
curl -N -X POST http://localhost:8000/api/asr/v2/messages/stream \
  -F "audio=@/absolute/path/to/clip.webm"
```

Streaming response events:

```json
{"type":"partial","text":"draft answer"}
{"type":"partial","text":"draft answer with more text"}
{"type":"completed","message":"draft answer with more text"}
```

Text-stream example:

```bash
curl -N -X POST http://localhost:8000/api/asr/v2/messages/text/stream \
  -H 'Content-Type: application/json' \
  -d '{"text":"what is photosynthesis"}'
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
http://localhost:8000/
```

## Docker

Build:

```bash
docker build -t kushi-asr:latest .
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
  kushi-asr:latest
```

Open the player at:

```text
http://localhost:8000/
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
- `GEMINI_ENABLE_THINKING`
- `AI_MODERATION_API_KEY`
- `AI_MODERATION_BASE_URL`
- `AI_MODERATION_MODEL`
- `AI_MODERATION_TIMEOUT_SECONDS`

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

- Root `/` redirects to the `v2` player.
- `GET /api/asr/v2/health` is the primary health endpoint.
- The browser player streams PCM audio to FastAPI over `WS /api/asr/v2/stream` while you hold the button, then falls back to upload-on-release when streaming capture is unavailable.
- FastAPI uses Google Speech-to-Text v2 over the Python gRPC transport. Live hold-to-talk uses bidirectional `StreamingRecognize`; the upload fallback uses unary `Recognize`.
- The v2 player transcribes while you speak, then sends the final transcript to Gemini on release and streams the answer back as `{"message":"..."}`.
- `POST /api/asr/v2/messages/stream` still exists for audio-to-Gemini fallback, but the built-in v2 UI prefers the lower-latency transcript-to-Gemini path.
- If the AI moderation client is configured, v2 uses [ai_moderation_service.py](/Users/midhun/Developer/kushi-asr/ai_moderation_service.py) to classify the final transcript and apply the matching responsible prompt before generating the answer.
- `GEMINI_MODEL` defaults to `gemini-2.5-flash-lite` for lower-latency v2 generation.
- `GEMINI_SYSTEM_PROMPT` is intentionally empty in `.env.example`; set it in `.env` to control how spoken audio is rewritten before the JSON message response is returned.
- Set `GEMINI_ENABLE_THINKING=true` to enable low-thinking mode for v2. Leave it `false` for the lowest latency.
- Keep clips short; synchronous recognition is intended for brief local files rather than long recordings.
- To auto-forward UI transcripts, either set `ASR_OUTPUT_POST_URL` in `.env` or fill the callback URL field in the player UI.
- If `ASR_OUTPUT_BEARER_TOKEN` is set, outbound callback requests include `Authorization: Bearer <token>`.
- Stored transcripts expire after `ASR_RESULT_TTL_SECONDS` and are kept in memory only.
