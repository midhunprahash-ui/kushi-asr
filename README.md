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

## API Reference

### `GET /`

Use:
- Opens the app entrypoint in the browser.
- Redirects to the `v2` player page.

Typical use:
- Open `http://localhost:8000/` instead of remembering the full player path.

### `GET /api/asr/v2/health`

Use:
- Health and readiness check for the current app.
- Confirms the service is up and reports the active recognizer/model configuration.

Typical use:
- Docker health checks
- deployment checks
- quick debugging to confirm credentials/config are loaded

### `GET /api/asr/v2/player`

Use:
- Serves the built-in push-to-talk browser UI.
- This is the main UI for speaking, transcribing, and getting the final generated response.

Typical use:
- Open in Chrome/Edge and hold the talk button to speak.

### `WS /api/asr/v2/stream`

Use:
- Live speech-to-text websocket endpoint.
- The browser sends microphone audio chunks here while the user is still speaking.
- The server sends transcript events back while the utterance is in progress.

Typical use:
- Low-latency transcript capture for the built-in `v2` UI.
- Better than waiting for a full audio upload before transcription starts.

Event flow:

Client sends:
```json
{"type":"start","sample_rate_hz":48000}
```

Then binary audio chunks, then:
```json
{"type":"stop"}
```

Server sends events like:
```json
{"type":"session","session_id":"..."}
{"type":"chunk","received_chunks":1}
{"type":"partial","text":"what is"}
{"type":"final","text":"what is photosynthesis"}
{"type":"completed","text":"what is photosynthesis","id":"..."}
```

Purpose in the current app:
- This is the first stage of the low-latency `v2` flow.
- It captures the user’s spoken words before Gemini generation starts.

### `POST /api/asr/v2/messages`

Use:
- One-shot audio-to-answer API.
- Upload a recorded audio file directly to Gemini `generateContent`.

Request:
- Content type: `multipart/form-data`
- File field: `audio`

Response:
```json
{"message":"..."}
```

Typical use:
- Server-to-server or Postman testing
- simple one-request audio input without incremental output

### `POST /api/asr/v2/messages/stream`

Use:
- Streaming version of the direct audio-to-Gemini route.
- Upload one audio file, then receive the generated text incrementally as it is produced.

Request:
- Content type: `multipart/form-data`
- File field: `audio`

Response format:
- `application/x-ndjson`
- newline-delimited JSON events

Events:
```json
{"type":"partial","text":"draft answer"}
{"type":"partial","text":"draft answer with more text"}
{"type":"completed","message":"draft answer with more text"}
```

Typical use:
- progressive output for uploaded audio
- fallback path when transcript-first flow is not used

### `POST /api/asr/v2/messages/text/stream`

Use:
- Transcript-to-answer streaming route.
- Accepts plain text and streams Gemini output back incrementally.
- This is the main low-latency generation path used by the current `v2` UI after speech has already been transcribed.

Request:
```json
{"text":"what is photosynthesis"}
```

Response format:
- `application/x-ndjson`
- newline-delimited JSON events

Events:
```json
{"type":"partial","text":"Photosynthesis "}
{"type":"partial","text":"Photosynthesis is how plants make food."}
{"type":"completed","message":"Photosynthesis is how plants make food."}
```

Typical use:
- lowest-latency answer generation in the current app
- external integrations that already have text input

Responsible AI behavior:
- When the moderation client is configured, this route classifies the final user transcript first.
- If the transcript is flagged, the matching responsible prompt from [ai_moderation_service.py](/Users/midhun/Developer/kushi-asr/ai_moderation_service.py) is appended to the system instruction before Gemini generates the answer.

## Current v2 Flow

The built-in `v2` UI works in two stages:

1. While the user is speaking, the browser streams PCM audio to `WS /api/asr/v2/stream`.
2. When the user releases the button, the final transcript is sent to `POST /api/asr/v2/messages/text/stream`.
3. Gemini generates the answer text and the UI renders it as:

```json
{"message":"..."}
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
