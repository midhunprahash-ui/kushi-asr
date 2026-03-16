# Kushi ASR

Simple push-to-talk ASR microservice using Google Speech-to-Text v2 Chirp 3, FastAPI, and a built-in browser recorder.

## Endpoints

- `GET /api/asr/v1/health`
- `GET /api/asr/v1/player`
- `POST /api/asr/v1/transcript`

`POST /api/asr/v1/transcript` accepts one uploaded audio file and returns only text in the response.

If a callback URL is provided, the service also forwards the transcript as:

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
{"text":"hello world"}
```

Example:

```bash
curl -X POST http://localhost:8000/api/asr/v1/transcript \
  -F "audio=@/absolute/path/to/clip.webm"
```

Callback example:

```bash
curl -X POST http://localhost:8000/api/asr/v1/transcript \
  -F "audio=@/absolute/path/to/clip.webm" \
  -F "callback_url=http://your-other-service:9000/input"
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

## Docker

Build:

```bash
docker build -t kushi-asr .
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
  kushi-asr
```

## Docker Compose

Build and start:

```bash
docker compose up --build
```

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
- The browser player records a short clip and uploads it in one request.
- The backend uses synchronous recognition with format auto-detection, which is suited to short push-to-talk audio.
- Keep clips short; synchronous recognition is intended for brief local files rather than long recordings.
- To auto-forward UI transcripts, either set `ASR_OUTPUT_POST_URL` in `.env` or open the player with `?callback_url=...`.
