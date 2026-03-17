const API_PREFIX = "/api/asr/v1";
const TRANSCRIPTS_ENDPOINT = `${API_PREFIX}/transcripts`;
const callbackUrlParam = new URLSearchParams(window.location.search).get("callback_url") || "";
const CALLBACK_URL_STORAGE_KEY = "kushi-asr.callback-url";

const callbackUrlInput = document.getElementById("callbackUrlInput");
const talkButton = document.getElementById("talkButton");
const statusNode = document.getElementById("status");
const finalNode = document.getElementById("finalTranscript");
const latencyNode = document.getElementById("latencyValue");
const processingNode = document.getElementById("processingValue");
const recordingNode = document.getElementById("recordingValue");
const speechNode = document.getElementById("speechValue");
const audioSizeNode = document.getElementById("audioSizeValue");
const recognizerNode = document.getElementById("recognizerValue");
const transcriptIdNode = document.getElementById("transcriptIdValue");
const resultUrlNode = document.getElementById("resultUrlValue");
const deliveryStatusNode = document.getElementById("deliveryStatusValue");
const deliveryTargetNode = document.getElementById("deliveryTargetValue");

let mediaStream;
let recorder;
let chunks = [];
let state = "idle";
let stopRequestedDuringStart = false;
let recordingStartedAt = null;

function setStatus(message, state = "idle") {
  statusNode.textContent = message;
  statusNode.className = `status ${state}`;
}

function setMetric(node, value) {
  node.textContent = value;
}

function resetTranscript() {
  finalNode.textContent = "Transcript will appear here.";
  finalNode.classList.add("muted");
}

function resetMetrics() {
  setMetric(latencyNode, "-");
  setMetric(processingNode, "-");
  setMetric(recordingNode, "-");
  setMetric(speechNode, "-");
  setMetric(audioSizeNode, "-");
  setMetric(recognizerNode, "-");
}

function resetResultDetails() {
  setMetric(transcriptIdNode, "-");
  setMetric(resultUrlNode, "-");
  setMetric(deliveryStatusNode, "-");
  setMetric(deliveryTargetNode, "-");
}

function formatMilliseconds(milliseconds) {
  if (!Number.isFinite(milliseconds)) {
    return "-";
  }

  return `${Math.round(milliseconds)} ms`;
}

function formatSeconds(seconds) {
  if (!Number.isFinite(seconds)) {
    return "-";
  }

  const precision = seconds >= 10 ? 1 : 2;
  return `${seconds.toFixed(precision)} s`;
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) {
    return "-";
  }

  if (bytes < 1024) {
    return `${bytes} B`;
  }

  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }

  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatRecognizer(languageCode, model) {
  if (languageCode && model) {
    return `${languageCode} / ${model}`;
  }

  return languageCode || model || "-";
}

function formatDeliveryStatus(status) {
  if (status === "sent") {
    return "POST sent";
  }

  if (status === "failed") {
    return "POST failed";
  }

  return "Skipped";
}

function loadCallbackUrl() {
  if (callbackUrlParam) {
    return callbackUrlParam;
  }

  try {
    return window.localStorage.getItem(CALLBACK_URL_STORAGE_KEY) || "";
  } catch (error) {
    return "";
  }
}

function getCallbackUrl() {
  const value = callbackUrlInput.value.trim();

  try {
    if (value) {
      window.localStorage.setItem(CALLBACK_URL_STORAGE_KEY, value);
    } else {
      window.localStorage.removeItem(CALLBACK_URL_STORAGE_KEY);
    }
  } catch (error) {
    return value;
  }

  return value;
}

function buildResultUrl(transcriptId) {
  if (!transcriptId) {
    return "-";
  }

  return new URL(`${API_PREFIX}/transcripts/${transcriptId}`, window.location.origin).toString();
}

function getSupportedMimeType() {
  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];

  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
}

function setButtonState() {
  talkButton.disabled = state === "uploading";
  talkButton.classList.toggle("is-recording", state === "recording");
  talkButton.textContent = state === "recording" ? "Release to send" : "Hold to talk";
}

async function stopTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

async function beginRecording() {
  if (state !== "idle") {
    return;
  }

  state = "starting";
  stopRequestedDuringStart = false;
  setButtonState();
  setStatus("Waiting for microphone…", "idle");
  resetMetrics();
  resetResultDetails();
  finalNode.textContent = "Listening...";
  finalNode.classList.remove("muted");

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
  } catch (error) {
    state = "idle";
    setButtonState();
    setStatus(error.message || "Unable to access the microphone.", "error");
    finalNode.textContent = error.message || "Unable to access the microphone.";
    finalNode.classList.remove("muted");
    return;
  }

  chunks = [];
  recorder = new MediaRecorder(mediaStream, {
    mimeType: getSupportedMimeType() || undefined,
  });

  recorder.addEventListener("dataavailable", (event) => {
    if (event.data && event.data.size > 0) {
      chunks.push(event.data);
    }
  });

  recorder.start();
  state = "recording";
  recordingStartedAt = performance.now();
  setButtonState();
  setStatus("Recording…", "live");

  if (stopRequestedDuringStart) {
    await finishRecording();
  }
}

async function submitRecording(blob, recordingDurationMs) {
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(audioSizeNode, formatBytes(blob.size));

  const formData = new FormData();
  const fileExtension = blob.type.includes("mp4") ? "m4a" : "webm";
  formData.append("audio", blob, `recording.${fileExtension}`);

  const callbackUrl = getCallbackUrl();
  if (callbackUrl) {
    formData.append("callback_url", callbackUrl);
  }

  const requestStartedAt = performance.now();
  let response;
  try {
    response = await fetch(TRANSCRIPTS_ENDPOINT, {
      method: "POST",
      body: formData,
    });
  } catch (error) {
    setMetric(latencyNode, formatMilliseconds(performance.now() - requestStartedAt));
    throw error;
  }

  setMetric(latencyNode, formatMilliseconds(performance.now() - requestStartedAt));
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.detail?.message || "Transcription request failed.");
  }

  finalNode.textContent = payload.text || "";
  finalNode.classList.toggle("muted", !payload.text);
  setMetric(processingNode, formatMilliseconds(payload.processing_ms));
  setMetric(speechNode, formatSeconds(payload.speech_seconds));
  setMetric(recognizerNode, formatRecognizer(payload.language_code, payload.model));
  setMetric(transcriptIdNode, payload.id || "-");
  setMetric(resultUrlNode, buildResultUrl(payload.id));
  setMetric(deliveryStatusNode, formatDeliveryStatus(payload.delivery_status));
  setMetric(deliveryTargetNode, payload.delivery_target || "-");
  setStatus(payload.text ? "Transcript ready" : "No speech detected", "idle");
}

async function finishRecording() {
  if (state === "starting") {
    stopRequestedDuringStart = true;
    return;
  }

  if (state !== "recording" || !recorder) {
    return;
  }

  const activeRecorder = recorder;
  recorder = null;
  state = "uploading";
  setButtonState();
  setStatus("Uploading audio…", "idle");
  const recordingDurationMs =
    recordingStartedAt === null ? Number.NaN : performance.now() - recordingStartedAt;
  recordingStartedAt = null;

  const blob = await new Promise((resolve) => {
    activeRecorder.addEventListener(
      "stop",
      () => {
        const recordedBlob = new Blob(chunks, { type: activeRecorder.mimeType || "audio/webm" });
        resolve(recordedBlob);
      },
      { once: true },
    );
    activeRecorder.stop();
  });

  await stopTracks();

  try {
    await submitRecording(blob, recordingDurationMs);
  } catch (error) {
    finalNode.textContent = error.message || "Unable to transcribe audio.";
    finalNode.classList.remove("muted");
    setStatus(error.message || "Unable to transcribe audio.", "error");
  } finally {
    chunks = [];
    state = "idle";
    setButtonState();
  }
}

talkButton.addEventListener("pointerdown", async (event) => {
  event.preventDefault();
  talkButton.setPointerCapture(event.pointerId);
  await beginRecording();
});

talkButton.addEventListener("pointerup", async () => {
  await finishRecording();
});

talkButton.addEventListener("pointercancel", async () => {
  await finishRecording();
});

talkButton.addEventListener("lostpointercapture", async () => {
  await finishRecording();
});

talkButton.addEventListener("keydown", async (event) => {
  if ((event.code === "Space" || event.code === "Enter") && !event.repeat) {
    event.preventDefault();
    await beginRecording();
  }
});

talkButton.addEventListener("keyup", async (event) => {
  if (event.code === "Space" || event.code === "Enter") {
    event.preventDefault();
    await finishRecording();
  }
});

window.addEventListener("blur", async () => {
  await finishRecording();
});

callbackUrlInput.value = loadCallbackUrl();
resetTranscript();
resetMetrics();
resetResultDetails();
setButtonState();
