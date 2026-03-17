const API_PREFIX = "/api/asr/v2";
const MESSAGES_ENDPOINT = `${API_PREFIX}/messages`;

const talkButton = document.getElementById("talkButton");
const statusNode = document.getElementById("status");
const messageOutputNode = document.getElementById("messageOutput");
const latencyNode = document.getElementById("latencyValue");
const recordingNode = document.getElementById("recordingValue");
const audioSizeNode = document.getElementById("audioSizeValue");
const mimeTypeNode = document.getElementById("mimeTypeValue");
const endpointNode = document.getElementById("endpointValue");
const promptSourceNode = document.getElementById("promptSourceValue");
const responseShapeNode = document.getElementById("responseShapeValue");

let mediaStream;
let recorder;
let chunks = [];
let state = "idle";
let stopRequestedDuringStart = false;
let recordingStartedAt = null;
let recordingDurationMs = Number.NaN;

function setStatus(message, nextState = "idle") {
  statusNode.textContent = message;
  statusNode.className = `status ${nextState}`;
}

function setMetric(node, value) {
  node.textContent = value;
}

function setButtonState() {
  talkButton.disabled = state === "uploading";
  talkButton.classList.toggle("is-recording", state === "recording");
  talkButton.textContent = state === "recording" ? "Release to send" : "Hold to talk";
}

function getSupportedMimeType() {
  if (!window.MediaRecorder) {
    return "";
  }

  const candidates = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
  ];

  return candidates.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) || "";
}

function formatMilliseconds(milliseconds) {
  if (!Number.isFinite(milliseconds)) {
    return "-";
  }

  return `${Math.round(milliseconds)} ms`;
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

function resetMetrics() {
  setMetric(latencyNode, "-");
  setMetric(recordingNode, "-");
  setMetric(audioSizeNode, "-");
  setMetric(mimeTypeNode, "-");
}

function resetOutput() {
  messageOutputNode.textContent = '{\n  "message": "Awaiting input..."\n}';
  messageOutputNode.classList.add("muted");
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
  resetOutput();

  try {
    mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

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
    recordingStartedAt = performance.now();
    state = "recording";
    setButtonState();
    setStatus("Recording…", "live");
    messageOutputNode.textContent = '{\n  "message": "Listening..."\n}';
    messageOutputNode.classList.remove("muted");
  } catch (error) {
    state = "idle";
    setButtonState();
    setStatus(error.message || "Unable to access the microphone.", "error");
    messageOutputNode.textContent = error.message || "Unable to access the microphone.";
    messageOutputNode.classList.remove("muted");
    return;
  }

  if (stopRequestedDuringStart) {
    await finishRecording();
  }
}

async function submitRecording(blob) {
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(audioSizeNode, formatBytes(blob.size));
  setMetric(mimeTypeNode, blob.type || "audio/webm");

  const formData = new FormData();
  const fileExtension = blob.type.includes("mp4") ? "m4a" : "webm";
  formData.append("audio", blob, `recording.${fileExtension}`);

  const requestStartedAt = performance.now();
  let response;
  try {
    response = await fetch(MESSAGES_ENDPOINT, {
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
    throw new Error(payload?.detail?.message || "Message generation failed.");
  }

  messageOutputNode.textContent = JSON.stringify(payload, null, 2);
  messageOutputNode.classList.remove("muted");
  setStatus(payload.message ? "Message ready" : "No output returned", "idle");
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
  setStatus("Sending audio to Gemini…", "idle");
  recordingDurationMs =
    recordingStartedAt === null ? Number.NaN : performance.now() - recordingStartedAt;

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
    await submitRecording(blob);
  } catch (error) {
    messageOutputNode.textContent = error.message || "Unable to generate a message.";
    messageOutputNode.classList.remove("muted");
    setStatus(error.message || "Unable to generate a message.", "error");
  } finally {
    chunks = [];
    recordingStartedAt = null;
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

setMetric(endpointNode, new URL(MESSAGES_ENDPOINT, window.location.origin).toString());
setMetric(promptSourceNode, "GEMINI_SYSTEM_PROMPT (.env)");
setMetric(responseShapeNode, '{"message":"..."}');
resetMetrics();
resetOutput();
setButtonState();

if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
  talkButton.disabled = true;
  setStatus("Media recording is not supported in this browser.", "error");
  messageOutputNode.textContent = "Media recording is not supported in this browser.";
  messageOutputNode.classList.remove("muted");
}
