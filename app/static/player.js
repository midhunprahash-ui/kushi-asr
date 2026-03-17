const API_PREFIX = "/api/asr/v1";
const TRANSCRIPTS_ENDPOINT = `${API_PREFIX}/transcripts`;
const STREAM_ENDPOINT = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${API_PREFIX}/stream`;
const callbackUrlParam = new URLSearchParams(window.location.search).get("callback_url") || "";
const CALLBACK_URL_STORAGE_KEY = "kushi-asr.callback-url";

const callbackUrlInput = document.getElementById("callbackUrlInput");
const talkButton = document.getElementById("talkButton");
const statusNode = document.getElementById("status");
const finalNode = document.getElementById("finalTranscript");
const chunksSentNode = document.getElementById("chunksSentValue");
const chunksReceivedNode = document.getElementById("chunksReceivedValue");
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
let recordingMode = null;
let stopRequestedDuringStart = false;
let recordingStartedAt = null;
let recordingDurationMs = Number.NaN;

let audioContext;
let sourceNode;
let workletNode;
let silenceNode;
let streamingSocket;
let streamedBytes = 0;
let streamedChunkCount = 0;
let receivedChunkCount = 0;
let finalizationStartedAt = null;
let finalTranscriptSegments = [];
let interimTranscript = "";
let streamingCompletionResolve;
let streamingCompletionReject;
let streamingCompletionTimer;
let streamingCompleted = false;

function setStatus(message, nextState = "idle") {
  statusNode.textContent = message;
  statusNode.className = `status ${nextState}`;
}

function setMetric(node, value) {
  node.textContent = value;
}

function resetTranscript() {
  finalNode.textContent = "Transcript will appear here.";
  finalNode.classList.add("muted");
}

function resetMetrics() {
  setMetric(chunksSentNode, "-");
  setMetric(chunksReceivedNode, "-");
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

function supportsStreamingCapture() {
  return Boolean(
    navigator.mediaDevices?.getUserMedia &&
      window.WebSocket &&
      (window.AudioContext || window.webkitAudioContext) &&
      window.AudioWorkletNode,
  );
}

async function stopTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
}

function renderStreamingTranscript() {
  const text = [...finalTranscriptSegments, interimTranscript].filter(Boolean).join(" ").trim();
  finalNode.textContent = text || "Listening...";
  finalNode.classList.remove("muted");
}

function convertFloat32ToPcm16(float32Array) {
  const buffer = new ArrayBuffer(float32Array.length * 2);
  const view = new DataView(buffer);

  for (let index = 0; index < float32Array.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, float32Array[index]));
    const int16 = sample < 0 ? sample * 0x8000 : sample * 0x7fff;
    view.setInt16(index * 2, int16, true);
  }

  return buffer;
}

function cleanupStreamingPromise() {
  if (streamingCompletionTimer) {
    window.clearTimeout(streamingCompletionTimer);
    streamingCompletionTimer = null;
  }

  streamingCompletionResolve = undefined;
  streamingCompletionReject = undefined;
}

function rejectStreamingCompletion(error) {
  if (streamingCompletionReject) {
    const reject = streamingCompletionReject;
    cleanupStreamingPromise();
    reject(error);
  }
}

function resolveStreamingCompletion(payload) {
  if (streamingCompletionResolve) {
    const resolve = streamingCompletionResolve;
    cleanupStreamingPromise();
    resolve(payload);
  }
}

function closeStreamingSocket() {
  if (!streamingSocket) {
    return;
  }

  streamingSocket.onmessage = null;
  streamingSocket.onerror = null;
  streamingSocket.onclose = null;

  if (
    streamingSocket.readyState === WebSocket.OPEN ||
    streamingSocket.readyState === WebSocket.CONNECTING
  ) {
    streamingSocket.close();
  }

  streamingSocket = null;
}

async function cleanupStreamingAudio() {
  if (workletNode) {
    workletNode.port.onmessage = null;
    try {
      workletNode.disconnect();
    } catch (error) {
      // Ignore teardown errors.
    }
    workletNode = null;
  }

  if (sourceNode) {
    try {
      sourceNode.disconnect();
    } catch (error) {
      // Ignore teardown errors.
    }
    sourceNode = null;
  }

  if (silenceNode) {
    try {
      silenceNode.disconnect();
    } catch (error) {
      // Ignore teardown errors.
    }
    silenceNode = null;
  }

  if (audioContext) {
    const activeContext = audioContext;
    audioContext = null;
    await activeContext.close().catch(() => {});
  }

  await stopTracks();
}

async function resetStreamingState() {
  await cleanupStreamingAudio();
  closeStreamingSocket();
  cleanupStreamingPromise();
  streamedBytes = 0;
  streamedChunkCount = 0;
  receivedChunkCount = 0;
  finalizationStartedAt = null;
  finalTranscriptSegments = [];
  interimTranscript = "";
  streamingCompleted = false;
}

function applyCompletedPayload(payload) {
  finalNode.textContent = payload.text || "";
  finalNode.classList.toggle("muted", !payload.text);
  setMetric(chunksSentNode, String(streamedChunkCount));
  setMetric(chunksReceivedNode, String(receivedChunkCount));
  setMetric(latencyNode, formatMilliseconds(performance.now() - finalizationStartedAt));
  setMetric(processingNode, formatMilliseconds(payload.processing_ms));
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(speechNode, formatSeconds(payload.speech_seconds));
  setMetric(audioSizeNode, formatBytes(streamedBytes));
  setMetric(recognizerNode, formatRecognizer(payload.language_code, payload.model));
  setMetric(transcriptIdNode, payload.id || "-");
  setMetric(resultUrlNode, buildResultUrl(payload.id));
  setMetric(deliveryStatusNode, formatDeliveryStatus(payload.delivery_status));
  setMetric(deliveryTargetNode, payload.delivery_target || "-");
}

function handleStreamingMessage(event) {
  if (typeof event.data !== "string") {
    return;
  }

  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch (error) {
    return;
  }

  switch (payload.type) {
    case "session":
      receivedChunkCount = 0;
      setMetric(chunksReceivedNode, "0");
      return;
    case "chunk":
      if (Number.isFinite(payload.received_chunks)) {
        receivedChunkCount = payload.received_chunks;
        setMetric(chunksReceivedNode, String(receivedChunkCount));
      }
      return;
    case "partial":
      interimTranscript = payload.text || "";
      renderStreamingTranscript();
      return;
    case "final":
      if (payload.text) {
        finalTranscriptSegments.push(payload.text);
      }
      interimTranscript = "";
      renderStreamingTranscript();
      return;
    case "completed":
      streamingCompleted = true;
      applyCompletedPayload(payload);
      setStatus(payload.text ? "Transcript ready" : "No speech detected", "idle");
      state = "idle";
      recordingMode = null;
      setButtonState();
      resolveStreamingCompletion(payload);
      return;
    case "error": {
      const message = payload.message || "Streaming transcription failed.";
      finalNode.textContent = message;
      finalNode.classList.remove("muted");
      setStatus(message, "error");
      state = "idle";
      recordingMode = null;
      setButtonState();
      rejectStreamingCompletion(new Error(message));
      return;
    }
    default:
      return;
  }
}

function handleStreamingClose() {
  if (!streamingCompleted) {
    rejectStreamingCompletion(new Error("Streaming connection closed unexpectedly."));
  }

  closeStreamingSocket();
}

function createStreamingCompletionPromise() {
  return new Promise((resolve, reject) => {
    streamingCompletionResolve = resolve;
    streamingCompletionReject = reject;
    streamingCompletionTimer = window.setTimeout(() => {
      rejectStreamingCompletion(new Error("Timed out waiting for the final transcript."));
    }, 15000);
  });
}

async function beginStreamingRecording() {
  const callbackUrl = getCallbackUrl();
  const AudioContextConstructor = window.AudioContext || window.webkitAudioContext;

  mediaStream = await navigator.mediaDevices.getUserMedia({
    audio: {
      channelCount: 1,
      echoCancellation: true,
      noiseSuppression: true,
      autoGainControl: true,
    },
  });

  audioContext = new AudioContextConstructor({ latencyHint: "interactive" });
  await audioContext.audioWorklet.addModule("/static/player-worklet.js");
  if (audioContext.state === "suspended") {
    await audioContext.resume();
  }

  finalTranscriptSegments = [];
  interimTranscript = "";
  streamedBytes = 0;
  streamedChunkCount = 0;
  receivedChunkCount = 0;
  finalizationStartedAt = null;
  streamingCompleted = false;
  setMetric(chunksSentNode, "0");
  setMetric(chunksReceivedNode, "0");

  const socket = await new Promise((resolve, reject) => {
    const nextSocket = new WebSocket(STREAM_ENDPOINT);
    nextSocket.binaryType = "arraybuffer";

    nextSocket.addEventListener("open", () => resolve(nextSocket), { once: true });
    nextSocket.addEventListener(
      "error",
      () => reject(new Error("Unable to open the streaming connection.")),
      { once: true },
    );
  });

  streamingSocket = socket;
  streamingSocket.onmessage = handleStreamingMessage;
  streamingSocket.onerror = () => {
    if (state !== "idle") {
      rejectStreamingCompletion(new Error("Streaming connection failed."));
    }
  };
  streamingSocket.onclose = handleStreamingClose;

  streamingSocket.send(
    JSON.stringify({
      type: "start",
      language_code: "en-IN",
      callback_url: callbackUrl || null,
      sample_rate_hz: Math.round(audioContext.sampleRate),
    }),
  );

  sourceNode = audioContext.createMediaStreamSource(mediaStream);
  workletNode = new AudioWorkletNode(audioContext, "pcm-capture-processor");
  silenceNode = audioContext.createGain();
  silenceNode.gain.value = 0;

  workletNode.port.onmessage = ({ data }) => {
    if (!(data instanceof Float32Array)) {
      return;
    }

    if (!streamingSocket || streamingSocket.readyState !== WebSocket.OPEN) {
      return;
    }

    const pcmBuffer = convertFloat32ToPcm16(data);
    streamedBytes += pcmBuffer.byteLength;
    streamedChunkCount += 1;
    setMetric(chunksSentNode, String(streamedChunkCount));
    streamingSocket.send(pcmBuffer);
  };

  sourceNode.connect(workletNode);
  workletNode.connect(silenceNode);
  silenceNode.connect(audioContext.destination);

  recordingMode = "streaming";
  recordingDurationMs = Number.NaN;
  state = "recording";
  recordingStartedAt = performance.now();
  setButtonState();
  setStatus("Streaming…", "live");
}

async function finishStreamingRecording() {
  if (!streamingSocket || streamingSocket.readyState !== WebSocket.OPEN) {
    throw new Error("Streaming connection is not open.");
  }

  recordingDurationMs =
    recordingStartedAt === null ? Number.NaN : performance.now() - recordingStartedAt;
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(audioSizeNode, formatBytes(streamedBytes));
  finalizationStartedAt = performance.now();

  const completionPromise = createStreamingCompletionPromise();
  state = "uploading";
  setButtonState();
  setStatus("Finalizing transcript…", "idle");

  if (workletNode) {
    workletNode.port.postMessage({ type: "flush" });
  }

  await new Promise((resolve) => {
    window.setTimeout(resolve, 50);
  });

  await cleanupStreamingAudio();
  streamingSocket.send(JSON.stringify({ type: "stop" }));

  try {
    await completionPromise;
  } finally {
    recordingStartedAt = null;
    await resetStreamingState();
  }
}

async function beginUploadRecording() {
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
  recordingMode = "upload";
  recordingDurationMs = Number.NaN;
  state = "recording";
  recordingStartedAt = performance.now();
  setButtonState();
  setStatus("Recording…", "live");
}

async function submitRecording(blob) {
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

async function finishUploadRecording() {
  if (!recorder) {
    return;
  }

  const activeRecorder = recorder;
  recorder = null;
  state = "uploading";
  setButtonState();
  setStatus("Uploading audio…", "idle");
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
  } finally {
    chunks = [];
    recordingStartedAt = null;
    recordingMode = null;
    state = "idle";
    setButtonState();
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
  resetTranscript();
  finalNode.textContent = "Listening...";
  finalNode.classList.remove("muted");

  try {
    if (supportsStreamingCapture()) {
      await beginStreamingRecording();
    } else {
      await beginUploadRecording();
    }
  } catch (error) {
    await resetStreamingState();

    try {
      await beginUploadRecording();
    } catch (fallbackError) {
      state = "idle";
      recordingMode = null;
      setButtonState();
      setStatus(fallbackError.message || "Unable to access the microphone.", "error");
      finalNode.textContent = fallbackError.message || "Unable to access the microphone.";
      finalNode.classList.remove("muted");
      return;
    }
  }

  if (stopRequestedDuringStart) {
    await finishRecording();
  }
}

async function finishRecording() {
  if (state === "starting") {
    stopRequestedDuringStart = true;
    return;
  }

  if (state !== "recording") {
    return;
  }

  try {
    if (recordingMode === "streaming") {
      await finishStreamingRecording();
      return;
    }

    await finishUploadRecording();
  } catch (error) {
    finalNode.textContent = error.message || "Unable to transcribe audio.";
    finalNode.classList.remove("muted");
    setStatus(error.message || "Unable to transcribe audio.", "error");
    state = "idle";
    recordingMode = null;
    recordingStartedAt = null;
    setButtonState();
    await resetStreamingState();
    await stopTracks();
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
