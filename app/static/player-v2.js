const API_PREFIX = "/api/asr/v2";
const AUDIO_MESSAGES_STREAM_ENDPOINT = `${API_PREFIX}/messages/stream`;
const TEXT_MESSAGES_STREAM_ENDPOINT = `${API_PREFIX}/messages/text/stream`;
const TRANSCRIPT_STREAM_ENDPOINT = `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}${API_PREFIX}/stream`;

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
let recordingMode = null;
let stopRequestedDuringStart = false;
let recordingStartedAt = null;
let recordingDurationMs = Number.NaN;
let finishInFlight = false;

let audioContext;
let sourceNode;
let workletNode;
let silenceNode;
let streamingSocket;
let streamedBytes = 0;
let transcriptSegments = [];
let interimTranscript = "";
let streamingCompleted = false;
let transcriptCompletionResolve;
let transcriptCompletionReject;
let transcriptCompletionTimer;

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

function supportsStreamingCapture() {
  return Boolean(
    navigator.mediaDevices?.getUserMedia &&
      window.WebSocket &&
      (window.AudioContext || window.webkitAudioContext) &&
      window.AudioWorkletNode,
  );
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

function renderStreamingText(text) {
  messageOutputNode.textContent = text || "";
  messageOutputNode.classList.remove("muted");
}

function renderFinalMessage(message) {
  messageOutputNode.textContent = JSON.stringify({ message }, null, 2);
  messageOutputNode.classList.remove("muted");
}

function resetOutput() {
  renderFinalMessage("Awaiting input...");
  messageOutputNode.classList.add("muted");
}

async function stopTracks() {
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
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

function cleanupTranscriptPromise() {
  if (transcriptCompletionTimer) {
    window.clearTimeout(transcriptCompletionTimer);
    transcriptCompletionTimer = null;
  }

  transcriptCompletionResolve = undefined;
  transcriptCompletionReject = undefined;
}

function rejectTranscriptCompletion(error) {
  if (!transcriptCompletionReject) {
    return;
  }

  const reject = transcriptCompletionReject;
  cleanupTranscriptPromise();
  reject(error);
}

function resolveTranscriptCompletion(payload) {
  if (!transcriptCompletionResolve) {
    return;
  }

  const resolve = transcriptCompletionResolve;
  cleanupTranscriptPromise();
  resolve(payload);
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
  cleanupTranscriptPromise();
  streamedBytes = 0;
  transcriptSegments = [];
  interimTranscript = "";
  streamingCompleted = false;
}

function handleTranscriptMessage(event) {
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
      return;
    case "partial":
      interimTranscript = payload.text || "";
      return;
    case "final":
      if (payload.text) {
        transcriptSegments.push(payload.text);
      }
      interimTranscript = "";
      return;
    case "completed":
      streamingCompleted = true;
      resolveTranscriptCompletion(payload);
      return;
    case "error":
      rejectTranscriptCompletion(
        new Error(payload.message || "Streaming transcription failed.")
      );
      return;
    default:
      return;
  }
}

function handleTranscriptClose() {
  if (!streamingCompleted) {
    rejectTranscriptCompletion(new Error("Streaming transcription closed unexpectedly."));
  }

  closeStreamingSocket();
}

function createTranscriptCompletionPromise() {
  return new Promise((resolve, reject) => {
    transcriptCompletionResolve = resolve;
    transcriptCompletionReject = reject;
    transcriptCompletionTimer = window.setTimeout(() => {
      rejectTranscriptCompletion(new Error("Timed out waiting for the final transcript."));
    }, 15000);
  });
}

async function beginStreamingRecording() {
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

  streamedBytes = 0;
  transcriptSegments = [];
  interimTranscript = "";
  streamingCompleted = false;
  setMetric(mimeTypeNode, "live/pcm16");

  const socket = await new Promise((resolve, reject) => {
    const nextSocket = new WebSocket(TRANSCRIPT_STREAM_ENDPOINT);
    nextSocket.binaryType = "arraybuffer";

    nextSocket.addEventListener("open", () => resolve(nextSocket), { once: true });
    nextSocket.addEventListener(
      "error",
      () => reject(new Error("Unable to open the live transcription connection.")),
      { once: true },
    );
  });

  streamingSocket = socket;
  streamingSocket.onmessage = handleTranscriptMessage;
  streamingSocket.onerror = () => {
    if (state !== "idle") {
      rejectTranscriptCompletion(new Error("Live transcription connection failed."));
    }
  };
  streamingSocket.onclose = handleTranscriptClose;

  streamingSocket.send(
    JSON.stringify({
      type: "start",
      language_code: "en-IN",
      callback_url: null,
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
    streamingSocket.send(pcmBuffer);
  };

  sourceNode.connect(workletNode);
  workletNode.connect(silenceNode);
  silenceNode.connect(audioContext.destination);

  recordingMode = "streaming";
  state = "recording";
  recordingStartedAt = performance.now();
  setButtonState();
  setStatus("Transcribing while you speak…", "live");
  renderFinalMessage("Listening...");
}

async function finishStreamingRecording() {
  if (!streamingSocket || streamingSocket.readyState !== WebSocket.OPEN) {
    throw new Error("Live transcription connection is not open.");
  }

  recordingDurationMs =
    recordingStartedAt === null ? Number.NaN : performance.now() - recordingStartedAt;
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(audioSizeNode, formatBytes(streamedBytes));

  const completionPromise = createTranscriptCompletionPromise();
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

  await new Promise((resolve) => {
    window.setTimeout(resolve, 150);
  });

  const transcriptSnapshot = [...transcriptSegments, interimTranscript].filter(Boolean).join(" ").trim();
  if (transcriptSnapshot) {
    cleanupTranscriptPromise();
    await resetStreamingState();
    return transcriptSnapshot;
  }

  try {
    const payload = await completionPromise;
    const transcript = (payload?.text || "").trim();
    if (!transcript) {
      throw new Error("No speech was detected.");
    }
    return transcript;
  } finally {
    await resetStreamingState();
  }
}

function decodeNdjsonEvents(buffer) {
  const events = [];
  let nextBuffer = buffer;
  let newlineIndex = nextBuffer.indexOf("\n");

  while (newlineIndex !== -1) {
    const rawLine = nextBuffer.slice(0, newlineIndex).trim();
    nextBuffer = nextBuffer.slice(newlineIndex + 1);

    if (rawLine) {
      events.push(JSON.parse(rawLine));
    }

    newlineIndex = nextBuffer.indexOf("\n");
  }

  return { events, buffer: nextBuffer };
}

async function streamMessageResponse(url, options) {
  const requestStartedAt = performance.now();
  let response;
  try {
    response = await fetch(url, options);
  } catch (error) {
    throw error;
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    throw new Error(payload?.detail?.message || "Message generation failed.");
  }

  if (!response.body) {
    throw new Error("Streaming responses are not supported in this browser.");
  }

  setStatus("Generating answer…", "live");
  renderStreamingText("Generating...");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalMessage = "";
  let firstResponseRecorded = false;

  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });

    const parsed = decodeNdjsonEvents(buffer);
    buffer = parsed.buffer;

    for (const event of parsed.events) {
      if (!firstResponseRecorded && (event.type === "partial" || event.type === "completed")) {
        setMetric(latencyNode, formatMilliseconds(performance.now() - requestStartedAt));
        firstResponseRecorded = true;
      }

      if (event.type === "partial") {
        renderStreamingText(event.text || "");
        continue;
      }

      if (event.type === "completed") {
        finalMessage = event.message || "";
        renderFinalMessage(finalMessage);
        setStatus(finalMessage ? "Message ready" : "No output returned", "idle");
        continue;
      }

      if (event.type === "error") {
        throw new Error(event.message || "Message generation failed.");
      }
    }

    if (done) {
      break;
    }
  }

  buffer += decoder.decode();
  const trailing = decodeNdjsonEvents(buffer);
  for (const event of trailing.events) {
    if (!firstResponseRecorded && (event.type === "partial" || event.type === "completed")) {
      setMetric(latencyNode, formatMilliseconds(performance.now() - requestStartedAt));
      firstResponseRecorded = true;
    }

    if (event.type === "completed") {
      finalMessage = event.message || "";
      renderFinalMessage(finalMessage);
      setStatus(finalMessage ? "Message ready" : "No output returned", "idle");
      continue;
    }

    if (event.type === "partial") {
      renderStreamingText(event.text || "");
      continue;
    }

    if (event.type === "error") {
      throw new Error(event.message || "Message generation failed.");
    }
  }

  if (!finalMessage) {
    throw new Error("Message stream ended before a final response was received.");
  }
}

async function submitTranscript(text) {
  await streamMessageResponse(TEXT_MESSAGES_STREAM_ENDPOINT, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text }),
  });
}

async function submitRecording(blob) {
  setMetric(recordingNode, formatMilliseconds(recordingDurationMs));
  setMetric(audioSizeNode, formatBytes(blob.size));
  setMetric(mimeTypeNode, blob.type || "audio/webm");

  const formData = new FormData();
  const fileExtension = blob.type.includes("mp4") ? "m4a" : "webm";
  formData.append("audio", blob, `recording.${fileExtension}`);

  await streamMessageResponse(AUDIO_MESSAGES_STREAM_ENDPOINT, {
    method: "POST",
    body: formData,
  });
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
  state = "recording";
  recordingStartedAt = performance.now();
  setButtonState();
  setStatus("Recording…", "live");
  renderFinalMessage("Listening...");
}

async function finishUploadRecording() {
  if (!recorder) {
    return;
  }

  const activeRecorder = recorder;
  recorder = null;
  state = "uploading";
  setButtonState();
  setStatus("Uploading audio to Gemini…", "idle");
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
  chunks = [];

  await submitRecording(blob);
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
    if (supportsStreamingCapture()) {
      await beginStreamingRecording();
    } else {
      await beginUploadRecording();
    }
  } catch (error) {
    await resetStreamingState();
    await stopTracks();
    state = "idle";
    recordingMode = null;
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

async function finishRecording() {
  if (finishInFlight) {
    return;
  }

  if (state === "starting") {
    stopRequestedDuringStart = true;
    return;
  }

  if (state !== "recording" && !(state === "uploading" && recordingMode === "streaming")) {
    return;
  }

  finishInFlight = true;
  try {
    if (recordingMode === "streaming") {
      const transcript = await finishStreamingRecording();
      await submitTranscript(transcript);
    } else if (recordingMode === "upload") {
      await finishUploadRecording();
    }
  } catch (error) {
    messageOutputNode.textContent = error.message || "Unable to generate a message.";
    messageOutputNode.classList.remove("muted");
    setStatus(error.message || "Unable to generate a message.", "error");
  } finally {
    chunks = [];
    recordingMode = null;
    recordingStartedAt = null;
    state = "idle";
    setButtonState();
    finishInFlight = false;
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

setMetric(endpointNode, new URL(TEXT_MESSAGES_STREAM_ENDPOINT, window.location.origin).toString());
setMetric(
  promptSourceNode,
  "GEMINI_SYSTEM_PROMPT (.env) + AI moderation prompts when configured + live transcript",
);
setMetric(responseShapeNode, 'application/x-ndjson -> {"message":"..."}');
resetMetrics();
resetOutput();
setButtonState();

if (!navigator.mediaDevices?.getUserMedia) {
  talkButton.disabled = true;
  setStatus("Media recording is not supported in this browser.", "error");
  messageOutputNode.textContent = "Media recording is not supported in this browser.";
  messageOutputNode.classList.remove("muted");
}
