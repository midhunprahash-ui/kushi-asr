const API_PREFIX = "/api/asr/v1";
const callbackUrl = new URLSearchParams(window.location.search).get("callback_url");

const talkButton = document.getElementById("talkButton");
const statusNode = document.getElementById("status");
const finalNode = document.getElementById("finalTranscript");

let mediaStream;
let recorder;
let chunks = [];
let state = "idle";
let stopRequestedDuringStart = false;

function setStatus(message, state = "idle") {
  statusNode.textContent = message;
  statusNode.className = `status ${state}`;
}

function resetTranscript() {
  finalNode.textContent = "Transcript will appear here.";
  finalNode.classList.add("muted");
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
  setButtonState();
  setStatus("Recording…", "live");

  if (stopRequestedDuringStart) {
    await finishRecording();
  }
}

async function submitRecording(blob) {
  const formData = new FormData();
  const fileExtension = blob.type.includes("mp4") ? "m4a" : "webm";
  formData.append("audio", blob, `recording.${fileExtension}`);
  if (callbackUrl) {
    formData.append("callback_url", callbackUrl);
  }

  const response = await fetch(`${API_PREFIX}/transcript`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload?.detail?.message || "Transcription request failed.");
  }

  finalNode.textContent = payload.text || "";
  finalNode.classList.toggle("muted", !payload.text);
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

resetTranscript();
setButtonState();
