class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.pending = [];
    this.pendingFrames = 0;
    this.flushSize = 4096;
    this.port.onmessage = (event) => {
      if (event.data?.type === "flush") {
        this.flush();
      }
    };
  }

  flush() {
    if (this.pendingFrames === 0) {
      return;
    }

    const combined = new Float32Array(this.pendingFrames);
    let offset = 0;
    for (const chunk of this.pending) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }

    this.port.postMessage(combined, [combined.buffer]);
    this.pending = [];
    this.pendingFrames = 0;
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0] || input[0].length === 0) {
      return true;
    }

    const copy = new Float32Array(input[0]);
    this.pending.push(copy);
    this.pendingFrames += copy.length;

    if (this.pendingFrames >= this.flushSize) {
      this.flush();
    }

    return true;
  }
}

registerProcessor("pcm-capture-processor", PcmCaptureProcessor);
