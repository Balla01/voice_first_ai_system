// capture-client.js — implements the top row of the diagram:
//   Microphone -> \
//                   PCM Converter -> WebSocket client (+session_id) -> backend
//   System Audio -> /
//
// Two independent pipelines run in parallel (mic, system), each with its own
// AudioWorklet + WebSocket connection, tagged with the same session_id but
// a different stream_id — matching "WSS | stream_id: mic/system | session_id header".

const BACKEND_WS_BASE = 'ws://localhost:8000/ws/audio'; // use wss:// only once backend is served over TLS
const SAMPLE_RATE = 16000;

// The audio worklet posts a message roughly every ~128 samples (~8ms at
// 16kHz) — sending each of those as its own WebSocket frame is ~125
// messages/sec per channel and was causing choppy/incomplete audio in
// testing. Instead we buffer incoming chunks and flush one combined
// message every FLUSH_INTERVAL_MS, which is both smoother and still far
// under Deepgram's timeout window.
const FLUSH_INTERVAL_MS = 100;

function makeSessionId() {
  return crypto.randomUUID();
}

function concatInt16Buffers(buffers) {
  let totalLength = 0;
  for (const b of buffers) totalLength += b.byteLength;
  const merged = new Uint8Array(totalLength);
  let offset = 0;
  for (const b of buffers) {
    merged.set(new Uint8Array(b), offset);
    offset += b.byteLength;
  }
  return merged.buffer;
}

class AudioStreamPipeline {
  constructor(stream, streamId, sessionId) {
    this.stream = stream;
    this.streamId = streamId; // "mic" | "system"
    this.sessionId = sessionId;
    this.audioContext = null;
    this.workletNode = null;
    this.socket = null;
    this._pendingChunks = [];
    this._flushTimer = null;
  }

  async start() {
    this.audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });
    await this.audioContext.audioWorklet.addModule('pcm-worklet-processor.js');

    const source = this.audioContext.createMediaStreamSource(this.stream);
    this.workletNode = new AudioWorkletNode(this.audioContext, 'pcm-worklet-processor');

    const url = `${BACKEND_WS_BASE}?stream_id=${this.streamId}&session_id=${this.sessionId}`;
    this.socket = new WebSocket(url);
    this.socket.binaryType = 'arraybuffer';

    this.socket.onopen = () => console.log(`[${this.streamId}] WS connected`);
    this.socket.onerror = (err) => console.error(`[${this.streamId}] WS error`, err);
    this.socket.onclose = (evt) => console.log(`[${this.streamId}] WS closed`, evt.code, evt.reason);

    this.workletNode.port.onmessage = (event) => {
      this._pendingChunks.push(event.data); // just buffer, don't send yet
    };

    this._flushTimer = setInterval(() => this._flush(), FLUSH_INTERVAL_MS);

    source.connect(this.workletNode);
    // Not connecting workletNode to audioContext.destination — we only need
    // to *capture* audio here, not play it back through the speakers.
  }

  _flush() {
    if (this._pendingChunks.length === 0) return;
    if (this.socket.readyState !== WebSocket.OPEN) return;

    const combined = concatInt16Buffers(this._pendingChunks);
    this._pendingChunks = [];
    this.socket.send(combined);
  }

  stop() {
    if (this._flushTimer) clearInterval(this._flushTimer);
    this._flush(); // send any remaining buffered audio before closing
    this.workletNode?.disconnect();
    this.audioContext?.close();
    this.socket?.close();
  }
}

export async function startCapture() {
  const sessionId = makeSessionId();

  // Microphone -> agent side
  const micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: SAMPLE_RATE },
  });

  // System / tab audio -> customer side (e.g. phone call audio routed through
  // the system, or the customer's leg of a softphone call). Requires the user
  // to pick "share tab/window audio" in the browser's picker.
  const systemStream = await navigator.mediaDevices.getDisplayMedia({
    video: true, // Chrome requires video:true for getDisplayMedia even if unused
    audio: true,
  });

  const micPipeline = new AudioStreamPipeline(micStream, 'mic', sessionId);
  const systemPipeline = new AudioStreamPipeline(systemStream, 'system', sessionId);

  await micPipeline.start();
  await systemPipeline.start();

  console.log('Capture started for session', sessionId);
  return { sessionId, micPipeline, systemPipeline };
}