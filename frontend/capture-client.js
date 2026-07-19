// capture-client.js — implements the top row of the diagram:
//   Microphone -> \
//                   PCM Converter -> WebSocket client (+session_id) -> backend
//   System Audio -> /
//
// Two independent pipelines run in parallel (mic, system), each with its own
// AudioWorklet + WebSocket connection, tagged with the same session_id but
// a different stream_id — matching "WSS | stream_id: mic/system | session_id header".
//
// CaptureController exposes each leg (mic / system) as an independently
// startable/stoppable pipeline so the UI can toggle either one on its own —
// denying the system-audio share picker (or the mic prompt) only takes down
// that leg, not the whole call.

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
  constructor(stream, streamId, sessionId, onServerMessage) {
    this.stream = stream;
    this.streamId = streamId; // "mic" | "system"
    this.sessionId = sessionId;
    this.onServerMessage = onServerMessage; // (data) => void, parsed JSON from the backend
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

    // The backend pushes RAG suggestions back as JSON text frames (currently
    // only over the "mic" stream's socket — see auido_capture/main.py's
    // _send_to_session). Audio frames we SEND are binary/ArrayBuffer; this
    // handles the separate JSON messages the server sends back to us.
    this.socket.onmessage = (event) => {
      if (typeof event.data !== 'string') return; // ignore any non-JSON/binary frames
      if (!this.onServerMessage) return;
      try {
        this.onServerMessage(JSON.parse(event.data));
      } catch (err) {
        console.error(`[${this.streamId}] Failed to parse server message`, err, event.data);
      }
    };

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

/**
 * Owns the mic ("agent") and system-audio ("customer") legs of a single call
 * as two independently startable/stoppable pipelines sharing one session_id.
 *
 * @param {(data: object) => void} [onServerMessage] - called with each JSON
 *   message the backend pushes back over the mic socket:
 *     {"type": "transcript", "speaker": "agent"|"customer", "text": string}
 *     {"type": "suggestion", "query": string, "answer": string, "tool": string}
 */
export class CaptureController {
  constructor(onServerMessage) {
    this.sessionId = makeSessionId();
    this.onServerMessage = onServerMessage;
    this._mic = null;
    this._system = null;
  }

  get micOn() { return this._mic !== null; }
  get systemOn() { return this._system !== null; }
  get anyOn() { return this.micOn || this.systemOn; }

  async startMic() {
    if (this._mic) return;
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: SAMPLE_RATE },
    });
    const pipeline = new AudioStreamPipeline(stream, 'mic', this.sessionId, this.onServerMessage);
    await pipeline.start();
    this._mic = pipeline;
  }

  async startSystem() {
    if (this._system) return;
    // System / tab audio -> customer side (e.g. phone call audio routed
    // through the system, or the customer's leg of a softphone call).
    // Requires the user to pick "share tab/window audio" in the browser's
    // picker.
    const stream = await navigator.mediaDevices.getDisplayMedia({
      video: true, // Chrome requires video:true for getDisplayMedia even if unused
      audio: true,
    });

    // getDisplayMedia's audio:true is only a REQUEST -- the browser silently
    // returns zero audio tracks if the user picked "Entire Screen"/a window
    // instead of a Chrome Tab, or didn't check "Share tab audio" in the
    // picker. Fail here with a clear, actionable message instead of the
    // cryptic "MediaStream has no audio track" error createMediaStreamSource
    // throws.
    if (stream.getAudioTracks().length === 0) {
      stream.getTracks().forEach((t) => t.stop());
      throw new Error(
        'No system audio track was shared. In the share picker, choose the ' +
        '"Chrome Tab" option (not Entire Screen/Window) and check "Share tab audio", then try again.'
      );
    }

    const pipeline = new AudioStreamPipeline(stream, 'system', this.sessionId, this.onServerMessage);
    await pipeline.start();
    this._system = pipeline;
  }

  stopMic() {
    this._mic?.stop();
    this._mic = null;
  }

  stopSystem() {
    this._system?.stop();
    this._system = null;
  }

  stopAll() {
    this.stopMic();
    this.stopSystem();
  }
}

/**
 * Joins a session whose "system" (customer) side is already being fed by
 * something other than browser tab-audio-share — e.g. a live Twilio phone
 * call forked to /ws/twilio (see backend/main/auido_capture/twilio_voice.py
 * and main.py's ws_twilio handler). Only starts the mic ("agent") pipeline;
 * sessionId must match the call's session_id (its Twilio CallSid) so both
 * sides land in the same AudioRouter.
 *
 * @param {string} sessionId - the call's session_id (CallSid), from the
 *   join URL logged/returned by /twilio/voice or /twilio/call.
 * @param {(data: object) => void} [onServerMessage] - see CaptureController.
 */
export async function joinCall(sessionId, onServerMessage) {
  if (!sessionId) throw new Error('joinCall requires a sessionId (the call\'s CallSid)');

  const micStream = await navigator.mediaDevices.getUserMedia({
    audio: { channelCount: 1, sampleRate: SAMPLE_RATE },
  });

  const micPipeline = new AudioStreamPipeline(micStream, 'mic', sessionId, onServerMessage);
  await micPipeline.start();

  console.log('Joined call, session', sessionId);
  return { sessionId, micPipeline, systemPipeline: null };
}
