// pcm-worklet-processor.js — the "PCM Converter" box in the diagram.
//
// Runs on the audio rendering thread. Converts incoming Float32 samples
// (range -1..1) to 16-bit signed PCM (linear16), which is what Deepgram's
// streaming API expects, then posts the raw bytes back to the main thread.

class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;

    const channelData = input[0]; // mono
    if (!channelData || channelData.length === 0) return true;

    const pcm16 = new Int16Array(channelData.length);
    for (let i = 0; i < channelData.length; i++) {
      const s = Math.max(-1, Math.min(1, channelData[i]));
      pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }

    this.port.postMessage(pcm16.buffer, [pcm16.buffer]);
    return true;
  }
}

registerProcessor('pcm-worklet-processor', PCMWorkletProcessor);