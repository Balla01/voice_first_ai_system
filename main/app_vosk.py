import argparse
import json
import gradio as gr
import numpy as np
from vosk import Model, KaldiRecognizer

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="vosk-model-small-en-us-0.15",
    help="Vosk model name (auto-download) or full path to a local model folder",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load Vosk model
# ---------------------------------------------------------------------------
import os
if os.path.isdir(args.model):
    _vosk_model = Model(model_path=args.model)
    print(f"[vosk] loaded from path: {args.model}")
else:
    _vosk_model = Model(model_name=args.model)
    print(f"[vosk] loaded model: {args.model}")



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def to_mono_int16(data, orig_sr):
    """Convert any incoming audio chunk to 16kHz mono int16 bytes for Vosk."""
    data = np.array(data)
    if data.ndim > 1:
        data = data.mean(axis=1)

    # If float32 (already normalized), scale back to int16 range
    if data.dtype in (np.float32, np.float64):
        data = (data * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        data = data.astype(np.int16)

    # Resample to 16kHz
    if orig_sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / orig_sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n),
            np.arange(len(data)),
            data.astype(np.float32),
        ).astype(np.int16)

    return data.tobytes()


# ---------------------------------------------------------------------------
# Gradio streaming callback
# ---------------------------------------------------------------------------
def new_recognizer():
    rec = KaldiRecognizer(_vosk_model, SAMPLE_RATE)
    rec.SetWords(True)
    return rec


def process(audio, state):
    if audio is None:
        return state, state["display"] if state else ""

    if state is None:
        state = {"rec": new_recognizer(), "final": "", "display": ""}

    sr, data = audio
    raw = to_mono_int16(data, sr)

    try:
        accepted = state["rec"].AcceptWaveform(raw)
    except Exception:
        # Decoder hit a bad state — reset and skip this chunk
        state["rec"] = new_recognizer()
        return state, state["display"]

    if accepted:
        text = json.loads(state["rec"].Result()).get("text", "").strip()
        if text:
            state["final"] = (state["final"] + " " + text).strip()
        state["display"] = state["final"]
    else:
        partial = json.loads(state["rec"].PartialResult()).get("partial", "").strip()
        state["display"] = state["final"]
        if partial:
            state["display"] += f" [{partial}]"

    return state, state["display"]


def clear():
    return None, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Live Speech to Text — Vosk") as demo:
    gr.Markdown(
        f"# Live Speech to Text — Vosk\n"
        f"`model: {args.model}` — confirmed words appear normally, words being heard appear in `[brackets]`"
    )

    with gr.Row():
        with gr.Column():
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            clear_btn = gr.Button("Clear", variant="secondary")
        with gr.Column():
            output = gr.Textbox(label="Transcription", lines=10, interactive=False)

    state = gr.State(None)
    mic.stream(process, inputs=[mic, state], outputs=[state, output])
    clear_btn.click(clear, outputs=[state, output])

demo.launch()
