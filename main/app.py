import argparse
import gradio as gr
import numpy as np

SAMPLE_RATE = 16000
CHUNK_SECONDS = 5

# ---------------------------------------------------------------------------
# CLI flags
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend",
    choices=["whisper", "faster_whisper"],
    default="faster_whisper",
    help="whisper = openai-whisper  |  faster_whisper = faster-whisper (int8, faster on CPU)",
)
parser.add_argument(
    "--model",
    default="small",
    help="Model size: tiny | base | small | medium | large-v2 | large-v3",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Load model based on chosen backend
# ---------------------------------------------------------------------------
if args.backend == "faster_whisper":
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from faster_whisper import WhisperModel

    # Download without symlinks — avoids WinError 1314 on Windows
    local_model_dir = Path("models") / f"faster-whisper-{args.model}"
    if not local_model_dir.exists():
        print(f"Downloading faster-whisper-{args.model} ...")
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{args.model}",
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,
        )

    _model = WhisperModel(str(local_model_dir), device="cpu", compute_type="int8")
    BACKEND = "faster_whisper"
    print(f"[faster-whisper] model={args.model}  device=cpu  compute=int8")
else:
    import whisper
    _model = whisper.load_model(args.model)
    BACKEND = "whisper"
    print(f"[openai-whisper] model={args.model}")


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------
def transcribe_chunk(chunk):
    if BACKEND == "faster_whisper":
        segments, _ = _model.transcribe(chunk, language="en", beam_size=5)
        return " ".join(s.text.strip() for s in segments)
    else:
        result = _model.transcribe(chunk, language="en", fp16=False)
        return result["text"].strip()


def resample(data, orig_sr):
    if orig_sr == SAMPLE_RATE:
        return data
    n = int(len(data) * SAMPLE_RATE / orig_sr)
    return np.interp(
        np.linspace(0, len(data) - 1, n), np.arange(len(data)), data
    ).astype(np.float32)


def has_speech(data, threshold=0.01):
    return np.sqrt(np.mean(data ** 2)) > threshold


# ---------------------------------------------------------------------------
# Gradio streaming callback
# ---------------------------------------------------------------------------
def process(audio, state):
    if audio is None:
        return state, state["text"] if state else ""

    if state is None:
        state = {"buffer": np.array([], dtype=np.float32), "text": ""}

    sr, data = audio
    data = data.astype(np.float32)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data /= 32768.0
    data = resample(data, sr)

    state["buffer"] = np.concatenate([state["buffer"], data])

    if len(state["buffer"]) < SAMPLE_RATE * CHUNK_SECONDS:
        return state, state["text"]

    chunk = state["buffer"]
    state["buffer"] = np.array([], dtype=np.float32)

    if not has_speech(chunk):
        return state, state["text"]

    segment = transcribe_chunk(chunk)
    if segment:
        state["text"] = (state["text"] + " " + segment).strip()

    return state, state["text"]


def clear():
    return None, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
label = f"Backend: {args.backend}  |  Model: {args.model}"

with gr.Blocks(title="Live Speech to Text") as demo:
    gr.Markdown(f"# Live Speech to Text\n`{label}` — transcription appends every ~{CHUNK_SECONDS}s of speech.")

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
