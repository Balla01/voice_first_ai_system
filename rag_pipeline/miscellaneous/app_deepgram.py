import queue
import threading
import gradio as gr
import numpy as np
from deepgram import DeepgramClient
from deepgram.core.events import EventType

SAMPLE_RATE = 16000


# ---------------------------------------------------------------------------
# Load API key from .env
# ---------------------------------------------------------------------------
def _load_env(path=".env"):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

client = DeepgramClient(api_key=_load_env()["deep_gram_key"])
print("[deepgram] client ready  model=nova-3  streaming=live")


# ---------------------------------------------------------------------------
# Audio helper
# ---------------------------------------------------------------------------
def to_mono_int16(data, orig_sr):
    data = np.array(data)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.dtype in (np.float32, np.float64):
        data = (data * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        data = data.astype(np.int16)
    if orig_sr != SAMPLE_RATE:
        n = int(len(data) * SAMPLE_RATE / orig_sr)
        data = np.interp(
            np.linspace(0, len(data) - 1, n),
            np.arange(len(data)),
            data.astype(np.float32),
        ).astype(np.int16)
    return data


# ---------------------------------------------------------------------------
# Open a Deepgram live WebSocket connection
# ---------------------------------------------------------------------------
def open_connection():
    q_final = queue.Queue()
    q_partial = queue.Queue()

    # Use __enter__ to keep connection alive across Gradio callbacks
    conn_cm = client.listen.v1.connect(
        model="nova-3",
        encoding="linear16",
        sample_rate=SAMPLE_RATE,
        language="en",
        interim_results="true",   # must be string literal, not bool
        smart_format="true",
    )
    conn = conn_cm.__enter__()

    def on_message(msg):
        try:
            # v1 message structure: msg.channel.alternatives[0].transcript
            transcript = msg.channel.alternatives[0].transcript
            if not transcript:
                return
            if getattr(msg, "is_final", False):
                q_final.put(transcript)
            else:
                while not q_partial.empty():
                    try: q_partial.get_nowait()
                    except queue.Empty: break
                q_partial.put(transcript)
        except Exception:
            pass

    conn.on(EventType.MESSAGE, on_message)
    threading.Thread(target=conn.start_listening, daemon=True).start()

    return conn, conn_cm, q_final, q_partial


# ---------------------------------------------------------------------------
# Gradio streaming callback
# ---------------------------------------------------------------------------
def process(audio, state):
    if audio is None:
        return state, state["display"] if state else ""

    if state is None:
        conn, conn_cm, q_final, q_partial = open_connection()
        state = {
            "conn": conn, "conn_cm": conn_cm,
            "q_final": q_final, "q_partial": q_partial,
            "text": "", "display": "",
        }

    sr, data = audio
    state["conn"].send_media(to_mono_int16(data, sr).tobytes())

    # Drain confirmed finals
    while not state["q_final"].empty():
        try:
            text = state["q_final"].get_nowait()
            if text:
                state["text"] = (state["text"] + " " + text).strip()
        except queue.Empty:
            break

    # Latest partial for live preview
    partial = ""
    while not state["q_partial"].empty():
        try: partial = state["q_partial"].get_nowait()
        except queue.Empty: break

    state["display"] = state["text"]
    if partial:
        state["display"] += f" [{partial}]"

    return state, state["display"]


def clear(state):
    if state and state.get("conn_cm"):
        try:
            state["conn_cm"].__exit__(None, None, None)
        except Exception:
            pass
    return None, ""


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="Live Speech to Text — Deepgram") as demo:
    gr.Markdown(
        "# Live Speech to Text — Deepgram\n"
        "`nova-3 | en | live streaming` — confirmed words appear normally, "
        "words being heard appear in `[brackets]`"
    )

    with gr.Row():
        with gr.Column():
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            clear_btn = gr.Button("Clear", variant="secondary")
        with gr.Column():
            output = gr.Textbox(label="Transcription", lines=10, interactive=False)

    state = gr.State(None)
    mic.stream(process, inputs=[mic, state], outputs=[state, output])
    clear_btn.click(clear, inputs=[state], outputs=[state, output])

demo.launch()
