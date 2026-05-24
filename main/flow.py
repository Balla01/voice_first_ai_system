"""
Real-time Insurance Sales AI Copilot
STT : Deepgram nova-3 (live WebSocket)
RAG : gte-large-en-v1.5  +  Qdrant (local)
LLM : Groq llama-3.1-8b-instant
UI  : Gradio
"""
import time
import queue
import logging
import threading
from datetime import datetime
from pathlib import Path

import gradio as gr
import numpy as np
from dotenv import load_dotenv
from deepgram import DeepgramClient
from deepgram.core.events import EventType
from groq import Groq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# ── Config ────────────────────────────────────────────────────────────────────
SAMPLE_RATE  = 16000
LLM_INTERVAL = 5                                          # seconds between AI cycles
QDRANT_PATH  = r"C:\projects\audio_transition\main\embed"
COLLECTION   = "insurance_docs"
TOP_K        = 3
LLM_MODEL    = "llama-3.1-8b-instant"
LOG_DIR      = Path(r"C:\projects\audio_transition\main\logs")

# ── Environment ───────────────────────────────────────────────────────────────
load_dotenv()

def _load_env(path=".env"):
    env = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env

_env = _load_env()

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
_log_path = LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(_log_path, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger("flow")

# ── Load models (once at startup) ────────────────────────────────────────────
log.info("Loading embedding model ...")
_embed = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5", trust_remote_code=True)
_embed[0].auto_model.config.unpad_inputs = False
log.info("Embedding model ready.")

_qdrant    = QdrantClient(path=QDRANT_PATH)
_groq      = Groq(api_key=_env["groq_api"])
_dg_client = DeepgramClient(api_key=_env["deep_gram_key"])

# ── Audio helper ──────────────────────────────────────────────────────────────
def _to_mono_int16(data, orig_sr):
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

# ── Deepgram connection ───────────────────────────────────────────────────────
def _open_dg():
    q_final, q_partial = queue.Queue(), queue.Queue()
    cm = _dg_client.listen.v1.connect(
        model="nova-3", encoding="linear16", sample_rate=SAMPLE_RATE,
        language="en", interim_results="true", smart_format="true",
    )
    conn = cm.__enter__()

    def _on_msg(msg):
        try:
            text = msg.channel.alternatives[0].transcript
            if not text:
                return
            if getattr(msg, "is_final", False):
                q_final.put(text)
            else:
                while not q_partial.empty():
                    try: q_partial.get_nowait()
                    except queue.Empty: break
                q_partial.put(text)
        except Exception:
            pass

    conn.on(EventType.MESSAGE, _on_msg)
    threading.Thread(target=conn.start_listening, daemon=True).start()
    return conn, cm, q_final, q_partial

# ── RAG ───────────────────────────────────────────────────────────────────────
def _retrieve(text):
    t0 = time.perf_counter()
    vec = _embed.encode([text], normalize_embeddings=True).tolist()[0]
    t_embed = time.perf_counter() - t0

    t1 = time.perf_counter()
    result = _qdrant.query_points(collection_name=COLLECTION, query=vec, limit=TOP_K)
    t_search = time.perf_counter() - t1

    context = "\n".join(f"- {h.payload['text']}" for h in result.points)
    return context, t_embed, t_search

# ── LLM ───────────────────────────────────────────────────────────────────────
_SYSTEM = (
    "You are a real-time AI copilot for an sales agent on a live call. "
    "Based on the last few seconds of conversation and relevant knowledge, "
    "give exactly 3 short, actionable suggestions for the agent. "
    "Format: numbered list. One sentence each. Be direct and specific."
)

def _suggest(chunk, context):
    t0 = time.perf_counter()
    resp = _groq.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Conversation:\n{chunk}\n\nPolicy context:\n{context}"},
        ],
        temperature=0.4,
        max_completion_tokens=256,
        stream=False,
    )
    return resp.choices[0].message.content.strip(), time.perf_counter() - t0

# ── Background LLM worker ─────────────────────────────────────────────────────
_req_q = queue.Queue()   # (chunk, session_id, cycle_no, t_chunk_start)
_res_q = queue.Queue()   # (suggestions, session_id)

def _worker():
    session_stats = {}   # session_id -> list of cycle total_ms

    while True:
        item = _req_q.get()
        if item is None:
            break
        chunk, sid, cycle, t_chunk_start = item
        t_cycle = time.perf_counter()

        try:
            context, t_embed, t_search = _retrieve(chunk)
            suggestions, t_llm = _suggest(chunk, context)
            t_total = time.perf_counter() - t_cycle
            chunk_dur = time.perf_counter() - t_chunk_start

            session_stats.setdefault(sid, []).append(t_total * 1000)
            avg_ms = sum(session_stats[sid]) / len(session_stats[sid])

            # Per-cycle log
            log.info(
                f"[{sid}] Cycle #{cycle:03d} | "
                f"chunk={chunk_dur:.1f}s | "
                f"embed={t_embed*1000:.0f}ms | "
                f"search={t_search*1000:.0f}ms | "
                f"llm={t_llm*1000:.0f}ms | "
                f"total={t_total*1000:.0f}ms | "
                f"avg={avg_ms:.0f}ms"
            )
            log.info(f"  chunk   : {chunk[:120].replace(chr(10), ' ')}")
            log.info(f"  suggest : {suggestions[:200].replace(chr(10), ' ')}")
            log.info("  " + "-" * 64)

            _res_q.put((suggestions, sid))
        except Exception as e:
            log.error(f"[{sid}] Cycle #{cycle}: {e}")

threading.Thread(target=_worker, daemon=True).start()

# ── Gradio callbacks ──────────────────────────────────────────────────────────
_IDLE = "Listening... suggestions appear after the first 5 seconds of speech."

def process(audio, state):
    if audio is None:
        return state, state["display"] if state else "", state["suggestions"] if state else _IDLE

    if state is None:
        conn, cm, q_f, q_p = _open_dg()
        sid = datetime.now().strftime("%H%M%S%f")[:9]
        state = {
            "conn": conn, "cm": cm, "q_f": q_f, "q_p": q_p,
            "transcript": "", "display": "", "suggestions": _IDLE,
            "last_llm_t": time.perf_counter(),
            "last_chunk_t": time.perf_counter(),
            "last_pos": 0,
            "cycle": 0,
            "sid": sid,
            "t_session": time.perf_counter(),
        }
        log.info(f"{'='*20} SESSION {sid} STARTED {'='*20}")

    # Send audio chunk to Deepgram
    sr, data = audio
    state["conn"].send_media(_to_mono_int16(data, sr).tobytes())

    # Collect confirmed finals
    while not state["q_f"].empty():
        try:
            t = state["q_f"].get_nowait()
            if t:
                state["transcript"] = (state["transcript"] + " " + t).strip()
        except queue.Empty:
            break

    # Live partial preview in brackets
    partial = ""
    while not state["q_p"].empty():
        try: partial = state["q_p"].get_nowait()
        except queue.Empty: break

    state["display"] = state["transcript"]
    if partial:
        state["display"] += f" [{partial}]"

    # Trigger LLM cycle every LLM_INTERVAL seconds
    now = time.perf_counter()
    if now - state["last_llm_t"] >= LLM_INTERVAL:
        new_text = state["transcript"][state["last_pos"]:].strip()
        state["last_llm_t"] = now
        if new_text:
            state["cycle"] += 1
            state["last_pos"] = len(state["transcript"])
            _req_q.put((new_text, state["sid"], state["cycle"], state["last_chunk_t"]))
            state["last_chunk_t"] = now

    # Pull latest suggestion for this session
    while not _res_q.empty():
        try:
            sugg, sid = _res_q.get_nowait()
            if sid == state["sid"]:
                state["suggestions"] = sugg
        except queue.Empty:
            break

    return state, state["display"], state["suggestions"]


def end_session(state):
    if state:
        try: state["cm"].__exit__(None, None, None)
        except Exception: pass
        elapsed = time.perf_counter() - state["t_session"]
        log.info(
            f"{'='*20} SESSION {state['sid']} ENDED | "
            f"duration={elapsed:.1f}s | cycles={state['cycle']} {'='*20}\n"
        )
    return None, "", _IDLE

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Insurance Sales AI Copilot") as demo:
    gr.Markdown(
        "# Insurance Sales AI Copilot\n"
        "Speak naturally — transcription updates live, AI suggestions refresh every **5 seconds**."
    )

    with gr.Row():
        with gr.Column(scale=1):
            mic = gr.Audio(sources=["microphone"], streaming=True, label="Microphone")
            end_btn = gr.Button("End Session", variant="stop")
            gr.Markdown(f"Log: `{_log_path}`")

        with gr.Column(scale=2):
            transcript_box = gr.Textbox(
                label="Live Transcription",
                lines=14, interactive=False,
                placeholder="Transcription will appear here...",
            )

        with gr.Column(scale=2):
            suggestion_box = gr.Textbox(
                label="AI Suggestions  (refreshes every 5s)",
                lines=14, interactive=False,
                placeholder=_IDLE,
            )

    state = gr.State(None)
    mic.stream(process, inputs=[mic, state], outputs=[state, transcript_box, suggestion_box])
    end_btn.click(end_session, inputs=[state], outputs=[state, transcript_box, suggestion_box])

demo.launch()
