# voice_first_ai_system

A real-time AI conversation assistant for sales/support calls: captures customer and agent audio, converts speech to text, tracks rolling conversation memory, detects intent, retrieves relevant knowledge from memory/DBs, builds context-aware prompts, and streams LLM responses live to the UI.

---

## Project Structure

```
voice_first_ai_system/
└── main/
    └── src/
        ├── main.py                      # Complete RAG pipeline (parallel search + history + LLM)
        ├── constants.py                 # All paths, model config, and feature flags
        ├── check_db.py                  # Inspect both Qdrant databases
        ├── delete_rows.py               # Delete rows 0-24 by ID range
        ├── run_pipeline.py              # Entry point: PDF → embed → Qdrant
        ├── run_data_dump.py             # Entry point: PDF → chunks.json
        ├── data_dump/
        │   ├── dump_pipeline.py         # PDF → chunk → embed → Qdrant (batched)
        │   └── dump_data.py             # PDF → chunks.json / chunks.txt
        └── history/
            ├── history_pipeline.py      # Runtime history + summary pipeline
            └── test_history.py          # 10-query history test
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r main/requirements.txt
```

### 2. Configure environment

Create a `.env` file in `main/src/` (or project root):

```
groq_api=your_groq_api_key_here
```

### 3. Place PDFs

Drop your PDF files into:

```
C:\projects\audio_transition_projects\data\pdfs\
```

---

## How to Run

All commands run from:

```bash
cd voice_first_ai_system\main\src
```

### Dump PDF data to JSON

Extracts text from PDFs and writes chunks to `chunks.json` / `chunks.txt`.

```bash
python run_data_dump.py
```

### Build the vector database

Reads PDFs in 5-page batches → generates embeddings → upserts into Qdrant (`insurance_docs` collection).

```bash
python run_pipeline.py
```

### Run the main pipeline (complete flow)

Parallel search across 3 collections (runtime history, session summaries, insurance docs) with recency-weighted re-ranking and Groq LLM streaming response. Maintains conversation history per session.

```bash
python main.py
```

### Run the history pipeline test

10 test queries through the runtime history pipeline with session/customer filtering, eviction, and session summarization.

```bash
python history/test_history.py
```

### Inspect the database

Shows all Qdrant collections, row counts, and sample payloads for both the main DB and history DB.

```bash
python check_db.py
```

### Delete rows by ID range

Deletes point IDs 0 through 24 from the `insurance_docs` collection.

```bash
python delete_rows.py
```

---

## Key Configuration (`constants.py`)

| Constant | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `Alibaba-NLP/gte-large-en-v1.5` | Sentence transformer model (1024-dim) |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `PAGES_PER_BATCH` | `3` | PDF pages processed per batch |
| `EMBED_BATCH_SIZE` | `4` | Chunks encoded at once (tune for RAM) |
| `MAX_HISTORY_CHUNKS` | `20` | RAM cap before eviction triggers |
| `EVICT_COUNT` | `5` | Chunks evicted and summarized at once |
| `CLEAR_RUNTIME_HISTORY` | `False` | `True` = RAM only; `False` = persist to disk |
| `GROQ_MODEL` | `llama-3.1-8b-instant` | LLM used for responses and summarization |

---

## Pipeline Flow

```
PDF files
   │
   ├─ run_data_dump.py  →  chunks.json / chunks.txt
   │
   └─ run_pipeline.py   →  embed → Qdrant (insurance_docs)

Query (main.py)
   │
   ├─ embed query
   ├─ get_recent_history(5)          last 5 turns, chronological
   │
   ├─ ThreadPoolExecutor (3 workers)
   │   ├─ search runtime_history     semantic + recency re-rank
   │   ├─ search session_summaries   semantic + recency re-rank
   │   └─ search insurance_docs      semantic search
   │
   ├─ build_context()
   └─ Groq LLM (streaming)
        │
        └─ history.add(user + assistant)
              └─ eviction check → summarize → summary DB
```
