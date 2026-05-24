"""
Application Constants and Configuration
Define all paths, settings, and configurations here
"""

from pathlib import Path

# ── PROJECT PATHS ──
PROJECT_ROOT = Path(r"C:\projects\audio_transition_projects")
VOICE_AI_ROOT = PROJECT_ROOT / "voice_first_ai_system" / "main"

# ── DATA PATHS ──
DATA_ROOT = PROJECT_ROOT / "data"
PDFS_DIR = DATA_ROOT / "pdfs"
OUTPUT_DIR = DATA_ROOT / "output"

# ── SRC PATHS ──
SRC_DIR = VOICE_AI_ROOT / "src"
DATA_DUMP_DIR = SRC_DIR / "data_dump"

# ── MODELS & EMBEDDINGS ──
MODELS_DIR = VOICE_AI_ROOT / "models"
EMBED_DIR = VOICE_AI_ROOT / "embed"
LOGS_DIR = VOICE_AI_ROOT / "logs"

# ── TEXT PROCESSING CONFIG ──
CHUNK_SIZE = 1000  # Increased from 500 to reduce memory usage
CHUNK_OVERLAP = 150  # Adjusted for larger chunks

# ── LLM CONFIG ──
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 300

# ── EMBEDDINGS CONFIG ──
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_DISTANCE = "COSINE"

# ── QDRANT CONFIG ──
QDRANT_COLLECTION = "insurance_docs"
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100

# ── OUTPUT FILE NAMES ──
CHUNKS_JSON_FILE = "chunks.json"
CHUNKS_TEXT_FILE = "chunks.txt"
METADATA_JSON_FILE = "metadata.json"

# ── FEATURE FLAGS ──
USE_LLM_ENRICHMENT = False
USE_SUMMARIES = False
USE_KEY_INFO = False

# ── HISTORY CONFIG ──
USE_HISTORY           = True
HISTORY_COLLECTION    = "runtime_history"    # active session turns
SUMMARY_COLLECTION    = "session_summaries"  # summarized chunks (always on disk)
MAX_HISTORY_CHUNKS    = 20   # cap: evict when history exceeds this
EVICT_COUNT           = 5    # how many oldest chunks to evict at once

# True  → history lives in RAM only (:memory:), wiped when session ends
# False → history persists to disk, survives restarts, queryable across sessions
CLEAR_RUNTIME_HISTORY = False

# ── LLM CONFIG (Groq) ──
GROQ_MODEL         = "llama-3.1-8b-instant"

# ── PIPELINE BATCH CONFIG ──
# Tune these based on available RAM:
#   Low RAM  (4-6 GB)  -> EMBED_BATCH_SIZE = 4,  PAGES_PER_BATCH = 3
#   Mid RAM  (8 GB)    -> EMBED_BATCH_SIZE = 8,  PAGES_PER_BATCH = 5
#   High RAM (16+ GB)  -> EMBED_BATCH_SIZE = 16, PAGES_PER_BATCH = 5
PAGES_PER_BATCH     = 3   # pages read from PDF per iteration
EMBED_BATCH_SIZE    = 4   # chunks passed to model.encode() at once — main RAM knob
QDRANT_UPSERT_BATCH = 100 # points per Qdrant upsert call (safe at any RAM size)


def verify_paths():
    """Verify all required directories exist"""
    required_dirs = [PROJECT_ROOT, DATA_ROOT, VOICE_AI_ROOT, SRC_DIR]
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"⚠️  Warning: {dir_path} does not exist")
    
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDFS_DIR.mkdir(parents=True, exist_ok=True)
    
    return True


def get_output_paths():
    """Get all output file paths"""
    return {
        "chunks_json": OUTPUT_DIR / CHUNKS_JSON_FILE,
        "chunks_text": OUTPUT_DIR / CHUNKS_TEXT_FILE,
        "metadata_json": OUTPUT_DIR / METADATA_JSON_FILE,
        "output_dir": OUTPUT_DIR
    }


# Print configuration when imported
if __name__ == "__main__":
    print("=" * 60)
    print("📋 APPLICATION CONSTANTS")
    print("=" * 60)
    print(f"🗂️  Project Root: {PROJECT_ROOT}")
    print(f"🗂️  Voice AI Root: {VOICE_AI_ROOT}")
    print(f"📂 Data Root: {DATA_ROOT}")
    print(f"📂 PDFs Dir: {PDFS_DIR}")
    print(f"📂 Output Dir: {OUTPUT_DIR}")
    print(f"📂 Src Dir: {SRC_DIR}")
    print(f"\n⚙️  Processing Config:")
    print(f"   Chunk Size: {CHUNK_SIZE}")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"   LLM Enrichment: {USE_LLM_ENRICHMENT}")
    print(f"   Embedding Model: {EMBEDDING_MODEL}")
    print("=" * 60)
