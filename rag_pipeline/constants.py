"""
Application Constants and Configuration
Define all paths, settings, and configurations here
"""

import os
from pathlib import Path

# Derived from this file's own location. constants.py lives directly in
# rag_pipeline/, so its parent IS the pipeline root. Everything (code, vector
# DBs, history, embeddings) is kept self-contained under this one folder, so the
# whole rag_pipeline directory stays portable if moved/renamed.
VOICE_AI_ROOT = Path(__file__).resolve().parent          # .../insurance_assist_ai/rag_pipeline
PROJECT_ROOT = VOICE_AI_ROOT.parent                       # .../insurance_assist_ai

# ── DATA PATHS ──
# All runtime data lives INSIDE rag_pipeline (VOICE_AI_ROOT), not the outer project.
DATA_ROOT = VOICE_AI_ROOT / "data"
PDFS_DIR = DATA_ROOT / "pdfs"
OUTPUT_DIR = DATA_ROOT / "output"

# ── PRODUCT PDF DATA (current main data source) ──
# Layout (flat — one folder per product type, PDFs directly inside, no
# category/sub_category/plan-number nesting):
#   PRODUCT_PDFS_ROOT/insurance-plan/*.pdf
#   PRODUCT_PDFS_ROOT/pinsion_data/*.pdf
# Maps each source folder name to the normalized product_type value stored
# in chunk metadata.
PRODUCT_PDFS_ROOT = DATA_ROOT / "product_pdfs"
PRODUCT_TYPE_FOLDERS = {
    "insurance-plan": "insurance",
    "pinsion_data":   "pension",
}

# ── COMPLETE FOLDER STRUCTURE MODE (real LIC folder layout) ──
# When True, run_pipeline.py walks LIC_DATA_ROOT's real nested layout instead
# of the flat PRODUCT_PDFS_ROOT layout above:
#   LIC_DATA_ROOT/insurance-plans/<category>/<plan-folder>/*.pdf
#   LIC_DATA_ROOT/pension-plans/<plan-folder>/*.pdf              (no category level)
# <plan-folder> is named e.g. "lic-amritbaal-774-512n365v02" — the trailing
# "774" (plan_no) and "512n365v02" (uin_no) are parsed out and stored on every
# chunk from that folder (data_dump/lic_folder_walker.py). Only the two
# top-level folders below are ingested — sibling folders under LIC_DATA_ROOT
# (micro-insurance-plans, unit-linked-plans, withdrawn-plans, ...) are
# intentionally skipped.
COMPLETE_FOLDER_STRUCTURE = True
LIC_DATA_ROOT = Path(r"C:\projects\audio_transition_projects\data\LIC")
LIC_PRODUCT_TYPE_FOLDERS = {
    "insurance-plans": "insurance",
    "pension-plans":   "pension",
}

# ── SRC PATHS ──
# Code is flat inside rag_pipeline (no src/ subfolder), so SRC_DIR == VOICE_AI_ROOT.
SRC_DIR = VOICE_AI_ROOT
DATA_DUMP_DIR = SRC_DIR / "data_dump"

# ── MODELS & EMBEDDINGS ──
MODELS_DIR = VOICE_AI_ROOT / "models"
LOGS_DIR = VOICE_AI_ROOT / "logs"

EMBED_DIR = DATA_ROOT / "embed_files"
# Dedicated Qdrant storage path for the product-docs collection.
# Default lives at the project root (VOICE_AI_ROOT/vector_data_docs) — this is
# the DB the retrieval/API path reads from.
# Overridable via env so you can point retrieval/API at a backup or alternate DB
# for testing, then switch back by unsetting the var:
#   set DOCS_VECTOR_DIR_OVERRIDE=C:\voice_assistant_project\backup_data\docs_data_vectors_2
_DOCS_VECTOR_DIR_OVERRIDE = os.getenv("DOCS_VECTOR_DIR_OVERRIDE")
DOCS_VECTOR_DIR = Path(_DOCS_VECTOR_DIR_OVERRIDE) if _DOCS_VECTOR_DIR_OVERRIDE else VOICE_AI_ROOT / "vector_data_docs"

# Runtime history (chat turns) + session summaries live here, separate from
# the product-docs vector storage above.
#   HISTORY_SUMMARY_DIR/collection/session_summaries  — persistent summaries
#   HISTORY_SUMMARY_DIR/history/collection/runtime_history — persistent chat turns
HISTORY_SUMMARY_DIR = DATA_ROOT / "history_summary_data"

# ── TEXT PROCESSING CONFIG ──
CHUNK_SIZE = 1000  # Increased from 500 to reduce memory usage
CHUNK_OVERLAP = 150  # Adjusted for larger chunks

# ── LLM CONFIG ──
LLM_MODEL = "llama-3.1-8b-instant"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 300

# ── EMBEDDINGS CONFIG ──
# Used by history/summary collections (history_pipeline.py) and legacy docs code.
EMBEDDING_MODEL = "Alibaba-NLP/gte-large-en-v1.5"
EMBEDDING_DIM = 1024
EMBEDDING_DISTANCE = "COSINE"

# ── DOCS EMBEDDING CONFIG (hybrid dense+sparse via BGE-M3) ──
# Separate from EMBEDDING_MODEL above: history/summary stay on gte-large (unchanged),
# only the product-docs collection uses BGE-M3, since it's the one that benefits from
# lexical (sparse) matching on clause numbers, plan names, premium figures, etc.
DOCS_EMBEDDING_MODEL = "BAAI/bge-m3"
DOCS_EMBEDDING_DIM = 1024
DOCS_DENSE_VECTOR_NAME = "dense"
DOCS_SPARSE_VECTOR_NAME = "sparse"

# ── QDRANT CONFIG ──
# Product-docs collection (insurance + pension PDFs), stored at DOCS_VECTOR_DIR.
# "_v2" because the vector schema changed (single unnamed vector -> named
# dense+sparse vectors) — incompatible with the old collection's config, so this
# is a fresh collection rather than an in-place migration. The old
# "lic_insurance_docs" collection is left untouched on disk.
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION_OVERRIDE", "lic_insurance_docs_v2")
QDRANT_HNSW_M = 16
QDRANT_HNSW_EF_CONSTRUCT = 100

# Payload fields that get a Qdrant field index at collection-creation time
# (must exist before ingestion — see dump_pipeline.py's _ensure_collection).
# plan_no/uin_no/category only get populated in COMPLETE_FOLDER_STRUCTURE mode
# (empty string otherwise) but are indexed unconditionally so switching modes
# later doesn't require recreating the collection.
DOCS_PAYLOAD_INDEX_FIELDS = [
    "doc_id", "product_type", "doc_type", "plan_name", "tenant_id", "doc_version",
    "plan_no", "uin_no", "category", "plan_folder",
]

# ── CHUNKING CONFIG (token-aware, paragraph/table split) ──
# Replaces the old character-sliding-window chunker. Sizes are in BGE-M3 tokens.
CHUNK_TOKEN_SIZE = 400
CHUNK_TOKEN_OVERLAP_RATIO = 0.15   # 15% of CHUNK_TOKEN_SIZE

# ── TABLE EXTRACTION CONFIG (docling) ──
DOCLING_TABLE_MODE = "accurate"   # "fast" | "accurate" — see TableFormerMode

# ── LLM METADATA TAGGING (Qwen, local, see data_dump/metadata_enricher.py) ──
# Off by default: this box has no CUDA (torch reports cuda.is_available()=False),
# so Qwen3.5-0.8B generation runs on CPU — noticeably slower per chunk. Flip on
# deliberately once you've measured throughput on your corpus size.
USE_LLM_METADATA = True
QWEN_METADATA_MODEL = "Qwen/Qwen3.5-0.8B"
QWEN_METADATA_MAX_NEW_TOKENS = 256

# ── MULTI-TENANCY (payload field only for now; no filtering wired in yet) ──
DEFAULT_TENANT_ID = "default"

# ── RETRIEVAL: CROSS-ENCODER RERANKER (feature d) ──
# The single biggest precision lever: after hybrid RRF pulls a candidate pool,
# a cross-encoder re-scores (query, chunk) pairs and we keep the best few.
# CPU-only here, so it adds latency; flip USE_RERANKER off to skip it.
USE_RERANKER = False
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
RERANK_CANDIDATE_POOL = 20     # size of the RRF-fused shortlist fed to the reranker
RERANK_PREFETCH_LIMIT = 40     # per-branch (dense/sparse) prefetch depth before fusion
RERANK_BATCH_SIZE = 16         # (query, chunk) pairs scored per forward pass

# ── RETRIEVAL: LLM-DRIVEN METADATA FILTER (feature c) ──
# For each query, a fast LLM (Groq) extracts plan_name/doc_type/product_type
# mentioned in the query; validated against known values, then applied as a
# Qdrant query_filter so wrong-document chunks don't compete. A filtered search
# that returns nothing falls back to unfiltered (filtering never makes it worse).
USE_QUERY_FILTER = True
# QUERY_FILTER_MODEL is set to GROQ_MODEL below (after it is defined).
KNOWN_DOC_TYPES = ["Sales Brochure", "Customer Information Sheet", "Policy Document", "Other"]
KNOWN_PRODUCT_TYPES = ["insurance", "pension"]

# ── OUTPUT FILE NAMES ──
CHUNKS_JSON_FILE = "chunks.json"
CHUNKS_TEXT_FILE = "chunks.txt"
METADATA_JSON_FILE = "metadata.json"

# ── PARQUET MIRROR OF VECTOR DB ──
# Mirrors exactly what gets upserted into Qdrant (text + metadata, no vectors).
PARQUET_FILE_NAME = "lic_2_products_db_data.parquet"

# ── EVAL QUESTION BANK ──
QUESTIONS_JSON_PATH = DATA_ROOT / "questions.json"

# ── FEATURE FLAGS ──
USE_LLM_ENRICHMENT = False
USE_SUMMARIES = False
USE_KEY_INFO = False

# When True, api.py prints the full (untruncated) text of every chunk
# retrieved from the docs collection, instead of the normal 30-word preview.
DEBUG = True

# Number of chunks pulled from the docs collection per query — decoupled from
# the history/summary retrieval breadth (see main.py's parallel_search).
DOCS_SEARCH_K = 10

# ── HISTORY CONFIG ──
USE_HISTORY           = True
HISTORY_COLLECTION    = "runtime_history"    # active session turns
SUMMARY_COLLECTION    = "session_summaries"  # summarized chunks (always on disk)
MAX_HISTORY_CHUNKS    = 20   # cap: evict when history exceeds this
EVICT_COUNT           = 5    # how many oldest chunks to evict at once

# ── ADVANCED-FILTER MODE: chat_bot_ask_ai (api.py, request.advanced_filter) ──
# Persistent log of every {query, answer} pair asked under advanced_filter=True,
# separate from runtime_history (raw turns) and session_summaries (LLM digests) —
# used to pull "recent 5 Q&A" + "semantically relevant past Q&A" as extra context.
CHAT_ASK_AI_COLLECTION = "chat_bot_ask_ai"
CHAT_ASK_AI_DIR = HISTORY_SUMMARY_DIR / "chat_ask_ai"

# True  → history lives in RAM only (:memory:), wiped when session ends
# False → history persists to disk, survives restarts, queryable across sessions
CLEAR_RUNTIME_HISTORY = False

# ── LLM CONFIG (Groq) ──
GROQ_MODEL         = "llama-3.1-8b-instant"

# LLM used to extract the metadata filter from a query (see query_understanding.py).
# Groq (fast/cloud) — deliberately NOT the CPU Qwen model, which would add ~50s/query.
QUERY_FILTER_MODEL = GROQ_MODEL

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
    required_dirs = [PROJECT_ROOT, DATA_ROOT, VOICE_AI_ROOT, SRC_DIR, PRODUCT_PDFS_ROOT]

    for dir_path in required_dirs:
        if not dir_path.exists():
            print(f"⚠️  Warning: {dir_path} does not exist")

    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    return True


def get_output_paths():
    """Get all output file paths"""
    return {
        "chunks_json": OUTPUT_DIR / CHUNKS_JSON_FILE,
        "chunks_text": OUTPUT_DIR / CHUNKS_TEXT_FILE,
        "metadata_json": OUTPUT_DIR / METADATA_JSON_FILE,
        "parquet": OUTPUT_DIR / PARQUET_FILE_NAME,
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
