# PDF Data Dumping Pipeline

Processes insurance PDFs and creates embeddings-ready chunks with LLM-based enrichment.

## Configuration

All paths and settings are centralized in `src/constants.py`:

```python
PDFS_DIR = "C:\projects\audio_transition_projects\data\pdfs"
OUTPUT_DIR = "C:\projects\audio_transition_projects\data\output"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
USE_LLM_ENRICHMENT = True
```

Edit `constants.py` to customize paths and settings globally.

- **PDF Extraction**: Extracts text, tables, and metadata from PDFs
- **Smart Chunking**: Splits text into chunks with sliding window (preserves context)
- **LLM Enrichment**: Uses Groq LLM to generate summaries and extract key information
- **Metadata Preservation**: Maintains source file, page numbers, and chunk positions
- **Multiple Outputs**: JSON (for embeddings), Text (for review), Metadata

## File Structure

```
dump_data/
├── dump_utils.py       # Core processing utilities
├── dump_data.py        # Main execution script
├── README.md           # This file
└── output/             # Generated outputs
    ├── chunks.json     # Chunks ready for embeddings
    ├── chunks.txt      # Formatted text for review
    └── metadata.json   # Processing metadata
```

## Usage

### Basic Usage

```bash
cd C:\projects\audio_transition_projects\voice_first_ai_system\main\src\data_dump
python dump_data.py
```

### Configuration

Edit `dump_data.py` to adjust:

```python
CHUNK_SIZE = 500           # Characters per chunk
CHUNK_OVERLAP = 100        # Overlap between chunks
USE_LLM_ENRICHMENT = True  # Enable LLM processing
USE_SUMMARIES = True       # Add LLM summaries
USE_KEY_INFO = False       # Extract key information
```

## Processing Strategy

### 1. PDF Extraction
- Uses `pdfplumber` for high-quality text extraction
- Preserves page structure and table data
- Extracts page-level metadata

### 2. Text Chunking
- Splits text into overlapping chunks
- Avoids cutting mid-word
- Preserves context with overlap

### 3. LLM Enrichment (Optional)
- Uses Groq API (llama-3.1-8b-instant)
- Generates summaries for better context
- Extracts key information from important sections

### 4. Output Generation
- **chunks.json**: Machine-readable format for embeddings
- **chunks.txt**: Human-readable format for review
- **metadata.json**: Processing statistics and PDF info

## Output Format

### chunks.json Structure
```json
[
  {
    "text": "The insurance corporation...",
    "chunk_start_char": 0,
    "chunk_end_char": 500,
    "chunk_length": 500,
    "source_file": "ILP.pdf",
    "total_pages": 45,
    "llm_summary": "This section covers...",
    "llm_key_info": "Key points: Rule 1..."
  }
]
```

## Next Steps

Use the generated `chunks.json` with the embedding pipeline:

```python
from sentence_transformers import SentenceTransformer
import json

# Load chunks
with open("output/chunks.json") as f:
    chunks = json.load(f)

# Extract text
texts = [c["text"] for c in chunks]

# Generate embeddings
model = SentenceTransformer("Alibaba-NLP/gte-large-en-v1.5")
embeddings = model.encode(texts, normalize_embeddings=True)

# Store in Qdrant (see parent ../dump_data.py for example)
```

## Dependencies

```
pdfplumber>=0.10.0
sentence-transformers>=2.2.0
qdrant-client>=2.0.0
groq>=0.4.0
python-dotenv>=1.0.0
```

Install with:
```bash
pip install pdfplumber sentence-transformers qdrant-client groq python-dotenv
```

## Requirements

- Python 3.8+
- `.env` file with `groq_api` key (optional, for LLM enrichment)
- Qdrant running (for embedding storage)

## Performance Notes

- Processing time depends on PDF size and LLM enrichment
- API calls with LLM enrichment may incur costs
- Recommend: 500-char chunks with 100-char overlap for optimal context
