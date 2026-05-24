"""
Entry point: PDF -> Chunk -> Embed -> Qdrant

Usage:
    cd main/src
    python run_pipeline.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from data_dump.dump_pipeline import main

if __name__ == "__main__":
    main()
