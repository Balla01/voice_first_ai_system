"""
DB Inspector — shows both databases.
Usage (from main/src/):  python check_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import EMBED_DIR, CLEAR_RUNTIME_HISTORY
from qdrant_client import QdrantClient


def inspect(label: str, db_path: Path):
    if not db_path.exists():
        print(f"\n[{label}] path not found: {db_path}")
        return

    client = QdrantClient(path=str(db_path))
    collections = client.get_collections().collections

    print(f"\n{'='*58}")
    print(f"  DB      : {label}")
    print(f"  Path    : {db_path}")
    print(f"  Collections: {len(collections)}")

    for col in collections:
        name  = col.name
        count = client.count(name).count
        info  = client.get_collection(name)

        print(f"\n  {'─'*56}")
        print(f"  Collection : {name}")
        print(f"  Total rows : {count}  ← each row = one chunk/summary")
        print(f"  Vector dim : {info.config.params.vectors.size}")
        print(f"  Distance   : {info.config.params.vectors.distance}")

        if count > 0:
            results, _ = client.scroll(
                collection_name=name,
                limit=10,
                with_vectors=True,
                with_payload=True,
            )
            print(f"\n  Showing {len(results)} row(s):")
            for point in results:
                print(f"\n    id={point.id}")
                for key, val in point.payload.items():
                    display = str(val)[:110] + "..." if len(str(val)) > 110 else str(val)
                    print(f"    {key:<16}: {display}")
                print(f"    {'vector':<16}: [{point.vector[0]:.5f}, {point.vector[1]:.5f}, ...]"
                      f"  ({len(point.vector)} dims)")

    client.close()


# ── 1. Main DB: insurance_docs + session_summaries ────────────────────────────
inspect("MAIN DB (summaries + docs)", EMBED_DIR)

# ── 2. History DB: runtime_history (only when CLEAR_RUNTIME_HISTORY=False) ────
if not CLEAR_RUNTIME_HISTORY:
    inspect("HISTORY DB (session turns)", EMBED_DIR / "history")
else:
    print(f"\n[HISTORY DB] CLEAR_RUNTIME_HISTORY=True → history was RAM-only, nothing on disk")

print(f"\n{'='*58}\n")
