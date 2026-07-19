"""
DB Inspector — shows both databases.
Usage (from main/src/):  python check_db.py
"""

import sys
from pathlib import Path

# Windows consoles default to the cp1252 codepage, which can't encode the
# box-drawing characters (─) used below — force UTF-8 so this runs in any terminal.
sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import HISTORY_SUMMARY_DIR, DOCS_VECTOR_DIR, CLEAR_RUNTIME_HISTORY
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

        vectors_cfg = info.config.params.vectors
        if hasattr(vectors_cfg, "size"):
            print(f"  Vector dim : {vectors_cfg.size}")
            print(f"  Distance   : {vectors_cfg.distance}")
        else:
            # Named vectors (e.g. docs collection: dense + sparse) — dict of name -> VectorParams
            for vname, vparams in vectors_cfg.items():
                print(f"  Vector[{vname}] dim : {getattr(vparams, 'size', 'sparse (no fixed dim)')}"
                      f"  dist={getattr(vparams, 'distance', 'n/a')}")
            sparse_cfg = getattr(info.config.params, "sparse_vectors", None) or {}
            for sname in sparse_cfg:
                print(f"  Sparse vector[{sname}] : present")

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

                if isinstance(point.vector, dict):
                    for vname, vec in point.vector.items():
                        if hasattr(vec, "indices"):  # SparseVector(indices=[...], values=[...])
                            print(f"    vector[{vname}]  : sparse, {len(vec.indices)} non-zero dims")
                        else:
                            print(f"    vector[{vname}]  : [{vec[0]:.5f}, {vec[1]:.5f}, ...] ({len(vec)} dims)")
                else:
                    print(f"    {'vector':<16}: [{point.vector[0]:.5f}, {point.vector[1]:.5f}, ...]"
                          f"  ({len(point.vector)} dims)")

    client.close()


# ── 1. Summary DB: session_summaries ───────────────────────────────────────────
inspect("SUMMARY DB", HISTORY_SUMMARY_DIR)

# ── 2. Docs DB: product-docs vector collection ─────────────────────────────────
inspect("DOCS DB (product PDFs)", DOCS_VECTOR_DIR)

# ── 3. History DB: runtime_history (only when CLEAR_RUNTIME_HISTORY=False) ────
if not CLEAR_RUNTIME_HISTORY:
    inspect("HISTORY DB (session turns)", HISTORY_SUMMARY_DIR / "history")
else:
    print(f"\n[HISTORY DB] CLEAR_RUNTIME_HISTORY=True → history was RAM-only, nothing on disk")

print(f"\n{'='*58}\n")
