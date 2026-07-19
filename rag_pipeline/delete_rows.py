"""
Delete specific rows from Qdrant by ID range.
Usage: python delete_rows.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from constants import DOCS_VECTOR_DIR, QDRANT_COLLECTION

from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

client = QdrantClient(path=str(DOCS_VECTOR_DIR))

DELETE_IDS = list(range(0, 25))  # 0 to 24 inclusive

before = client.count(QDRANT_COLLECTION).count
print(f"Rows before delete : {before}")
print(f"Deleting IDs       : {DELETE_IDS[0]} to {DELETE_IDS[-1]} ({len(DELETE_IDS)} rows)")

client.delete(
    collection_name=QDRANT_COLLECTION,
    points_selector=PointIdsList(points=DELETE_IDS),
    wait=True,
)

after = client.count(QDRANT_COLLECTION).count
print(f"Rows after delete  : {after}")
print(f"Deleted            : {before - after} rows")

client.close()
