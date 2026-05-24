"""
PDF -> Chunk -> Embed -> Qdrant Pipeline

For each batch of PAGES_PER_BATCH pages:
  1. Read pages from PDF
  2. Chunk the text
  3. Generate embeddings (sub-batched at EMBED_BATCH_SIZE)
  4. Upsert into Qdrant (sub-batched at QDRANT_UPSERT_BATCH)
  5. Free all batch memory before moving to the next batch
"""

import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pdfplumber
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, OrderBy, Direction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    PDFS_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_COLLECTION, QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT,
    EMBED_DIR,
    PAGES_PER_BATCH, EMBED_BATCH_SIZE, QDRANT_UPSERT_BATCH,
)


class PDFEmbeddingPipeline:
    """
    Processes PDFs in batches of PAGES_PER_BATCH pages.
    Each batch goes through the full read -> chunk -> embed -> upsert cycle
    before any memory from that batch is held over to the next.
    """

    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        self.model = SentenceTransformer(EMBEDDING_MODEL, trust_remote_code=True)
        # Fixes corrupt position_ids in Alibaba GTE model — must be set before any encode call
        self.model[0].auto_model.config.unpad_inputs = False
        print("Model ready.")

        EMBED_DIR.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(EMBED_DIR))
        self._ensure_collection()

    # ── Collection ────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        """Create the Qdrant collection if it does not exist; reuse it if it does."""
        existing = {c.name for c in self.client.get_collections().collections}
        if QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM,
                    distance=Distance.COSINE,
                ),
                hnsw_config=HnswConfigDiff(
                    m=QDRANT_HNSW_M,
                    ef_construct=QDRANT_HNSW_EF_CONSTRUCT,
                ),
            )
            print(f"Created collection '{QDRANT_COLLECTION}'")
        else:
            count = self.client.count(QDRANT_COLLECTION).count
            print(f"Reusing collection '{QDRANT_COLLECTION}' ({count} existing points)")

    # ── Next ID ───────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        """
        Return max existing point ID + 1.
        Orders by the 'chunk_index' payload field (which equals the native ID)
        so deletions never cause ID collisions with surviving rows.
        """
        results, _ = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1,
            order_by=OrderBy(key="chunk_index", direction=Direction.DESC),
            with_vectors=False,
            with_payload=True,
        )
        if results:
            return results[0].payload["chunk_index"] + 1
        return 0

    # ── Step 1: Read ──────────────────────────────────────────────────────────

    def _read_pages(self, pdf, page_indices: List[int]) -> str:
        """
        Extract text from the given page indices.
        Flush each page's cache immediately after extraction so pdfplumber
        does not accumulate parsed objects across pages.
        """
        parts = []
        for idx in page_indices:
            page = pdf.pages[idx]
            text = page.extract_text() or ""
            page.flush_cache()
            if text.strip():
                parts.append(text)
        return "\n\n".join(parts)

    # ── Step 2: Chunk ─────────────────────────────────────────────────────────

    def _chunk(self, text: str, source_meta: Dict) -> List[Dict]:
        """
        Sliding-window chunker.
        Overlap is capped at 50 chars to keep batch memory predictable.
        source_meta is merged into every chunk dict.
        """
        chunks = []
        overlap = min(CHUNK_OVERLAP, 50)
        buf = text

        while len(buf) >= CHUNK_SIZE:
            end = CHUNK_SIZE
            # Avoid splitting mid-word
            if end < len(buf) and buf[end] not in (" ", "\n"):
                space = buf.rfind(" ", 0, end)
                if space > 0:
                    end = space

            chunk_text = buf[:end].strip()
            if chunk_text:
                chunks.append({"text": chunk_text, **source_meta})

            buf = buf[max(0, end - overlap):]

        # Remaining text shorter than CHUNK_SIZE
        if buf.strip():
            chunks.append({"text": buf.strip(), **source_meta})

        return chunks

    # ── Step 3: Embed ─────────────────────────────────────────────────────────

    def _embed(self, chunks: List[Dict]) -> list:
        """
        Generate L2-normalised embeddings for all chunk texts.

        - batch_size controls how many texts are passed to the model at once,
          keeping GPU/CPU memory bounded.
        - normalize_embeddings=True is required for COSINE distance in Qdrant.
        - The numpy array is converted to a plain Python list (Qdrant requirement)
          and then deleted explicitly because .tolist() copies, not views.
        """
        texts = [c["text"] for c in chunks]
        embeddings_np = self.model.encode(
            texts,
            batch_size=EMBED_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        result = embeddings_np.tolist()
        del embeddings_np  # free the numpy array — .tolist() made a full copy
        return result

    # ── Step 4: Upsert ────────────────────────────────────────────────────────

    def _upsert(self, chunks: List[Dict], embeddings: list, start_id: int):
        """
        Build PointStructs with rich payload and upsert to Qdrant.

        Points are sub-batched at QDRANT_UPSERT_BATCH to avoid large single
        network/write calls. wait=True ensures each sub-batch is fully written
        before the next one starts.
        """
        points = [
            PointStruct(
                id=start_id + i,
                vector=emb,
                payload={
                    "text":         chunk["text"],
                    "source_file":  chunk.get("source_file", ""),
                    "page_range":   chunk.get("page_range", ""),
                    "batch_num":    chunk.get("batch_num", 0),
                    "chunk_index":  start_id + i,
                    "chunk_length": len(chunk["text"]),
                    "total_pages":  chunk.get("total_pages", 0),
                    "processed_at": chunk.get("processed_at", ""),
                },
            )
            for i, (chunk, emb) in enumerate(zip(chunks, embeddings))
        ]

        for i in range(0, len(points), QDRANT_UPSERT_BATCH):
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i : i + QDRANT_UPSERT_BATCH],
                wait=True,
            )

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str) -> int:
        """
        Process one PDF in batches of PAGES_PER_BATCH pages.

        Each batch completes the full pipeline and releases its memory
        before the next batch begins.  Returns total chunks inserted.
        """
        filename = os.path.basename(pdf_path)
        total_inserted = 0

        print(f"\n{'='*55}")
        print(f"PDF : {filename}")

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            total_batches = (total_pages + PAGES_PER_BATCH - 1) // PAGES_PER_BATCH
            print(f"Pages: {total_pages}  |  Batches: {total_batches}  ({PAGES_PER_BATCH} pages/batch)")
            print(f"{'='*55}")

            for batch_num, batch_start in enumerate(
                range(0, total_pages, PAGES_PER_BATCH), start=1
            ):
                batch_end   = min(batch_start + PAGES_PER_BATCH, total_pages)
                page_range  = f"{batch_start + 1}-{batch_end}"
                page_indices = list(range(batch_start, batch_end))

                print(f"  [{batch_num}/{total_batches}] pages {page_range}", end=" | ", flush=True)

                # ── 1. Read ───────────────────────────────────────────────
                text = self._read_pages(pdf, page_indices)
                if not text.strip():
                    print("no text — skipped")
                    continue

                # ── 2. Chunk ──────────────────────────────────────────────
                source_meta = {
                    "source_file":  filename,
                    "page_range":   page_range,
                    "batch_num":    batch_num,
                    "total_pages":  total_pages,
                    "processed_at": datetime.now().isoformat(),
                }
                chunks = self._chunk(text, source_meta)
                del text  # text no longer needed once chunked

                if not chunks:
                    print("no chunks — skipped")
                    gc.collect()
                    continue

                print(f"chunks={len(chunks)}", end=" | ", flush=True)

                # ── 3. Embed ──────────────────────────────────────────────
                embeddings = self._embed(chunks)

                # ── 4. Upsert ─────────────────────────────────────────────
                start_id = self._next_id()
                self._upsert(chunks, embeddings, start_id)
                total_inserted += len(chunks)
                print(f"upserted (ids {start_id}..{start_id + len(chunks) - 1})")

                # ── 5. Free batch memory ──────────────────────────────────
                del chunks, embeddings
                gc.collect()

        print(f"\n  Total inserted from '{filename}': {total_inserted} chunks")
        return total_inserted

    def close(self):
        self.client.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not PDFS_DIR.exists():
        print(f"PDFs directory not found: {PDFS_DIR}")
        return

    pdf_files = sorted(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PDFS_DIR}")
        return

    print(f"Found {len(pdf_files)} PDF(s)")
    print(f"Config: {PAGES_PER_BATCH} pages/batch | embed batch={EMBED_BATCH_SIZE} | upsert batch={QDRANT_UPSERT_BATCH}")

    pipeline = PDFEmbeddingPipeline()
    try:
        grand_total = 0
        for pdf_path in pdf_files:
            grand_total += pipeline.process_pdf(str(pdf_path))
        print(f"\nAll done. Grand total in DB: {grand_total} chunks")
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
