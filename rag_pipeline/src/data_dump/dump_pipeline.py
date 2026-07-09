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
import pyarrow as pa
import pyarrow.parquet as pq
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, HnswConfigDiff, OrderBy, Direction

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    LIC_ROOT,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    QDRANT_COLLECTION, QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT,
    EMBED_DIR,
    PAGES_PER_BATCH, EMBED_BATCH_SIZE, QDRANT_UPSERT_BATCH,
    get_output_paths,
)
from data_dump.lic_metadata import iter_lic_pdfs


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

        # Parquet mirror of the Qdrant collection (text + metadata, no vectors).
        # Rebuilt in full at the end of the run — see export_parquet().
        self._parquet_path = get_output_paths()["parquet"]
        self._parquet_path.parent.mkdir(parents=True, exist_ok=True)

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
            # Avoid splitting mid-word — but only trust the word boundary if it
            # still leaves room to advance past `overlap`; dense/tabular text
            # with no early spaces could otherwise pin end <= overlap forever,
            # making buf[max(0, end-overlap):] a no-op and looping infinitely.
            if end < len(buf) and buf[end] not in (" ", "\n"):
                space = buf.rfind(" ", 0, end)
                if space > overlap:
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

    def _build_payloads(self, chunks: List[Dict], start_id: int) -> List[Dict]:
        """Build the payload dict for each chunk — shared by Qdrant upsert and the Parquet mirror."""
        return [
            {
                "chunk_index":  start_id + i,
                "text":         chunk["text"],
                "source_file":  chunk.get("source_file", ""),
                "product_type": chunk.get("product_type", ""),
                "category":     chunk.get("category", ""),
                "sub_category": chunk.get("sub_category", ""),
                "plan_no":      chunk.get("plan_no", ""),
                "uin":          chunk.get("uin", ""),
                "doc_type":     chunk.get("doc_type", ""),
                "page_range":   chunk.get("page_range", ""),
                "batch_num":    chunk.get("batch_num", 0),
                "chunk_length": len(chunk["text"]),
                "total_pages":  chunk.get("total_pages", 0),
                "processed_at": chunk.get("processed_at", ""),
            }
            for i, chunk in enumerate(chunks)
        ]

    def _upsert(self, payloads: List[Dict], embeddings: list):
        """
        Build PointStructs from the given payloads and upsert to Qdrant.

        Points are sub-batched at QDRANT_UPSERT_BATCH to avoid large single
        network/write calls. wait=True ensures each sub-batch is fully written
        before the next one starts.
        """
        points = [
            PointStruct(id=payload["chunk_index"], vector=emb, payload=payload)
            for payload, emb in zip(payloads, embeddings)
        ]

        for i in range(0, len(points), QDRANT_UPSERT_BATCH):
            self.client.upsert(
                collection_name=QDRANT_COLLECTION,
                points=points[i : i + QDRANT_UPSERT_BATCH],
                wait=True,
            )

    # ── Parquet export ────────────────────────────────────────────────────────

    # Older points (pre-dating this metadata schema) may be missing some of these
    # keys — every row is normalized to this fixed column set before writing.
    _PARQUET_STR_FIELDS = (
        "text", "source_file", "product_type", "category", "sub_category",
        "plan_no", "uin", "doc_type", "page_range", "processed_at",
    )
    _PARQUET_INT_FIELDS = ("chunk_index", "batch_num", "chunk_length", "total_pages")

    def _normalize_payload(self, payload: Dict) -> Dict:
        row = {f: payload.get(f, "") for f in self._PARQUET_STR_FIELDS}
        row.update({f: payload.get(f, 0) for f in self._PARQUET_INT_FIELDS})
        return row

    def export_parquet(self):
        """
        Rebuild the Parquet mirror from the ENTIRE current Qdrant collection
        (old points + everything upserted this run), so it always reflects
        exactly what's in the vector db rather than just this run's new rows.
        """
        print(f"\nExporting Parquet mirror of '{QDRANT_COLLECTION}'...")
        writer = None
        offset = None
        total_rows = 0
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=QDRANT_COLLECTION,
                    limit=500,
                    offset=offset,
                    with_vectors=False,
                    with_payload=True,
                )
                if points:
                    rows = [self._normalize_payload(p.payload) for p in points]
                    table = pa.Table.from_pylist(rows)
                    if writer is None:
                        writer = pq.ParquetWriter(str(self._parquet_path), table.schema)
                    writer.write_table(table)
                    total_rows += len(rows)
                if next_offset is None:
                    break
                offset = next_offset
        finally:
            if writer is not None:
                writer.close()
        print(f"Saved Parquet ({total_rows} rows): {self._parquet_path}")

    # ── Main loop ─────────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str, folder_meta: Dict) -> int:
        """
        Process one PDF in batches of PAGES_PER_BATCH pages.

        folder_meta (product_type/category/sub_category/plan_no/uin/doc_type,
        derived from the LIC folder tree) is merged into every chunk's metadata.

        Each batch completes the full pipeline and releases its memory
        before the next batch begins.  Returns total chunks inserted.
        """
        filename = os.path.basename(pdf_path)
        total_inserted = 0

        print(f"\n{'='*55}")
        print(f"PDF : {filename}  ({folder_meta.get('sub_category', '')} / {folder_meta.get('doc_type', '')})")

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
                    **folder_meta,
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
                payloads = self._build_payloads(chunks, start_id)
                self._upsert(payloads, embeddings)
                total_inserted += len(chunks)
                print(f"upserted (ids {start_id}..{start_id + len(chunks) - 1})")

                # ── 5. Free batch memory ──────────────────────────────────
                del chunks, embeddings, payloads
                gc.collect()

        print(f"\n  Total inserted from '{filename}': {total_inserted} chunks")
        return total_inserted

    def close(self):
        self.client.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if not LIC_ROOT.exists():
        print(f"LIC root not found: {LIC_ROOT}")
        return

    pdf_entries = list(iter_lic_pdfs(LIC_ROOT))
    if not pdf_entries:
        print(f"No PDFs found under {LIC_ROOT}")
        return

    print(f"Found {len(pdf_entries)} PDF(s)")
    print(f"Config: {PAGES_PER_BATCH} pages/batch | embed batch={EMBED_BATCH_SIZE} | upsert batch={QDRANT_UPSERT_BATCH}")

    pipeline = PDFEmbeddingPipeline()
    try:
        grand_total = 0
        for pdf_path, folder_meta in pdf_entries:
            grand_total += pipeline.process_pdf(str(pdf_path), folder_meta)
        print(f"\nAll done. Grand total in DB: {grand_total} chunks")
        pipeline.export_parquet()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
