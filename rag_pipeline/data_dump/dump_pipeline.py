"""
PDF -> Chunk (extract_tables reading-order chunker) -> Qwen metadata per chunk
     -> Embed content (BGE-M3 hybrid dense+sparse) -> Qdrant Pipeline

Chunking is delegated wholesale to src/extract_tables.py's build_chunks():
given a PDF it decides the chunks itself (reading-order, one common token
budget shared by prose + tables, multi-page tables stitched, tables split with
their header repeated). Each chunk it returns looks like:

    {
      "chunk_id": "chunk_0001", "type": "prose|table|mixed",
      "pages": [1,2], "page_span": "1-2", "token_count": 361,
      "parts": [ ... ],                       # structured breakdown
      "content": "…text + table markdown…"    # <-- the ONLY field we embed
    }

For every chunk:
  1. text  = chunk["content"]                 (nothing else is embedded)
  2. meta  = Qwen(chunk["content"])           (per-chunk metadata, when enabled)
  3. dense+sparse = BGE-M3(chunk["content"])  (hybrid vector)
  4. upsert to Qdrant: vector + {content, chunk metadata, doc metadata, qwen metadata}

A whole PDF is skipped up front if its exact (doc_id, doc_version) is already
present in the collection — re-running only ingests new/changed documents.
"""

import gc
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pyarrow as pa
import pyarrow.parquet as pq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, PointStruct, HnswConfigDiff,
    OrderBy, Direction, PayloadSchemaType,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extract_tables
from constants import (
    PRODUCT_PDFS_ROOT,
    COMPLETE_FOLDER_STRUCTURE, LIC_DATA_ROOT,
    DOCS_EMBEDDING_DIM, DOCS_DENSE_VECTOR_NAME, DOCS_SPARSE_VECTOR_NAME,
    QDRANT_COLLECTION, QDRANT_HNSW_M, QDRANT_HNSW_EF_CONSTRUCT, DOCS_PAYLOAD_INDEX_FIELDS,
    DOCS_VECTOR_DIR,
    CHUNK_TOKEN_SIZE, CHUNK_TOKEN_OVERLAP_RATIO,
    EMBED_BATCH_SIZE, QDRANT_UPSERT_BATCH,
    USE_LLM_METADATA, DEFAULT_TENANT_ID,
    get_output_paths,
)
from data_dump.lic_metadata import iter_product_pdfs
from data_dump.lic_folder_walker import iter_complete_structure_pdfs
from data_dump.doc_versioning import derive_doc_metadata, already_indexed
from data_dump.embedder import embed_hybrid

if USE_LLM_METADATA:
    from data_dump.metadata_enricher import extract_metadata

CHUNK_OVERLAP_TOKENS = int(CHUNK_TOKEN_SIZE * CHUNK_TOKEN_OVERLAP_RATIO)


class PDFEmbeddingPipeline:
    """Chunk each PDF via extract_tables, tag + embed each chunk's content, upsert to Qdrant."""

    def __init__(self):
        # Make extract_tables count tokens with the SAME tokenizer BGE-M3 uses,
        # so chunk token budgets match what the embedding model actually sees.
        extract_tables.init_tokenizer(exact=True)

        DOCS_VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(DOCS_VECTOR_DIR))
        self._ensure_collection()

        self._parquet_path = get_output_paths()["parquet"]
        self._parquet_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Collection ────────────────────────────────────────────────────────────

    def _ensure_collection(self):
        """Create the collection (named dense+sparse vectors + payload indexes) if missing."""
        existing = {c.name for c in self.client.get_collections().collections}
        if QDRANT_COLLECTION not in existing:
            self.client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config={
                    DOCS_DENSE_VECTOR_NAME: VectorParams(size=DOCS_EMBEDDING_DIM, distance=Distance.COSINE),
                },
                sparse_vectors_config={DOCS_SPARSE_VECTOR_NAME: SparseVectorParams()},
                hnsw_config=HnswConfigDiff(m=QDRANT_HNSW_M, ef_construct=QDRANT_HNSW_EF_CONSTRUCT),
            )
            for field in DOCS_PAYLOAD_INDEX_FIELDS:
                self.client.create_payload_index(
                    collection_name=QDRANT_COLLECTION,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD,
                )
            print(f"Created collection '{QDRANT_COLLECTION}' (dense+sparse, {len(DOCS_PAYLOAD_INDEX_FIELDS)} payload indexes)")
        else:
            count = self.client.count(QDRANT_COLLECTION).count
            print(f"Reusing collection '{QDRANT_COLLECTION}' ({count} existing points)")

    # ── Next ID ───────────────────────────────────────────────────────────────

    def _next_id(self) -> int:
        results, _ = self.client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=1,
            order_by=OrderBy(key="chunk_index", direction=Direction.DESC),
            with_vectors=False, with_payload=True,
        )
        return results[0].payload["chunk_index"] + 1 if results else 0

    # ── Qwen per-chunk metadata ───────────────────────────────────────────────

    def _qwen_metadata(self, content: str) -> Dict:
        """Run the Qwen metadata model on ONE chunk's content (when USE_LLM_METADATA)."""
        if not USE_LLM_METADATA:
            return {}
        meta = extract_metadata(content)
        return {
            "llm_section_title":  meta.get("section_title", ""),
            "llm_chunk_type":     meta.get("chunk_type", "other"),
            "llm_key_terms":      meta.get("key_terms", []),
            "llm_contains_table": meta.get("contains_table", False),
            "llm_clause_numbers": meta.get("clause_numbers", []),
            "llm_summary":        meta.get("summary", ""),
        }

    # ── Payload ───────────────────────────────────────────────────────────────

    def _build_payload(self, chunk: Dict, point_id: int, doc_meta: Dict, source_meta: Dict) -> Dict:
        content = chunk["content"]
        pages = chunk.get("pages", [])
        payload = {
            "chunk_index":  point_id,
            "chunk_id":     f"{doc_meta['doc_id']}::{chunk['chunk_id']}",
            "local_chunk_id": chunk["chunk_id"],
            "doc_id":       doc_meta["doc_id"],
            # 'text' holds the embedded content — the query side reads payload['text'].
            "text":         content,
            "layout_type":  chunk.get("type", "prose"),        # prose | table | mixed
            "is_table":     chunk.get("type") == "table",
            "pages":        pages,
            "page":         min(pages) if pages else 0,
            "page_span":    chunk.get("page_span", ""),
            "token_count":  chunk.get("token_count", 0),
            "chunk_length": len(content),
            "source_file":  source_meta["source_file"],
            "product_type": source_meta["product_type"],
            "doc_type":     source_meta["doc_type"],
            "plan_name":    doc_meta["plan_name"],
            "doc_version":  doc_meta["doc_version"],
            "effective_date": doc_meta["effective_date"],
            "effective_date_source": doc_meta["effective_date_source"],
            "tenant_id":    DEFAULT_TENANT_ID,
            "processed_at": source_meta["processed_at"],
            # Only populated in COMPLETE_FOLDER_STRUCTURE mode (see
            # lic_folder_walker.py) — "" in flat mode, never absent, so
            # DOCS_PAYLOAD_INDEX_FIELDS can index them unconditionally.
            "category":     source_meta.get("category") or "",
            "plan_no":      source_meta.get("plan_no") or "",
            "uin_no":       source_meta.get("uin_no") or "",
            # The plan's folder name itself — stable across all PDFs under one
            # plan, unlike plan_name (regex-derived per-PDF from messy filenames).
            "plan_folder":  source_meta.get("plan_folder") or "",
        }
        payload.update(self._qwen_metadata(content))
        return payload

    def _upsert(self, payloads: List[Dict], dense_vecs: list, sparse_vecs: list):
        points = [
            PointStruct(
                id=p["chunk_index"],
                vector={DOCS_DENSE_VECTOR_NAME: dense, DOCS_SPARSE_VECTOR_NAME: sparse},
                payload=p,
            )
            for p, dense, sparse in zip(payloads, dense_vecs, sparse_vecs)
        ]
        for i in range(0, len(points), QDRANT_UPSERT_BATCH):
            self.client.upsert(collection_name=QDRANT_COLLECTION, points=points[i : i + QDRANT_UPSERT_BATCH], wait=True)

    # ── Parquet export ────────────────────────────────────────────────────────

    _PARQUET_STR_FIELDS = (
        "chunk_id", "local_chunk_id", "doc_id", "text", "layout_type", "page_span",
        "source_file", "product_type", "doc_type", "plan_name", "doc_version",
        "effective_date", "effective_date_source", "tenant_id", "processed_at",
        "category", "plan_no", "uin_no", "plan_folder",
    )
    _PARQUET_INT_FIELDS = ("chunk_index", "page", "token_count", "chunk_length")
    _PARQUET_BOOL_FIELDS = ("is_table",)

    def _normalize_payload(self, payload: Dict) -> Dict:
        row = {f: payload.get(f, "") for f in self._PARQUET_STR_FIELDS}
        row.update({f: payload.get(f, 0) for f in self._PARQUET_INT_FIELDS})
        row.update({f: payload.get(f, False) for f in self._PARQUET_BOOL_FIELDS})
        return row

    def export_parquet(self):
        print(f"\nExporting Parquet mirror of '{QDRANT_COLLECTION}'...")
        writer, offset, total_rows = None, None, 0
        try:
            while True:
                points, next_offset = self.client.scroll(
                    collection_name=QDRANT_COLLECTION, limit=500, offset=offset,
                    with_vectors=False, with_payload=True,
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

    # ── Main per-PDF ──────────────────────────────────────────────────────────

    def process_pdf(self, pdf_path: str, folder_meta: Dict, force: bool = False,
                     index: int = None, total: int = None) -> int:
        filename = os.path.basename(pdf_path)
        doc_meta = derive_doc_metadata(Path(pdf_path), folder_meta["product_type"], folder_meta["doc_type"])

        progress = f"[{index}/{total}] " if index is not None and total is not None else ""
        folder_bits = " / ".join(
            b for b in (folder_meta.get("category"), folder_meta.get("plan_folder")) if b
        )

        print(f"\n{'='*55}")
        print(f"{progress}PDF : {filename}  ({folder_meta.get('product_type', '')} / {folder_meta.get('doc_type', '')})")
        if folder_bits:
            print(f"Folder: {folder_bits}")
        print(f"doc_id={doc_meta['doc_id']} | version={doc_meta['doc_version']} | effective_date={doc_meta['effective_date']} ({doc_meta['effective_date_source']})")

        if not force and already_indexed(self.client, QDRANT_COLLECTION, doc_meta["doc_id"], doc_meta["doc_version"]):
            print("  Already indexed at this version — skipped (pass force=True to re-embed)")
            return 0

        # ── 1. Chunk the whole PDF (extract_tables decides the chunks) ──
        chunks = extract_tables.build_chunks(
            Path(pdf_path),
            chunk_tokens=CHUNK_TOKEN_SIZE,
            overlap_tokens=CHUNK_OVERLAP_TOKENS,
            table_max_rows=0,
        )
        if not chunks:
            print("  No chunks produced — skipped")
            return 0
        print(f"Chunks: {len(chunks)}  |  embed_batch={EMBED_BATCH_SIZE}  |  qwen_metadata={USE_LLM_METADATA}")
        print(f"{'='*55}")

        source_meta = {
            "source_file": filename,
            "processed_at": datetime.now().isoformat(),
            **folder_meta,
        }

        total_inserted = 0
        start_id = self._next_id()

        # ── 2-4. Embed content + Qwen metadata + upsert, in sub-batches ──
        for b0 in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[b0 : b0 + EMBED_BATCH_SIZE]
            contents = [c["content"] for c in batch]      # <-- only 'content' is embedded

            dense_vecs, sparse_vecs = embed_hybrid(contents, batch_size=EMBED_BATCH_SIZE)
            payloads = [
                self._build_payload(c, start_id + b0 + j, doc_meta, source_meta)
                for j, c in enumerate(batch)
            ]
            self._upsert(payloads, dense_vecs, sparse_vecs)
            total_inserted += len(batch)
            print(f"  upserted {total_inserted}/{len(chunks)} chunks (ids {start_id + b0}..{start_id + b0 + len(batch) - 1})")

            del dense_vecs, sparse_vecs, payloads
            gc.collect()

        print(f"\n  Total inserted from '{filename}': {total_inserted} chunks")
        return total_inserted

    def close(self):
        self.client.close()


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    if COMPLETE_FOLDER_STRUCTURE:
        root = LIC_DATA_ROOT
        if not root.exists():
            print(f"LIC data root not found: {root}")
            return
        pdf_entries = list(iter_complete_structure_pdfs(root))
    else:
        root = PRODUCT_PDFS_ROOT
        if not root.exists():
            print(f"Product PDFs root not found: {root}")
            return
        pdf_entries = list(iter_product_pdfs(root))

    if not pdf_entries:
        print(f"No PDFs found under {root}")
        return

    print(f"Found {len(pdf_entries)} PDF(s)")
    print(f"Config: chunk_tokens={CHUNK_TOKEN_SIZE} | overlap={CHUNK_OVERLAP_TOKENS} | embed_batch={EMBED_BATCH_SIZE} | qwen_metadata={USE_LLM_METADATA}")

    pipeline = PDFEmbeddingPipeline()
    try:
        grand_total = 0
        total = len(pdf_entries)
        for i, (pdf_path, folder_meta) in enumerate(pdf_entries, start=1):
            grand_total += pipeline.process_pdf(str(pdf_path), folder_meta, index=i, total=total)
        print(f"\nAll done. Grand total newly inserted: {grand_total} chunks")
        pipeline.export_parquet()
    finally:
        pipeline.close()


if __name__ == "__main__":
    main()
