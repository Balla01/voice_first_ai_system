"""
Main PDF Data Dumping Script
Processes all PDFs from data folder and creates embeddings-ready chunks
"""

import os
import json
import sys
import gc
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    PRODUCT_PDFS_ROOT, OUTPUT_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    USE_LLM_ENRICHMENT, USE_SUMMARIES, USE_KEY_INFO,
    get_output_paths, verify_paths
)
from data_dump.dump_utils import PDFProcessor, LLMEnricher, DataDumpWriter
from data_dump.lic_metadata import iter_product_pdfs


def main():
    """Main execution function"""
    
    # ── VERIFY PATHS ──
    verify_paths()
    
    # ── CONFIGURATION (imported from constants.py) ──
    print("=" * 60)
    print("📄 PDF DATA DUMPING PIPELINE")
    print("=" * 60)
    print(f"\n⚙️  Configuration:")
    print(f"   Product PDFs Root: {PRODUCT_PDFS_ROOT}")
    print(f"   Output Dir: {OUTPUT_DIR}")
    print(f"   Chunk Size: {CHUNK_SIZE} chars")
    print(f"   Chunk Overlap: {CHUNK_OVERLAP} chars")
    print(f"   LLM Enrichment: {USE_LLM_ENRICHMENT}")
    
    # ── STEP 1: Initialize components ──
    print("\n[1/5] Initializing components...")
    processor = PDFProcessor(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    writer = DataDumpWriter()
    
    # Initialize LLM if enrichment is enabled
    enricher = None
    llm_enabled = USE_LLM_ENRICHMENT
    if llm_enabled:
        from dotenv import load_dotenv
        from groq import Groq
        load_dotenv()
        api_key = os.getenv("groq_api")
        if not api_key:
            print("Warning: groq_api not found in .env, skipping LLM enrichment")
            llm_enabled = False
        else:
            client = Groq(api_key=api_key)
            enricher = LLMEnricher(client)
            print("LLM enricher initialized (Groq)")
    
    # ── STEP 2: Find and list PDFs ──
    print(f"\n[2/5] Scanning for PDFs in {PRODUCT_PDFS_ROOT}...")

    if not PRODUCT_PDFS_ROOT.exists():
        print(f"❌ Product PDFs root not found: {PRODUCT_PDFS_ROOT.absolute()}")
        return

    pdf_entries = list(iter_product_pdfs(PRODUCT_PDFS_ROOT))

    if not pdf_entries:
        print(f"❌ No PDFs found under {PRODUCT_PDFS_ROOT}")
        return

    print(f"✅ Found {len(pdf_entries)} PDF files:")
    for pdf_path, meta in pdf_entries:
        print(f"   • {pdf_path.name}  ({meta['product_type']} [{meta['doc_type']}])")
    
    # ── STEP 3 + 5: Process each PDF and stream directly to disk ──
    print(f"\n[3/5] Processing PDFs (streaming output to disk)...")
    all_metadata = []
    total_chunks = 0

    output_paths = get_output_paths()
    chunks_json_path = output_paths["chunks_json"]
    chunks_text_path = output_paths["chunks_text"]
    metadata_path = output_paths["metadata_json"]

    # Open output files once and stream into them
    chunks_json_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_text_path.parent.mkdir(parents=True, exist_ok=True)

    with open(chunks_json_path, 'w', encoding='utf-8') as json_f, \
         open(chunks_text_path, 'w', encoding='utf-8') as text_f:

        json_f.write('[\n')
        first_chunk = True

        for idx, (pdf_path, folder_meta) in enumerate(pdf_entries, 1):
            print(f"\n   [{idx}/{len(pdf_entries)}] Processing: {pdf_path.name}")

            pdf_metadata = None

            for item in processor.stream_pdf_chunks(str(pdf_path), extra_meta=folder_meta):
                # Last item from the generator is always metadata
                if "_metadata" in item:
                    pdf_metadata = item["_metadata"]
                    break

                # Optional LLM enrichment per chunk
                if llm_enabled and enricher:
                    try:
                        if USE_SUMMARIES:
                            item["llm_summary"] = enricher.generate_summary(item["text"])
                        if USE_KEY_INFO:
                            item["llm_key_info"] = enricher.extract_key_info(item["text"])
                    except Exception as e:
                        print(f"      LLM enrichment failed: {e}")

                # Write chunk immediately to disk — no list accumulation
                if not first_chunk:
                    json_f.write(',\n')
                json_f.write('  ' + json.dumps(item, ensure_ascii=False))
                first_chunk = False

                source = item.get("source_file", "Unknown")
                text_f.write(f"=== CHUNK {total_chunks + 1} (from {source}) ===\n")
                text_f.write(f"{item['text']}\n\n")
                total_chunks += 1

            if pdf_metadata is None:
                print(f"      Error: generator returned no metadata")
                continue

            if "error" in pdf_metadata:
                print(f"      Error: {pdf_metadata['error']}")
                continue

            print(f"      Extracted: {pdf_metadata['total_pages']} pages, {pdf_metadata['total_chars']} chars")
            print(f"      Created: {pdf_metadata['num_chunks']} chunks")
            all_metadata.append(pdf_metadata)
            gc.collect()

        json_f.write('\n]\n')

    print(f"\nTotal chunks created: {total_chunks}")
    print(f"Saved JSON: {chunks_json_path}")
    print(f"Saved Text: {chunks_text_path}")

    # ── STEP 4: (moved inside loop above when LLM enabled) ──
    print(f"\n[4/5] LLM enrichment: {'applied per-PDF' if llm_enabled else 'skipped'}")

    # ── STEP 5: Save metadata ──
    print(f"\n[5/5] Saving metadata...")
    writer.save_metadata({
        "processing_date": all_metadata[0].get("processed_at", "N/A") if all_metadata else "N/A",
        "total_pdfs": len(pdf_entries),
        "total_chunks": total_chunks,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "llm_enriched": llm_enabled,
        "pdf_metadata": all_metadata
    }, str(metadata_path))

    # ── SUMMARY ──
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Summary:")
    print(f"   PDFs processed: {len(pdf_entries)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Chunk size: {CHUNK_SIZE} chars (with {CHUNK_OVERLAP} overlap)")
    print(f"   LLM enriched: {llm_enabled}")
    print(f"\nOutput files:")
    print(f"   {chunks_json_path.absolute()}")
    print(f"   {chunks_text_path.absolute()}")
    print(f"   {metadata_path.absolute()}")
    print("\nNext step: Use chunks.json with embedding model")
    print("=" * 60)


if __name__ == "__main__":
    main()
