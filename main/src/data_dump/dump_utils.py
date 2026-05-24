"""
PDF Processing Utilities for Data Dumping
Handles extraction, chunking, and LLM-based processing
"""

import os
import json
import pdfplumber
from pathlib import Path
from typing import List, Dict
from datetime import datetime


class PDFProcessor:
    """Process PDFs and extract structured data"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        """
        Initialize PDF processor
        
        Args:
            chunk_size: Characters per chunk
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def stream_pdf_chunks(self, pdf_path: str):
        """
        Generator: yields one chunk dict at a time, processing one page at a time.
        Never loads the full PDF text into memory — buffer is at most one page + chunk_size.
        The final item yielded always contains '_metadata' with processing stats.
        """
        import gc

        filename = os.path.basename(pdf_path)
        total_pages = 0
        total_chars = 0
        chunk_count = 0
        buffer = ""

        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages = len(pdf.pages)
                source_info = {"source_file": filename, "total_pages": total_pages}

                for page_idx, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    page.flush_cache()

                    buffer = (buffer + "\n\n" + page_text) if buffer else page_text
                    total_chars += len(page_text)
                    del page_text

                    is_last_page = (page_idx == total_pages - 1)
                    overlap = min(self.chunk_overlap, 50)

                    while True:
                        if len(buffer) < self.chunk_size:
                            if not is_last_page:
                                break  # accumulate more pages
                            # last page: yield whatever remains
                            if buffer.strip():
                                chunk_count += 1
                                yield {"text": buffer.strip(),
                                       "chunk_length": len(buffer.strip()),
                                       **source_info}
                            buffer = ""
                            break

                        end = self.chunk_size
                        if buffer[end] not in (' ', '\n'):
                            space_pos = buffer.rfind(' ', 0, end)
                            if space_pos > 0:
                                end = space_pos

                        chunk_text = buffer[:end].strip()
                        buffer = buffer[max(0, end - overlap):]

                        if chunk_text:
                            chunk_count += 1
                            yield {"text": chunk_text,
                                   "chunk_length": len(chunk_text),
                                   **source_info}

                    gc.collect()

        except Exception as e:
            yield {
                "_metadata": {
                    "filename": filename,
                    "error": str(e),
                    "total_pages": total_pages,
                    "total_chars": total_chars,
                    "num_chunks": chunk_count,
                    "processed_at": datetime.now().isoformat()
                }
            }
            return

        yield {
            "_metadata": {
                "filename": filename,
                "total_pages": total_pages,
                "total_chars": total_chars,
                "num_chunks": chunk_count,
                "processed_at": datetime.now().isoformat()
            }
        }


class LLMEnricher:
    """Enrich chunks with LLM-based insights"""
    
    def __init__(self, groq_client):
        """
        Initialize with Groq client
        
        Args:
            groq_client: Groq API client
        """
        self.client = groq_client
    
    def extract_key_info(self, text: str, max_tokens: int = 300) -> str:
        """
        Use LLM to extract key information from text
        
        Args:
            text: Text to process
            max_tokens: Max tokens in response
            
        Returns:
            Key information extracted by LLM
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Extract key information, important rules, dates, and requirements from the text. Be concise."
                    },
                    {
                        "role": "user",
                        "content": f"Extract key info:\n{text[:1000]}"
                    }
                ],
                temperature=0.3,
                max_completion_tokens=max_tokens,
                top_p=1
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_summary(self, text: str, max_tokens: int = 200) -> str:
        """
        Generate summary of text using LLM
        
        Args:
            text: Text to summarize
            max_tokens: Max tokens in response
            
        Returns:
            Summary generated by LLM
        """
        try:
            completion = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {
                        "role": "system",
                        "content": "Provide a concise summary of the text."
                    },
                    {
                        "role": "user",
                        "content": f"Summarize:\n{text[:1500]}"
                    }
                ],
                temperature=0.3,
                max_completion_tokens=max_tokens,
                top_p=1
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"
    
    def enrich_chunks(self, chunks: List[Dict], use_summary: bool = True, use_key_info: bool = False) -> List[Dict]:
        """
        Enrich chunks with LLM insights (memory-efficient generator approach)
        
        Args:
            chunks: List of text chunks
            use_summary: Whether to add summaries
            use_key_info: Whether to extract key info
            
        Returns:
            Enriched chunks (processed in batches to save memory)
        """
        enriched = []
        batch_size = 10  # Process in small batches
        
        for i, chunk in enumerate(chunks):
            enriched_chunk = chunk.copy()
            
            if use_summary:
                enriched_chunk["llm_summary"] = self.generate_summary(chunk["text"])
            
            if use_key_info and i % 3 == 0:  # Every 3rd chunk
                enriched_chunk["llm_key_info"] = self.extract_key_info(chunk["text"])
            
            enriched.append(enriched_chunk)
            
            # Periodic garbage cleanup for large batch processing
            if (i + 1) % batch_size == 0:
                import gc
                gc.collect()
        
        return enriched


class DataDumpWriter:
    """Write processed data to various formats"""
    
    @staticmethod
    def save_json(data: List[Dict], output_path: str) -> None:
        """Save data as JSON (streaming to avoid large in-memory string)"""
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('[\n')
            for i, item in enumerate(data):
                f.write('  ' + json.dumps(item, ensure_ascii=False))
                if i < len(data) - 1:
                    f.write(',')
                f.write('\n')
            f.write(']\n')
        print(f"✅ Saved JSON: {output_path}")
    
    @staticmethod
    def save_text(chunks: List[Dict], output_path: str) -> None:
        """Save chunks as formatted text"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, chunk in enumerate(chunks, 1):
                source = chunk.get("source_file", "Unknown")
                f.write(f"=== CHUNK {i} (from {source}) ===\n")
                f.write(f"{chunk['text']}\n\n")
        print(f"✅ Saved Text: {output_path}")
    
    @staticmethod
    def save_metadata(metadata: Dict, output_path: str) -> None:
        """Save metadata file"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved Metadata: {output_path}")
