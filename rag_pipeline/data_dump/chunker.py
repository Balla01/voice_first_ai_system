"""
Token-aware chunking for paragraph text, plus atomic (never-split) table chunks.

Paragraph chunking is a recursive splitter (paragraph -> sentence -> hard token
window fallback) sized in BGE-M3 tokens rather than characters, so chunk size
tracks what the embedding model actually sees. Plain recursive/fixed-size
chunking is used deliberately instead of semantic chunking — semantic chunking
tends to fragment into overly small pieces and doesn't reliably beat this on
benchmarks.

Tables are never split across chunks: one table (however large) becomes one
chunk, because splitting a premium/benefit table across chunks breaks row-level
lookups (e.g. "premium for age 45" needs the whole row, and often the header,
in the same chunk as the value).
"""
import re
from typing import List

from constants import CHUNK_TOKEN_SIZE, CHUNK_TOKEN_OVERLAP_RATIO

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _token_len(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def _split_into_sentences(paragraph: str) -> List[str]:
    return [s for s in _SENTENCE_SPLIT_RE.split(paragraph) if s.strip()]


def _hard_split_by_tokens(tokenizer, text: str, size: int, overlap: int) -> List[str]:
    """Fallback for a single sentence/paragraph longer than `size` tokens on its own."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= size:
        return [text]

    chunks = []
    start = 0
    step = max(size - overlap, 1)
    while start < len(ids):
        piece_ids = ids[start : start + size]
        chunks.append(tokenizer.decode(piece_ids))
        start += step
    return chunks


def chunk_paragraphs(
    tokenizer,
    paragraphs: List[str],
    size: int = CHUNK_TOKEN_SIZE,
    overlap_ratio: float = CHUNK_TOKEN_OVERLAP_RATIO,
) -> List[str]:
    """
    Greedily packs paragraphs -> sentences into chunks of at most `size` tokens,
    carrying `overlap_ratio * size` tokens of trailing context into the next
    chunk. Never splits a sentence unless that single sentence alone exceeds
    `size` tokens (rare; falls back to a hard token-window split).
    """
    overlap = int(size * overlap_ratio)
    units: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        for sent in _split_into_sentences(para):
            sent = sent.strip()
            if sent:
                units.append(sent)
        units.append("\n\n")  # paragraph boundary marker, joined back below

    chunks: List[str] = []
    current: List[str] = []
    current_tokens = 0

    def flush():
        text = " ".join(u for u in current if u != "\n\n").strip()
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            chunks.append(text)

    for unit in units:
        if unit == "\n\n":
            current.append(unit)
            continue

        unit_tokens = _token_len(tokenizer, unit)

        if unit_tokens > size:
            # Single oversized sentence: flush what we have, then hard-split it alone.
            if current:
                flush()
                current, current_tokens = [], 0
            for piece in _hard_split_by_tokens(tokenizer, unit, size, overlap):
                chunks.append(piece.strip())
            continue

        if current_tokens + unit_tokens > size and current:
            flush()
            # Carry trailing units worth ~overlap tokens into the next chunk
            carry: List[str] = []
            carry_tokens = 0
            for u in reversed(current):
                if u == "\n\n":
                    continue
                t = _token_len(tokenizer, u)
                if carry_tokens + t > overlap:
                    break
                carry.insert(0, u)
                carry_tokens += t
            current = carry
            current_tokens = carry_tokens

        current.append(unit)
        current_tokens += unit_tokens

    if current:
        flush()

    return chunks


def table_to_chunk(table_markdown: str, page: int) -> str:
    """Wraps a table's markdown as one atomic chunk — never split, regardless of size."""
    return f"[Table - Page {page}]\n{table_markdown}"
