"""
Cross-encoder reranking for the docs retrieval path (feature d).

After hybrid dense+sparse search + RRF fusion produce a candidate shortlist,
a cross-encoder re-scores each (query, chunk) pair jointly — far more precise
than the bi-encoder similarity that produced the shortlist, because it attends
to the query and passage together. This is the single biggest precision lever
in retrieval; we keep only the top few reranked chunks for the LLM context.

Model + call pattern come from miscellaneous/test_reranker.py:
    BAAI/bge-reranker-v2-m3 via raw transformers AutoModelForSequenceClassification,
    score = sigmoid(logits). CPU-only here (torch is the +cpu build).

Lazy singleton, matching embedder.py / metadata_enricher.py: the model loads
once on first use so importing this module is cheap and a run with
USE_RERANKER=False never pays the ~2.2GB download/load cost.
"""
import threading
from typing import List, Tuple

from constants import RERANKER_MODEL, RERANK_BATCH_SIZE

_model = None
_tokenizer = None
_lock = threading.Lock()


def _get():
    global _model, _tokenizer
    if _model is None:
        with _lock:
            if _model is None:
                import torch  # noqa: F401  (ensures torch present before load)
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                print(f"Loading reranker model: {RERANKER_MODEL}")
                _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
                _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL)
                _model.eval()
                print("Reranker model ready.")
    return _model, _tokenizer


def rerank(query: str, candidates: List[Tuple[str, float]], top_k: int,
           batch_size: int = RERANK_BATCH_SIZE) -> List[Tuple[str, float]]:
    """
    Re-score `candidates` (list of (text, prior_score)) against `query` with the
    cross-encoder and return the top_k as (text, rerank_score), highest first.
    The prior_score (RRF score) is discarded — the cross-encoder score replaces it.
    Empty input -> empty output.
    """
    if not candidates:
        return []

    import torch
    model, tokenizer = _get()
    texts = [c[0] for c in candidates]

    scores: List[float] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        pairs = [[query, t] for t in batch]
        with torch.no_grad():
            inputs = tokenizer(pairs, padding=True, truncation=True,
                               return_tensors="pt", max_length=512)
            logits = model(**inputs, return_dict=True).logits.view(-1).float()
            scores.extend(torch.sigmoid(logits).tolist())

    ranked = sorted(zip(texts, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
