"""
Hybrid (dense + sparse) embedding for the docs collection, via BGE-M3.

BGE-M3 produces dense, sparse (lexical/BM25-like), and multi-vector (ColBERT)
representations from a single forward pass. We use dense (semantic similarity)
+ sparse (exact term/clause-number/premium-figure matching) and skip the
ColBERT multi-vector output — it's a reranking-stage tool, not something
Qdrant's ANN index consumes directly, and would meaningfully increase storage
and indexing cost for a benefit this pipeline doesn't use yet.

Separate from history_pipeline.py's `_embed()` (gte-large-en-v1.5), which
still serves the runtime_history/session_summaries collections unchanged.
"""
import threading
from typing import Dict, List, Tuple

from qdrant_client.models import SparseVector

from constants import DOCS_EMBEDDING_MODEL

_model = None
_model_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from FlagEmbedding import BGEM3FlagModel
                print(f"Loading docs embedding model: {DOCS_EMBEDDING_MODEL}")
                _model = BGEM3FlagModel(DOCS_EMBEDDING_MODEL, use_fp16=False)
                print("Docs embedding model ready.")
    return _model


def _lexical_weights_to_sparse_vector(weights: Dict[str, float]) -> SparseVector:
    indices = [int(token_id) for token_id in weights.keys()]
    values = [float(w) for w in weights.values()]
    return SparseVector(indices=indices, values=values)


def embed_hybrid(texts: List[str], batch_size: int = 4) -> Tuple[List[List[float]], List[SparseVector]]:
    """Returns (dense_vectors, sparse_vectors) — one pair per input text, same order."""
    model = _get_model()
    output = model.encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = output["dense_vecs"].tolist()
    sparse = [_lexical_weights_to_sparse_vector(w) for w in output["lexical_weights"]]
    return dense, sparse


def embed_query_hybrid(query: str) -> Tuple[List[float], SparseVector]:
    """Single-query convenience wrapper for retrieval time (see history_pipeline.py)."""
    dense, sparse = embed_hybrid([query], batch_size=1)
    return dense[0], sparse[0]
