"""
layer4/intent_tier2_embedding.py — Tier 2: local sentence-embedding classifier.

Only runs if Tier 1's regex registry found zero matches. A full zero-shot
transformer (e.g. BART-MNLI) typically runs 200-500ms+ on CPU — too slow for
the ~30ms budget. Instead: embed the incoming text with a small model
(all-MiniLM-L6-v2, 22M params) and compare via cosine similarity against a
handful of reference example phrases per intent, reusing the same 6 intent
names as Tier 1. No training data or fine-tuning required.

The embedder is pluggable (Embedder protocol) so this degrades gracefully —
same pattern as layer3/tokens.py's tiktoken fallback — if sentence-transformers
can't download its model weights (e.g. huggingface.co blocked on a locked-down
network): Tier 2 just reports "no match" and the gate falls through to Tier 3,
rather than crashing the app.

NOTE: does NOT apply the 0.5 confidence threshold itself — it just returns
its best guess + score. TriggerGate applies the confidence gate uniformly,
per the design doc (Tier 1 is deterministic and always passes; only Tier 2's
genuine confidence score needs the threshold check).
"""

import logging
from typing import Dict, List, Optional, Protocol

from .models import IntentMatch
from .intent_tier1_regex import TRIGGER_REGISTRY

logger = logging.getLogger("insureassist.layer4")

REFERENCE_EXAMPLES = {
    "policy_inquiry": ["does this cover", "what's included in this plan", "tell me about the features"],
    "objection": ["this seems too costly", "I'm not sure I need this", "another company offers more"],
    "premium_concern": ["how much would this cost me", "what's the monthly payment", "any discount available"],
    "claim_question": ["how do I file a claim", "is the hospital cashless", "my claim got rejected"],
    "exclusion_concern": ["what's not covered", "is this excluded", "any restriction on pre-existing conditions"],
    "buying_signal": ["how do I sign up", "what documents are needed", "send me the payment link"],
}


class Embedder(Protocol):
    def embed(self, text: str) -> List[float]: ...


class MiniLMEmbedder:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        return self._model.encode(text)


def _cosine_similarity(a, b) -> float:
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class Tier2EmbeddingClassifier:
    def __init__(self, embedder: Optional[Embedder] = None):
        if embedder is not None:
            self._embedder = embedder
        else:
            try:
                self._embedder = MiniLMEmbedder()
                logger.debug("Tier2: MiniLM embedder loaded successfully")
            except Exception as e:
                logger.warning(
                    f"Tier2: embedder unavailable ({e}); Tier 2 will report no match on every "
                    "call, falling through to Tier 3. Fix network access to huggingface.co to "
                    "enable it, or pass a different Embedder."
                )
                self._embedder = None

        self._reference_embeddings: Dict[str, list] = {}
        if self._embedder is not None:
            for intent, examples in REFERENCE_EXAMPLES.items():
                self._reference_embeddings[intent] = [self._embedder.embed(ex) for ex in examples]

    def classify(self, text: str) -> Optional[IntentMatch]:
        if self._embedder is None:
            logger.debug("Tier2: embedder unavailable, skipping")
            return None

        query_vec = self._embedder.embed(text)
        best_intent, best_score = None, 0.0

        for intent, ref_vecs in self._reference_embeddings.items():
            score = max(_cosine_similarity(query_vec, rv) for rv in ref_vecs)
            if score > best_score:
                best_intent, best_score = intent, score

        logger.debug(f"Tier2 embedding: best_intent={best_intent} score={best_score:.3f}")

        if best_intent is None:
            return None

        config = TRIGGER_REGISTRY[best_intent]
        return IntentMatch(
            intent=best_intent,
            rag_collections=config["rag_collection"],
            response_tone=config["response_tone"],
            priority=config["priority"],
            confidence=best_score,
        )