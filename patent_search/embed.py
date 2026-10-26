"""Sentence-embedding wrapper (bge-small by default: 384-d, strong retrieval quality for its size)."""
import os

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# bge models are trained with an instruction prefix for short queries -> passages; passage-to-passage
# queries (a claim, a whole patent) are embedded without it.
QUERY_PREFIX = {"BAAI/bge-small-en-v1.5": "Represent this sentence for searching relevant passages: "}


def best_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Embedder:
    def __init__(self, model_name=DEFAULT_MODEL, device=None, fp16=True, max_seq_length=256):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device or best_device()
        self.model = SentenceTransformer(model_name, device=self.device)
        self.model.max_seq_length = max_seq_length
        if fp16 and self.device != "cpu":
            self.model.half()
        self.dim = (getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension)()

    def encode_passages(self, texts, batch_size=128, show_progress=False):
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        # sentence-transformers sorts by length inside one call, so passing large lists minimizes padding.
        v = self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                              convert_to_numpy=True, show_progress_bar=show_progress)
        return v.astype(np.float32, copy=False)

    def encode_queries(self, texts, instruction=True):
        prefix = QUERY_PREFIX.get(self.model_name, "") if instruction else ""
        return self.encode_passages([prefix + t for t in texts], batch_size=32)
