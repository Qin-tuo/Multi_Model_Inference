"""BGE query preprocessing on the SentenceTransformers embedder loaded by Mem0."""
from __future__ import annotations

QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


class BGEQueryAdapter:
    """Reuse the upstream model/pooling; apply the declared retrieval instruction."""

    def __init__(self, upstream, query_instruction=QUERY_INSTRUCTION):
        self.upstream = upstream
        self.model = upstream.model
        self.query_instruction = query_instruction
        self.config = upstream.config

    def embed(self, text, memory_action=None):
        return self.embed_batch([text], memory_action)[0]

    def embed_batch(self, texts, memory_action="add"):
        if not texts:
            return []
        inputs = [self.query_instruction + t if memory_action == "search" else t for t in texts]
        return self.model.encode(inputs, convert_to_numpy=True, normalize_embeddings=True).tolist()
