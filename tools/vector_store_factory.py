from __future__ import annotations

import hashlib
import math
from pathlib import Path

from config import settings
from tools.vector_store import InMemoryVectorStore


class HashEmbeddings:
    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


class LangChainVectorStore:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.store = None
        self.embeddings = HashEmbeddings()

    def add_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        metadata = metadata or [{} for _ in texts]
        if self.backend == "faiss":
            from langchain_community.vectorstores import FAISS

            self.store = FAISS.from_texts(texts, self.embeddings, metadatas=metadata)
            return

        if self.backend == "chroma":
            from langchain_community.vectorstores import Chroma

            Path(settings.vector_db_dir).mkdir(parents=True, exist_ok=True)
            self.store = Chroma.from_texts(
                texts,
                self.embeddings,
                metadatas=metadata,
                persist_directory=settings.vector_db_dir,
            )
            return

        raise ValueError(f"Unsupported vector store backend: {self.backend}")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if self.store is None:
            return []

        matches = self.store.similarity_search_with_score(query, k=top_k)
        return [
            {
                "text": document.page_content,
                "metadata": document.metadata,
                "score": 1.0 / (1.0 + float(score)),
            }
            for document, score in matches
        ]


def create_vector_store():
    backend = settings.vector_store_backend.lower()
    if backend == "memory":
        return InMemoryVectorStore()
    if backend in {"faiss", "chroma"}:
        return LangChainVectorStore(backend)
    raise ValueError("VECTOR_STORE_BACKEND must be one of: memory, faiss, chroma")
