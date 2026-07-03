import math

from tools.embeddings import SimpleTfidfEmbedder


def cosine_similarity(left: list[float], right: list[float]) -> float:
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self.embedder = SimpleTfidfEmbedder()
        self.documents: list[dict] = []
        self.matrix: list[list[float]] = []

    def add_texts(self, texts: list[str], metadata: list[dict] | None = None) -> None:
        metadata = metadata or [{} for _ in texts]
        self.documents = [
            {"text": text, "metadata": meta}
            for text, meta in zip(texts, metadata, strict=True)
        ]
        self.matrix = self.embedder.fit_transform(texts)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.matrix or not self.documents:
            return []

        query_vector = self.embedder.transform([query])[0]
        scores = [cosine_similarity(query_vector, vector) for vector in self.matrix]
        ranked_indexes = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)[:top_k]

        return [
            {
                "text": self.documents[index]["text"],
                "metadata": self.documents[index]["metadata"],
                "score": float(scores[index]),
            }
            for index in ranked_indexes
        ]
