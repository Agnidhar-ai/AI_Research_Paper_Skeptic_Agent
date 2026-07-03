import math
import re
from collections import Counter


TOKEN_PATTERN = re.compile(r"[a-zA-Z][a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


class SimpleTfidfEmbedder:
    def __init__(self) -> None:
        self.vocabulary: list[str] = []
        self.idf: dict[str, float] = {}

    def transform(self, texts: list[str]):
        if not self.vocabulary:
            raise RuntimeError("Embedder must be fit before transform is called")
        return [self._embed(text) for text in texts]

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        document_tokens = [set(tokenize(text)) for text in texts]
        self.vocabulary = sorted(set().union(*document_tokens)) if document_tokens else []

        document_count = max(len(texts), 1)
        self.idf = {
            token: math.log((1 + document_count) / (1 + sum(token in doc for doc in document_tokens))) + 1
            for token in self.vocabulary
        }
        return [self._embed(text) for text in texts]

    def _embed(self, text: str) -> list[float]:
        counts = Counter(tokenize(text))
        total = sum(counts.values()) or 1
        return [
            (counts[token] / total) * self.idf.get(token, 0.0)
            for token in self.vocabulary
        ]
