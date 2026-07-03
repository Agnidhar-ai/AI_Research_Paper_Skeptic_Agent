from tools.vector_store import InMemoryVectorStore


class Retriever:
    def __init__(self, vector_store: InMemoryVectorStore) -> None:
        self.vector_store = vector_store

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        return self.vector_store.search(query, top_k=top_k)
