from tools.retriever import Retriever
from tools.vector_store import InMemoryVectorStore


def query_store(store: InMemoryVectorStore, question: str, top_k: int = 5) -> list[dict]:
    retriever = Retriever(store)
    return retriever.search(question, top_k=top_k)
