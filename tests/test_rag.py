from tools.vector_store import InMemoryVectorStore


def test_vector_store_returns_matching_documents() -> None:
    store = InMemoryVectorStore()
    store.add_texts(["neural retrieval system", "gardening notes"])
    results = store.search("retrieval", top_k=1)
    assert results[0]["text"] == "neural retrieval system"
