from config import settings
from tools.chunker import chunk_text
from tools.pdf_loader import load_pdf_text
from tools.vector_store_factory import create_vector_store


def ingest_pdf(path: str):
    text = load_pdf_text(path)
    chunks = chunk_text(text, settings.chunk_size, settings.chunk_overlap)
    store = create_vector_store()
    store.add_texts(chunks, metadata=[{"source": path, "chunk": i} for i in range(len(chunks))])
    return store
