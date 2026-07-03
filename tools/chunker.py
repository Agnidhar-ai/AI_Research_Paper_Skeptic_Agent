def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be zero or less than chunk_size")

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text) and not text[end].isspace():
            boundary = text.rfind(" ", start + max(1, chunk_size // 2), end)
            if boundary > start:
                end = boundary

        chunks.append(text[start:end].strip())

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        if 0 < next_start < len(text) and not text[next_start].isspace():
            boundary = text.find(" ", next_start, min(len(text), next_start + overlap + 1))
            if boundary != -1:
                next_start = boundary + 1
        start = next_start
    return [chunk for chunk in chunks if chunk.strip()]
