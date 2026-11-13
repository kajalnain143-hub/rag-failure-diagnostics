import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def retrieve(question: str, stored_chunks: list[dict], top_k: int = 4) -> list[dict]:
    if not stored_chunks or not question.strip():
        return []

    texts = [item["content"] for item in stored_chunks]
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [question])
    except ValueError:
        return []

    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in ranked_indices:
        item = stored_chunks[int(index)]
        results.append({
            "chunk_id": item["id"],
            "document_id": item["document_id"],
            "filename": item["filename"],
            "content": item["content"],
            "score": round(float(scores[index]), 4),
        })
    return results
