import json
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def split_into_chunks(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Require chunk_size > overlap >= 0.")
    text = re.sub(r"\\s+", " ", text).strip()
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
        start = end - overlap
    return chunks


def _cosine(left: list[float], right: list[float]) -> float:
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else 0.0


def retrieve(question: str, stored_chunks: list[dict], top_k: int = 4,
             query_embedding: list[float] | None = None) -> list[dict]:
    if not stored_chunks or not question.strip():
        return []
    texts = [item["content"] for item in stored_chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [question])
        lexical_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    except ValueError:
        lexical_scores = np.zeros(len(stored_chunks))

    dense_scores = []
    for item in stored_chunks:
        raw = item.get("embedding_json")
        try:
            vector = json.loads(raw) if isinstance(raw, str) else raw
            dense_scores.append(_cosine(query_embedding, vector) if query_embedding and vector else 0.0)
        except (TypeError, ValueError, json.JSONDecodeError):
            dense_scores.append(0.0)

    has_dense = query_embedding is not None and any(score > 0 for score in dense_scores)
    results = []
    for index, item in enumerate(stored_chunks):
        lexical = float(lexical_scores[index])
        dense = float(dense_scores[index])
        hybrid = 0.45 * lexical + 0.55 * max(dense, 0.0) if has_dense else lexical
        results.append({
            "chunk_id": item["id"], "document_id": item["document_id"],
            "filename": item["filename"], "content": item["content"],
            "score": round(hybrid, 4),
            "lexical_score": round(lexical, 4),
            "semantic_score": round(dense, 4) if has_dense else None,
            "retrieval_mode": "hybrid" if has_dense else "tfidf_fallback",
        })
    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:max(1, min(int(top_k), 10))]
