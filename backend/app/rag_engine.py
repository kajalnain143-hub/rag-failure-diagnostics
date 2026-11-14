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


def _embedding_vectors(stored_chunks: list[dict]) -> list[list[float] | None]:
    vectors = []
    for item in stored_chunks:
        try:
            raw = item.get("embedding_json")
            vector = json.loads(raw) if isinstance(raw, str) else raw
            vectors.append([float(value) for value in vector] if vector else None)
        except (TypeError, ValueError, json.JSONDecodeError):
            vectors.append(None)
    return vectors


def _mmr_order(relevance: np.ndarray, similarity: np.ndarray, top_k: int,
               diversity: float = 0.35) -> list[int]:
    """Return indices selected by maximal marginal relevance."""
    remaining = set(range(len(relevance)))
    selected: list[int] = []
    while remaining and len(selected) < top_k:
        def mmr_score(index: int) -> float:
            redundancy = max((float(similarity[index, chosen]) for chosen in selected), default=0.0)
            return (1.0 - diversity) * float(relevance[index]) - diversity * redundancy
        best = max(remaining, key=lambda index: (mmr_score(index), float(relevance[index]), -index))
        selected.append(best)
        remaining.remove(best)
    return selected


def retrieve(question: str, stored_chunks: list[dict], top_k: int = 4,
             query_embedding: list[float] | None = None,
             strategy: str = "hybrid", mmr_diversity: float = 0.35) -> list[dict]:
    """Retrieve chunks using TF-IDF, hybrid scoring, or hybrid + MMR reranking."""
    if not stored_chunks or not question.strip():
        return []
    if strategy not in {"tfidf", "hybrid", "hybrid_mmr"}:
        raise ValueError("strategy must be one of: tfidf, hybrid, hybrid_mmr")
    if not 0.0 <= mmr_diversity <= 1.0:
        raise ValueError("mmr_diversity must be between 0 and 1.")

    texts = [item["content"] for item in stored_chunks]
    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(texts + [question])
        lexical_scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        lexical_similarity = cosine_similarity(matrix[:-1], matrix[:-1])
    except ValueError:
        lexical_scores = np.zeros(len(stored_chunks))
        lexical_similarity = np.zeros((len(stored_chunks), len(stored_chunks)))

    vectors = _embedding_vectors(stored_chunks)
    dense_scores = [
        _cosine(query_embedding, vector) if query_embedding and vector else 0.0
        for vector in vectors
    ]
    valid_dense = query_embedding is not None and any(score > 0 for score in dense_scores)
    has_dense = valid_dense and all(vector is not None for vector in vectors)
    results = []
    for index, item in enumerate(stored_chunks):
        lexical = float(lexical_scores[index])
        dense = float(dense_scores[index])
        hybrid = (0.45 * lexical + 0.55 * max(dense, 0.0)) if valid_dense else lexical
        score = lexical if strategy == "tfidf" else hybrid
        results.append({
            "chunk_id": item["id"], "document_id": item["document_id"],
            "filename": item["filename"], "content": item["content"],
            "score": round(score, 4),
            "lexical_score": round(lexical, 4),
            "semantic_score": round(dense, 4) if valid_dense else None,
            "retrieval_mode": "tfidf" if strategy == "tfidf" else ("hybrid" if valid_dense else "tfidf_fallback"),
        })

    limit = max(1, min(int(top_k), 10))
    if strategy == "hybrid_mmr":
        # Use embedding similarity when every chunk has a compatible vector;
        # otherwise use TF-IDF similarity so MMR remains deterministic offline.
        if has_dense:
            dense_matrix = np.asarray(vectors, dtype=float)
            norms = np.linalg.norm(dense_matrix, axis=1, keepdims=True)
            normalized = np.divide(dense_matrix, norms, out=np.zeros_like(dense_matrix), where=norms != 0)
            redundancy = normalized @ normalized.T
        else:
            redundancy = lexical_similarity
        relevance = np.asarray([item["score"] for item in results], dtype=float)
        selected = _mmr_order(relevance, redundancy, limit, mmr_diversity)
        results = [results[index] for index in selected]
        for item in results:
            item["retrieval_mode"] = "hybrid_mmr" if valid_dense else "tfidf_mmr_fallback"
    else:
        results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:limit]
    return results
