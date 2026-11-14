from app.rag_engine import retrieve, split_into_chunks


def test_hybrid_retrieval_returns_score_components():
    chunks = [
        {"id": 1, "document_id": 1, "filename": "a.txt", "content": "Car insurance policy coverage.", "embedding_json": "[1.0, 0.0]"},
        {"id": 2, "document_id": 1, "filename": "b.txt", "content": "Photosynthesis in plants.", "embedding_json": "[0.0, 1.0]"},
    ]
    results = retrieve("automobile coverage", chunks, 2, query_embedding=[1.0, 0.0])
    assert results[0]["chunk_id"] == 1
    assert results[0]["retrieval_mode"] == "hybrid"
    assert results[0]["semantic_score"] > results[1]["semantic_score"]


def test_tfidf_fallback_is_explicit():
    chunks = [{"id": 1, "document_id": 1, "filename": "a.txt", "content": "Python programming language."}]
    result = retrieve("Python", chunks)
    assert result[0]["retrieval_mode"] == "tfidf_fallback"
    assert result[0]["semantic_score"] is None


def test_invalid_chunk_overlap_rejected():
    import pytest
    with pytest.raises(ValueError):
        split_into_chunks("hello", chunk_size=10, overlap=10)
