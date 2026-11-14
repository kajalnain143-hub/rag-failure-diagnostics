from app.rag_engine import retrieve


def _chunks():
    return [
        {"id": 1, "document_id": 1, "filename": "a.txt", "content": "Python programming language and software development.", "embedding_json": "[1.0, 0.0]"},
        {"id": 2, "document_id": 1, "filename": "a.txt", "content": "Python programming language with software development tools.", "embedding_json": "[0.98, 0.02]"},
        {"id": 3, "document_id": 2, "filename": "b.txt", "content": "Python snakes are reptiles that live in forests.", "embedding_json": "[0.0, 1.0]"},
    ]


def test_tfidf_strategy_is_explicit_and_bounded():
    results = retrieve("Python programming", _chunks(), 2, query_embedding=[1.0, 0.0], strategy="tfidf")
    assert len(results) == 2
    assert results[0]["retrieval_mode"] == "tfidf"
    assert results[0]["semantic_score"] is not None


def test_hybrid_mmr_is_deterministic_and_returns_unique_chunks():
    first = retrieve("Python programming", _chunks(), 3, query_embedding=[1.0, 0.0], strategy="hybrid_mmr")
    second = retrieve("Python programming", _chunks(), 3, query_embedding=[1.0, 0.0], strategy="hybrid_mmr")
    assert [row["chunk_id"] for row in first] == [row["chunk_id"] for row in second]
    assert len({row["chunk_id"] for row in first}) == 3
    assert first[0]["retrieval_mode"] == "hybrid_mmr"


def test_mmr_falls_back_to_tfidf_without_embeddings():
    chunks = [{"id": 7, "document_id": 1, "filename": "x.txt", "content": "Python programming"}]
    result = retrieve("Python", chunks, strategy="hybrid_mmr")
    assert result[0]["retrieval_mode"] == "tfidf_mmr_fallback"


def test_unknown_retrieval_strategy_rejected():
    import pytest
    with pytest.raises(ValueError):
        retrieve("Python", _chunks(), strategy="bm25")
