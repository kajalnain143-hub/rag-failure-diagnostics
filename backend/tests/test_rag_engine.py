from app.rag_engine import retrieve, split_into_chunks
from app.diagnostics import diagnose, evaluate_retrieval


def test_chunking_preserves_text_and_overlap():
    text = "A" * 1200
    chunks = split_into_chunks(text, chunk_size=500, overlap=100)
    assert len(chunks) == 3
    assert chunks[0][-100:] == chunks[1][:100]


def test_retrieval_ranks_relevant_chunk_first():
    chunks = [
        {"id": 1, "document_id": 1, "filename": "a.txt", "content": "Python is a programming language."},
        {"id": 2, "document_id": 2, "filename": "b.txt", "content": "Photosynthesis converts light into chemical energy."},
    ]
    results = retrieve("What is Python programming language?", chunks, top_k=2)
    assert results[0]["chunk_id"] == 1
    assert results[0]["score"] > results[1]["score"]


def test_empty_retrieval_is_diagnosed():
    result = diagnose("What is in this file?", [])
    assert result["status"] == "no_context"


def test_reference_answer_coverage_is_bounded():
    chunks = [{"content": "Python supports object oriented programming."}]
    result = evaluate_retrieval(
        "What is Python?", "Python supports object oriented programming.", chunks
    )
    assert 0 <= result["expected_answer_term_coverage"] <= 1
