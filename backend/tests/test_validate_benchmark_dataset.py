from scripts.validate_benchmark_dataset import validate_dataset


def test_valid_dataset_matches_current_corpus():
    corpus = {"chunks": [{"chunk_id": 4}, {"chunk_id": 8}]}
    dataset = {"dataset_name": "manual-v1", "cases": [{
        "query_id": "q1", "question": "Where is the policy?",
        "relevant_chunk_ids": [4], "reference_answer": "See policy.", "k": 5,
    }]}
    assert validate_dataset(dataset, corpus) == []


def test_missing_chunk_ids_are_reported():
    corpus = {"chunks": [{"chunk_id": 4}]}
    dataset = [{"query_id": "q1", "question": "Q", "relevant_chunk_ids": [99],
                "reference_answer": "A"}]
    errors = validate_dataset(dataset, corpus)
    assert any("not present in current corpus" in error for error in errors)


def test_duplicate_query_ids_and_gold_ids_are_rejected():
    corpus = {"chunks": [{"chunk_id": 4}]}
    case = {"query_id": "same", "question": "Q", "relevant_chunk_ids": [4, 4],
            "reference_answer": "A"}
    errors = validate_dataset([case, dict(case)], corpus)
    assert any("duplicate query_id" in error for error in errors)
    assert any("contains duplicates" in error for error in errors)


def test_invalid_types_and_empty_dataset_are_rejected():
    assert validate_dataset([], {"chunks": []})
    case = {"query_id": "q", "question": "", "relevant_chunk_ids": [True],
            "reference_answer": "A", "k": True}
    errors = validate_dataset([case], {"chunks": [{"chunk_id": 1}]})
    assert any("question" in error for error in errors)
    assert any("integer chunk IDs" in error for error in errors)
    assert any("k must be an integer" in error for error in errors)
