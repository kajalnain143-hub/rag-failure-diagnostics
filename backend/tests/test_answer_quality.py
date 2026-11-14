import pytest

from app.answer_quality import context_precision_recall, parse_grounding_judgment


def test_parse_grounding_judgment_flags_unsupported_claims():
    result = parse_grounding_judgment(
        '{"faithfulness_score":0.5,"supported_claims":["A is stated"],'
        '"unsupported_claims":["B is not in evidence"]}'
    )
    assert result["faithfulness_score"] == 0.5
    assert result["hallucination_detected"] is True
    assert result["evaluation_type"] == "llm_as_judge"


def test_parse_grounding_judgment_rejects_invalid_score():
    with pytest.raises(ValueError):
        parse_grounding_judgment(
            '{"faithfulness_score":1.5,"supported_claims":[],"unsupported_claims":[]}'
        )


def test_parse_grounding_judgment_rejects_non_json():
    with pytest.raises(ValueError):
        parse_grounding_judgment("not json")


def test_context_precision_and_recall_use_gold_chunk_ids():
    result = context_precision_recall([1, 2, 3], [2, 3, 4])
    assert result["context_precision"] == 2 / 3
    assert result["context_recall"] == 2 / 3


def test_context_precision_and_recall_empty_inputs_are_safe():
    result = context_precision_recall([], [])
    assert result["context_precision"] == 0
    assert result["context_recall"] == 0
