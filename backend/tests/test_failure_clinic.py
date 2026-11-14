import pytest

from app.failure_clinic import PATTERNS, build_incident_prompt, validate_diagnosis, InvalidDiagnosisError


def test_taxonomy_contains_twelve_unique_patterns():
    ids = [pattern["id"] for pattern in PATTERNS]
    assert len(ids) == 12
    assert len(set(ids)) == 12


def test_prompt_includes_incident_and_taxonomy():
    prompt = build_incident_prompt("The answer ignored retrieved sources.")
    assert "P01" in prompt
    assert "The answer ignored retrieved sources." in prompt
    assert "Return ONLY valid JSON" in prompt


def test_valid_diagnosis_is_accepted():
    payload = {
        "primary_pattern_id": "P01",
        "secondary_pattern_ids": ["P09"],
        "severity": "high",
        "confidence": 0.88,
        "reasoning": ["Answer contradicted retrieved evidence."],
        "missing_evidence": ["Inspect the final prompt and source IDs."],
        "minimal_structural_fix": ["Add an evidence validation gate."],
        "verification_steps": ["Run a groundedness regression test."]
    }
    result = validate_diagnosis(__import__("json").dumps(payload))
    assert result["primary_pattern_id"] == "P01"


def test_unknown_pattern_is_rejected():
    payload = {
        "primary_pattern_id": "P99", "secondary_pattern_ids": [],
        "severity": "low", "confidence": 0.5, "reasoning": [],
        "missing_evidence": [], "minimal_structural_fix": [], "verification_steps": []
    }
    with pytest.raises(InvalidDiagnosisError):
        validate_diagnosis(__import__("json").dumps(payload))
