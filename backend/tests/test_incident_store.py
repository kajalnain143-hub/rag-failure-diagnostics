from app import database, incident_store


def _diagnosis(pattern="P03", severity="high"):
    return {"primary_pattern_id": pattern, "secondary_pattern_ids": ["P02"],
            "severity": severity, "confidence": 0.87, "reasoning": ["evidence"],
            "missing_evidence": [], "minimal_structural_fix": ["inspect ranking"],
            "verification_steps": ["rerun benchmark"]}


def test_incident_store_round_trip_filters_and_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "incidents.db")
    incident_store.init_incident_store()
    first = incident_store.save_incident(
        {"description": "Retrieval returns irrelevant chunks", "expected_behavior": "Relevant evidence",
         "logs": "score=0.01"}, _diagnosis())
    incident_store.save_incident(
        {"description": "A startup dependency is unavailable"}, _diagnosis("P10", "critical"))
    assert first["diagnosis"]["confidence"] == 0.87
    assert first["secondary_pattern_ids"] == ["P02"]
    assert len(incident_store.list_incidents(pattern_id="P03")) == 1
    assert len(incident_store.list_incidents(severity="critical")) == 1
    summary = incident_store.incident_pattern_summary()
    assert summary["total_incidents"] == 2
    assert {"pattern_id": "P03", "count": 1} in summary["patterns"]
    assert incident_store.get_incident("missing") is None


def test_incident_store_rejects_unknown_taxonomy(monkeypatch, tmp_path):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "invalid.db")
    incident_store.init_incident_store()
    try:
        incident_store.save_incident({"description": "test"}, _diagnosis("P99"))
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("Expected invalid pattern ID to be rejected")
