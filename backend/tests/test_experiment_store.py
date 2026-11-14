import pytest

from app.database import init_db
from app.experiment_store import init_experiment_store, list_experiments, save_experiment


def test_experiment_history_persists_and_limits(monkeypatch, tmp_path):
    db_path = tmp_path / "experiments.sqlite"
    monkeypatch.setattr("app.database.DB_PATH", db_path)
    init_db()
    init_experiment_store()
    first_id = save_experiment("TF-IDF baseline", {"top_k": 4}, {"mrr": 0.5})
    save_experiment("Hybrid run", {"top_k": 5}, {"mrr": 0.8}, "semantic vectors enabled")
    rows = list_experiments(limit=1)
    assert rows[0]["name"] == "Hybrid run"
    assert rows[0]["metrics"]["mrr"] == 0.8
    assert first_id > 0


def test_experiment_name_required(monkeypatch, tmp_path):
    monkeypatch.setattr("app.database.DB_PATH", tmp_path / "experiments.sqlite")
    init_db()
    init_experiment_store()
    with pytest.raises(ValueError):
        save_experiment("  ", {}, {})
