"""HTTP tests against the real saved model (needs the versions in requirements.txt)."""
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from mp044.api import app  # noqa: E402
from mp044.schemas import MilkRecord  # noqa: E402

GOOD = MilkRecord.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_model_info(client):
    assert len(client.get("/model-info").json()["features"]) == 8


def test_predict_matches_stage4_holdout_output(client):
    body = client.post("/predict", json=GOOD).json()
    assert body["record_id"] == "R50001"
    # same row and value as predictions/holdout_predictions.csv
    assert body["prediction"] == pytest.approx(807.3, abs=0.01)


def test_batch(client):
    res = client.post("/predict/batch", json={"records": [GOOD, GOOD]})
    assert res.status_code == 200 and len(res.json()["predictions"]) == 2


def test_invalid_input_returns_422(client):
    assert client.post("/predict", json={**GOOD, "cows": -3}).status_code == 422


def test_unloaded_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(app.state, "service", None)
    assert client.get("/health").status_code == 503
