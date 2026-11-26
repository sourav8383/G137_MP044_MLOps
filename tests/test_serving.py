"""Validation and prediction logic. Uses a tiny model trained here, not the saved file."""
import joblib
import pytest
from pydantic import ValidationError

from mp044.cleaning import clean_training_data, split_xy
from mp044.modeling import build_pipeline, make_estimator
from mp044.schemas import BatchRequest, MilkRecord
from mp044.serving import ModelService

GOOD = MilkRecord.model_config["json_schema_extra"]["example"]


@pytest.fixture()
def service(tiny_train, params, tmp_path):
    X, y = split_xy(clean_training_data(tiny_train, params), params)
    d = params["data"]
    pipe = build_pipeline(make_estimator("linear_regression", {}), d["numeric"], d["categorical"], True)
    path = tmp_path / "model.joblib"
    joblib.dump(pipe.fit(X, y), path)
    return ModelService(model_path=path, params=params)


def test_valid_record_is_accepted():
    assert MilkRecord(**GOOD).cows == 55


@pytest.mark.parametrize(
    "change",
    [
        {"cows": -1},
        {"cows": 2.5},
        {"temp_c": 99},
        {"feed_kg": "lots"},
        {"feed_kg": float("inf")},
        {"fodder_price_rs": -5},
        {"breed_mix": "unknown"},
        {"village_cluster": "Cluster Z"},
        {"litres_paid_in_passbook": 800},
    ],
)
def test_bad_input_is_rejected(change):
    with pytest.raises(ValidationError):
        MilkRecord(**{**GOOD, **change})


def test_missing_field_is_rejected():
    with pytest.raises(ValidationError):
        MilkRecord(**{k: v for k, v in GOOD.items() if k != "cows"})


def test_text_is_tidied_and_sentinel_becomes_missing():
    r = MilkRecord(**{**GOOD, "breed_mix": " Mixed ", "fodder_price_rs": -999})
    assert r.breed_mix == "mixed" and r.fodder_price_rs is None


def test_empty_batch_is_rejected():
    with pytest.raises(ValidationError):
        BatchRequest(records=[])


def test_predicts_one_number_per_record_even_without_price(service):
    records = [
        MilkRecord(**GOOD).model_dump(),
        MilkRecord(**{**GOOD, "fodder_price_rs": None}).model_dump(),
    ]
    out = service.predict(records)
    assert len(out) == 2 and all(isinstance(v, float) for v in out)
    assert len(service.predict([records[1]])) == 1  # single row where the whole column is null
