import json

import joblib
import numpy as np
import pandas as pd
from mlflow import MlflowClient

from mp044 import modeling, train
from mp044.cleaning import clean_training_data, split_xy


def test_time_split_is_chronological(tiny_train, params):
    clean = clean_training_data(tiny_train, params)
    fit, valid = modeling.time_split(clean, "date", 0.2)
    assert fit["date"].max() <= valid["date"].min()
    assert len(valid) == len(clean) - int(len(clean) * 0.8)


def test_pipeline_handles_missing_values_and_unseen_category(tiny_train, params):
    clean = clean_training_data(tiny_train, params)
    X, y = split_xy(clean, params)
    est = modeling.make_estimator("ridge", {"alpha": 1.0})
    pipe = modeling.build_pipeline(est, params["data"]["numeric"], params["data"]["categorical"], scale=True)
    pipe.fit(X, y)
    new = X.head(2).copy()
    new.loc[new.index[0], "cows"] = np.nan                # a blanked stuck value
    new.loc[new.index[1], "breed_mix"] = "never-seen"     # unseen text must not crash
    pred = pipe.predict(new)
    assert pred.shape == (2,) and np.isfinite(pred).all()


def test_mae_is_average_absolute_error():
    assert modeling.mae([10, 20, 30], [12, 18, 33]) == (2 + 2 + 3) / 3


def test_training_logs_runs_registers_and_sets_champion(tiny_train, params, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # keep mlruns/ out of the repository
    p = json.loads(json.dumps(params))  # deep copy
    p["experiments"] = p["experiments"][:3]  # 3 cheap linear runs
    p["mlflow"]["experiment_name"] = "test-exp"
    uri = f"sqlite:///{tmp_path / 'test.db'}"

    results, pipelines = train.run_experiments(tiny_train, p, "fakehash", tracking_uri=uri)
    assert len(results) == 3
    best = results[0]
    assert best["mae_valid"] == min(r["mae_valid"] for r in results)

    version = train.register_champion(best["run_id"], p)
    client = MlflowClient()
    assert client.get_model_version_by_alias(p["mlflow"]["registered_model_name"], "champion").version == version

    run = client.get_run(best["run_id"])
    assert run.data.params["data_sha256"] == "fakehash"
    assert any("mae" in k for k in run.data.metrics)

    # the model we would save for the API round-trips and predicts the right shape
    path = tmp_path / "model.joblib"
    joblib.dump(pipelines[best["run_id"]], path)
    X, _ = split_xy(clean_training_data(tiny_train, p), p)
    assert joblib.load(path).predict(X.head(4)).shape == (4,)
