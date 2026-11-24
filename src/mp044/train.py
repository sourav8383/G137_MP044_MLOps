"""Train v1 with MLflow tracking, register the best run, set @champion.

Run from the repository root:
    PYTHONPATH=src python -m mp044.train
"""
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature

from mp044.cleaning import clean_training_data, split_xy
from mp044.config import ROOT, file_sha256, load_params
from mp044.modeling import build_pipeline, mae, make_estimator, time_split


def run_experiments(train_raw, params, data_sha256, tracking_uri=None):
    """Train every entry of params['experiments'], one MLflow run each.

    Returns (results, pipelines): results is a list of dicts, best first;
    pipelines maps run_id -> pipeline refitted on ALL cleaned training rows.
    """
    d, t, m = params["data"], params["training"], params["mlflow"]
    mlflow.set_tracking_uri(tracking_uri or m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])

    clean = clean_training_data(train_raw, params)
    fit_part, valid_part = time_split(clean, d["date_column"], t["valid_fraction"])
    X_fit, y_fit = split_xy(fit_part, params)
    X_val, y_val = split_xy(valid_part, params)
    X_all, y_all = split_xy(clean, params)

    results, pipelines = [], {}
    for cfg in params["experiments"]:
        def new_pipeline():
            est = make_estimator(cfg["model"], cfg.get("hyperparameters"), t["random_state"])
            return build_pipeline(est, d["numeric"], d["categorical"], cfg.get("scale", False))

        with mlflow.start_run(run_name=cfg["name"]) as run:
            mlflow.log_param("data_sha256", data_sha256)
            mlflow.log_param("model_type", cfg["model"])
            mlflow.log_param("scale", cfg.get("scale", False))
            mlflow.log_params({f"hp_{k}": v for k, v in (cfg.get("hyperparameters") or {}).items()})
            mlflow.log_param("n_rows_clean", len(clean))
            mlflow.log_param("valid_fraction", t["valid_fraction"])
            mlflow.log_param("features", ",".join(d["numeric"] + d["categorical"]))

            check = new_pipeline().fit(X_fit, y_fit)
            mae_valid = mae(y_val, check.predict(X_val))
            mae_train = mae(y_fit, check.predict(X_fit))
            mlflow.log_metric("mae_valid", mae_valid)
            mlflow.log_metric("mae_train", mae_train)

            # The model we keep is refitted on every cleaned row (validation included).
            final = new_pipeline().fit(X_all, y_all)
            mlflow.sklearn.log_model(
                final,
                name="model",
                signature=infer_signature(X_all, final.predict(X_all)),
                input_example=X_all.head(3),
                serialization_format="cloudpickle",
            )
            pipelines[run.info.run_id] = final
            results.append({"run_id": run.info.run_id, "name": cfg["name"],
                            "mae_valid": mae_valid, "mae_train": mae_train})

    results.sort(key=lambda r: r["mae_valid"])
    return results, pipelines


def register_champion(best_run_id, params):
    """Register the best run's model and give it the 'champion' alias."""
    m = params["mlflow"]
    version = mlflow.register_model(f"runs:/{best_run_id}/model", m["registered_model_name"])
    MlflowClient().set_registered_model_alias(
        m["registered_model_name"], m["champion_alias"], version.version
    )
    return version.version


def main():
    params = load_params()
    train_path = ROOT / params["paths"]["train"]
    sha = file_sha256(train_path)
    train_raw = pd.read_csv(train_path)

    results, pipelines = run_experiments(train_raw, params, sha, tracking_uri=None)
    best = results[0]
    version = register_champion(best["run_id"], params)

    out = ROOT / params["paths"]["model_out"]
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipelines[best["run_id"]], out)

    summary = {
        "data_sha256": sha,
        "experiment_name": params["mlflow"]["experiment_name"],
        "registered_model_name": params["mlflow"]["registered_model_name"],
        "champion_version": int(version),
        "best_run_id": best["run_id"],
        "runs": results,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "v1_training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
