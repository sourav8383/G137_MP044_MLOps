"""Stage 4: drift check, per-cluster error, retrain v2, decide, predict holdout.

Run from the repository root (after `dvc pull` so the CSVs exist, and after Stage 2
training so models/model.joblib exists):
    PYTHONPATH=src python -m mp044.monitor
"""
import json

import joblib
import pandas as pd
import yaml
from sklearn.base import clone

from mp044.cleaning import clean_features, clean_training_data, split_xy
from mp044.config import ROOT, file_sha256, load_params
from mp044 import monitoring as mon


def run(params, train_raw, later_raw, holdout_raw, v1):
    d = params["data"]
    target, date_col, group = d["target"], d["date_column"], "village_cluster"

    train = clean_training_data(train_raw, params)
    later = clean_features(later_raw, params)

    # 1. drift: cleaned train vs later, 10 bins
    psi_values = mon.psi_table(train, later, d["numeric"], bins=10)

    # 2. v1 on all of later.csv, per village cluster
    X_later, y_later = split_xy(later, params)
    p1 = v1.predict(X_later)
    by_group = mon.mae_by_group(later, p1, group, target)

    # 3. retrain v2 = same pipeline recipe, train + oldest 70% of later; judge on newest 30%
    retrain_part, check = mon.split_check(later, date_col, 0.3)
    X_tr, y_tr = split_xy(train, params)
    X_rt, y_rt = split_xy(retrain_part, params)
    v2 = clone(v1).fit(pd.concat([X_tr, X_rt], ignore_index=True),
                       pd.concat([y_tr, y_rt], ignore_index=True))
    X_chk, y_chk = split_xy(check, params)
    mae_v1 = round(mon.mae(y_chk, v1.predict(X_chk)), 4)
    mae_v2 = round(mon.mae(y_chk, v2.predict(X_chk)), 4)
    decision = mon.decide(mae_v1, mae_v2)

    # 4. holdout predictions from whichever model wins
    champion = v2 if decision == "promote" else v1
    hold = clean_features(holdout_raw, params)
    X_h = hold[d["numeric"] + d["categorical"]]
    preds = pd.DataFrame({"record_id": hold["record_id"],
                          "prediction": champion.predict(X_h).round(2)})

    monitoring = {
        "psi": psi_values,
        "most_drifted_feature": mon.most_drifted(psi_values),
        "mae_by_group": by_group,
        "worst_served_group": mon.worst_group(by_group),
        "mae_v1_on_check": mae_v1,
        "mae_v2_on_check": mae_v2,
        "decision": decision,
    }
    return monitoring, v2, champion, preds


def update_findings(path, monitoring):
    """Replace only the `monitoring:` block of findings.yaml (keeps comments and other keys)."""
    text = path.read_text()
    head = text[: text.index("\nmonitoring:") + 1]
    block = yaml.safe_dump({"monitoring": monitoring}, sort_keys=False)
    path.write_text(head + block)


def main():
    params = load_params()
    p = params["paths"]
    train_raw = pd.read_csv(ROOT / p["train"])
    later_raw = pd.read_csv(ROOT / p["later"])
    holdout_raw = pd.read_csv(ROOT / p["holdout"])
    v1 = joblib.load(ROOT / p["model_out"])

    monitoring, v2, champion, preds = run(params, train_raw, later_raw, holdout_raw, v1)

    # Track everything in MLflow and register v2 only if it is promoted.
    import mlflow
    from mlflow import MlflowClient
    m = params["mlflow"]
    mlflow.set_tracking_uri(m["tracking_uri"])
    mlflow.set_experiment(m["experiment_name"])
    with mlflow.start_run(run_name="stage4_monitoring_retrain") as run_:
        mlflow.log_param("train_sha256", file_sha256(ROOT / p["train"]))
        mlflow.log_param("later_sha256", file_sha256(ROOT / p["later"]))
        mlflow.log_param("decision", monitoring["decision"])
        for k, v in monitoring["psi"].items():
            mlflow.log_metric(f"psi_{k}", v)
        for k, v in monitoring["mae_by_group"].items():
            mlflow.log_metric("mae_" + k.lower().replace(" ", "_"), v)
        mlflow.log_metric("mae_v1_on_check", monitoring["mae_v1_on_check"])
        mlflow.log_metric("mae_v2_on_check", monitoring["mae_v2_on_check"])
        mlflow.sklearn.log_model(v2, name="model_v2", serialization_format="cloudpickle")
        ver = mlflow.register_model(f"runs:/{run_.info.run_id}/model_v2",
                                    m["registered_model_name"])
        monitoring["v2_registered_version"] = int(ver.version)
        if monitoring["decision"] == "promote":
            MlflowClient().set_registered_model_alias(
                m["registered_model_name"], m["champion_alias"], ver.version)
            monitoring["champion_version"] = int(ver.version)
        monitoring["monitoring_run_id"] = run_.info.run_id

    (ROOT / "models").mkdir(exist_ok=True)
    joblib.dump(v2, ROOT / "models" / "model_v2.joblib")
    if monitoring["decision"] == "promote":
        joblib.dump(v2, ROOT / p["model_out"])  # the serving API loads this file

    (ROOT / "predictions").mkdir(exist_ok=True)
    preds.to_csv(ROOT / "predictions" / "holdout_predictions.csv", index=False)

    extra = ("champion_version", "v2_registered_version", "monitoring_run_id")
    report = {k: v for k, v in monitoring.items() if k not in extra}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "stage4_monitoring.json").write_text(json.dumps(report, indent=2))
    update_findings(ROOT / "findings.yaml", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
