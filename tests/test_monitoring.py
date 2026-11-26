import numpy as np
import pandas as pd
import pytest

from mp044 import monitoring as mon
from mp044.cleaning import clean_training_data
from mp044.modeling import build_pipeline, make_estimator
from mp044.monitor import run


def test_psi_identical_is_zero():
    x = np.random.default_rng(0).normal(size=1000)
    assert mon.psi(x, x) == pytest.approx(0, abs=1e-9)


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert mon.psi(rng.normal(0, 1, 2000), rng.normal(3, 1, 2000)) > 0.25


def test_psi_ignores_nan_and_handles_out_of_range():
    e = pd.Series([1, 2, 3, 4, 5, np.nan] * 20)
    a = pd.Series([100, 200, np.nan] * 20)
    assert np.isfinite(mon.psi(e, a))


def test_mae_by_group_and_worst():
    df = pd.DataFrame({"g": ["A", "A", "B"], "y": [10.0, 20.0, 30.0]})
    out = mon.mae_by_group(df, [11, 19, 40], "g", "y")
    assert out == {"A": 1.0, "B": 10.0}
    assert mon.worst_group(out) == "B"


def test_split_check_is_by_date_70_30():
    df = pd.DataFrame({"date": pd.date_range("2026-07-01", periods=10)[::-1].strftime("%Y-%m-%d")})
    a, b = mon.split_check(df, "date", 0.3)
    assert (len(a), len(b)) == (7, 3)
    assert a["date"].max() < b["date"].min()


def test_decide_needs_8_percent_improvement():
    assert mon.decide(30.0, 27.0) == "promote"   # 10% better
    assert mon.decide(30.0, 27.6) == "promote"   # exactly 8% better
    assert mon.decide(30.0, 28.5) == "keep"      # only 5% better
    assert mon.decide(30.0, 35.0) == "keep"


def test_run_end_to_end(params, tiny_train):
    train_raw = tiny_train
    later_raw = tiny_train.drop_duplicates().copy()
    later_raw["record_id"] = "L" + later_raw["record_id"]
    later_raw["date"] = pd.date_range("2026-07-01", periods=len(later_raw)).strftime("%Y-%m-%d")
    later_raw["cows"] = later_raw["cows"] + 20  # induce drift
    holdout = later_raw.drop(columns=["litres_per_day", "litres_paid_in_passbook"]).head(10)
    d = params["data"]
    train = clean_training_data(train_raw, params)
    v1 = build_pipeline(make_estimator("linear_regression", {}), d["numeric"], d["categorical"], True)
    v1.fit(train[d["numeric"] + d["categorical"]], train[d["target"]])

    monitoring, v2, champion, preds = run(params, train_raw, later_raw, holdout, v1)
    assert set(monitoring["psi"]) == set(d["numeric"])
    assert monitoring["most_drifted_feature"] == "cows"
    assert set(monitoring["mae_by_group"]) == {"Cluster A", "Cluster B", "Cluster C"}
    assert monitoring["decision"] in ("promote", "keep")
    assert len(preds) == 10 and preds["prediction"].notna().all()


def test_update_findings_keeps_other_keys(tmp_path):
    import yaml
    from mp044.monitor import update_findings
    f = tmp_path / "findings.yaml"
    f.write_text("# keep me\ngroup_id: G137\n\nmonitoring:\n  decision: null\n")
    update_findings(f, {"decision": "keep", "psi": {"cows": 0.1}})
    assert "# keep me" in f.read_text()
    data = yaml.safe_load(f.read_text())
    assert data["group_id"] == "G137" and data["monitoring"]["decision"] == "keep"
