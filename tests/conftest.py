"""Shared small fixtures. The real CSVs are NOT in git, so tests build their own data."""
import numpy as np
import pandas as pd
import pytest

from mp044.config import load_params


@pytest.fixture(scope="session")
def params():
    return load_params()


@pytest.fixture()
def tiny_train(params):
    """60 daily rows, in the same shape as train.csv, with the Stage 1 problems injected."""
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "record_id": [f"T{i:03d}" for i in range(n)],
            "date": pd.date_range("2026-02-01", periods=n).strftime("%Y-%m-%d"),
            "cows": rng.integers(10, 200, n),
            "buffaloes": rng.integers(5, 150, n),
            "fodder_price_rs": rng.uniform(10, 40, n).round(1),
            "temp_c": rng.uniform(5, 45, n).round(1),
            "vet_visits": rng.integers(0, 11, n),
            "feed_kg": rng.integers(100, 3000, n),
            "breed_mix": rng.choice(["desi", "crossbred", "mixed"], n),
            "village_cluster": rng.choice(["Cluster A", "Cluster B", "Cluster C"], n),
        }
    )
    df["litres_per_day"] = (2.0 * df.cows + 1.5 * df.buffaloes + rng.normal(0, 5, n)).round(1)
    df["litres_paid_in_passbook"] = df["litres_per_day"] + 1.0
    df.loc[3, "fodder_price_rs"] = -999  # sentinel_value
    df = pd.concat([df, df.iloc[[5, 6]]], ignore_index=True)  # duplicate_rows: 2 extra copies
    return df
