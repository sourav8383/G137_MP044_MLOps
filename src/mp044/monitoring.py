"""Stage 4 helpers: drift (PSI), per-group error, the v1-vs-v2 check and the decision.

Every function is small and pure so it can be unit-tested without the real CSVs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

EPS = 1e-4  # floor for empty bins so log() and division stay finite


def psi(expected, actual, bins=10):
    """Population Stability Index of `actual` against `expected`.

    Bin edges are the deciles of `expected` (the cleaned training data), the outer
    edges are opened to +/- infinity so out-of-range later values are still counted.
    Missing values are ignored. Rule of thumb: <0.1 stable, 0.1-0.25 moderate,
    >0.25 major drift.
    """
    e = pd.Series(expected).dropna().to_numpy(dtype=float)
    a = pd.Series(actual).dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.histogram(e, bins=edges)[0] / len(e)
    a_pct = np.histogram(a, bins=edges)[0] / len(a)
    e_pct, a_pct = np.clip(e_pct, EPS, None), np.clip(a_pct, EPS, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def psi_table(train, later, columns, bins=10):
    """{column: PSI} for every numeric feature."""
    return {c: round(psi(train[c], later[c], bins), 4) for c in columns}


def most_drifted(psi_values):
    return max(psi_values, key=psi_values.get)


def mae(y_true, y_pred):
    return float(mean_absolute_error(y_true, np.asarray(y_pred)))


def mae_by_group(df, y_pred, group_column, target):
    """{group value: MAE} of predictions `y_pred` (aligned with df rows)."""
    out = {}
    pred = pd.Series(np.asarray(y_pred), index=df.index)
    for g, part in df.groupby(group_column):
        out[str(g)] = round(mae(part[target], pred.loc[part.index]), 4)
    return out


def worst_group(mae_values):
    return max(mae_values, key=mae_values.get)


def split_check(later, date_column, check_fraction=0.3):
    """Oldest (1-f) of later.csv by date -> retraining part, newest f -> check set."""
    later = later.sort_values(date_column, kind="stable").reset_index(drop=True)
    cut = int(len(later) * (1 - check_fraction))
    return later.iloc[:cut], later.iloc[cut:]


def decide(mae_v1, mae_v2):
    """promote v2 only if it is strictly better than v1 on the same check rows."""
    return "promote" if mae_v2 < mae_v1 else "keep"
