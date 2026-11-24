"""Cleaning rules for the data problems found in Stage 1 (see data_audit.py).

Each rule is a small function so it can be tested alone.
Training data gets every rule; new data (later.csv, holdout, API requests)
only gets the rules that make sense for rows without a target.
"""
import numpy as np
import pandas as pd


def drop_leaking_columns(df, leaking_columns):
    """leaking_column: remove columns that are only known after the event."""
    return df.drop(columns=[c for c in leaking_columns if c in df.columns])


def remove_duplicates(df):
    """duplicate_rows: keep the first copy of every identical row."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_text(df, strip_columns, lowercase_columns):
    """Defensive text clean-up (trailing spaces, upper/lower case)."""
    df = df.copy()
    for col in strip_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    for col in lowercase_columns:
        if col in df.columns:
            df[col] = df[col].astype("string").str.lower()
    return df


def replace_sentinel(df, column, value):
    """sentinel_value: the typed-in number (e.g. -999) really means 'missing'."""
    df = df.copy()
    if column in df.columns:
        df[column] = df[column].replace(value, np.nan)
    return df


def blank_stuck_sensor(df, column, start_date, end_date, date_column="date"):
    """stuck_sensor: readings inside the stuck window are not real; make them missing.

    The model pipeline fills the gaps (median imputation), so we keep the rows.
    """
    df = df.copy()
    if column in df.columns and date_column in df.columns:
        in_window = (df[date_column] >= start_date) & (df[date_column] <= end_date)
        df[column] = df[column].astype("float64")
        df.loc[in_window, column] = np.nan
    return df


def clean_features(df, params):
    """Rules safe for ANY data with the feature columns (no target needed)."""
    c = params["cleaning"]
    df = clean_text(df, c["strip_text"], c["lowercase_text"])
    return replace_sentinel(df, c["sentinel"]["column"], c["sentinel"]["value"])


def clean_training_data(train, params):
    """Full cleaning of train.csv. Returns a new frame; the input is not changed."""
    c = params["cleaning"]
    df = train.copy()
    if c.get("drop_duplicates", True):
        df = remove_duplicates(df)
    df = clean_features(df, params)
    s = c["stuck_sensor"]
    df = blank_stuck_sensor(
        df, s["column"], s["start_date"], s["end_date"], params["data"]["date_column"]
    )
    return df


def split_xy(df, params):
    """Return (X, y) using only the feature columns the API will receive."""
    d = params["data"]
    features = d["numeric"] + d["categorical"]
    return df[features], df[d["target"]]
