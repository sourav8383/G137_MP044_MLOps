
import os
import re
import string
import pandas as pd
import numpy as np


def load_datasets(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    later = pd.read_csv(os.path.join(data_dir, "later.csv"))
    holdout = pd.read_csv(
        os.path.join(data_dir, "holdout_features.csv")
    )
    return train, later, holdout


def clean_training_data(train):
    train_clean = train.drop_duplicates().copy()

    train_clean["fodder_price_rs"] = (
        train_clean["fodder_price_rs"].replace(-999, np.nan)
    )

    return train_clean


def detect_leaking_column(train, holdout):
    column = "litres_paid_in_passbook"
    target = "litres_per_day"

    present = False

    if (
        column in train.columns
        and column not in holdout.columns
        and target in train.columns
    ):
        correlation = train[[column, target]].corr().iloc[0, 1]
        present = bool(abs(correlation) >= 0.99)

    return {
        "present": present,
        "column": column if present else None
    }


def detect_unit_mixup(train):
    numeric_columns = [
        "cows", "buffaloes", "fodder_price_rs",
        "temp_c", "vet_visits", "feed_kg"
    ]

    factors = [10, 12, 60, 100, 1000]

    result = {
        "present": False,
        "column": None,
        "group_value": None,
        "factor": None
    }

    for column in numeric_columns:
        if column not in train.columns:
            continue

        group_stats = train.groupby("village_cluster")[column].agg(
            ["min", "mean", "max"]
        )

        for group in group_stats.index:
            other_groups = group_stats.drop(index=group)

            for statistic in ["min", "mean", "max"]:
                affected_value = group_stats.loc[group, statistic]
                reference_value = other_groups[statistic].median()

                if (
                    pd.isna(affected_value)
                    or pd.isna(reference_value)
                    or affected_value == 0
                    or reference_value == 0
                ):
                    continue

                ratio = affected_value / reference_value

                for factor in factors:
                    if abs(ratio / factor - 1) <= 0.15:
                        return {
                            "present": True,
                            "column": column,
                            "group_value": group,
                            "factor": factor
                        }

                    if abs(ratio * factor - 1) <= 0.15:
                        return {
                            "present": True,
                            "column": column,
                            "group_value": group,
                            "factor": 1 / factor
                        }

    return result


def detect_sentinel_value(train):
    column = "fodder_price_rs"
    sentinel = -999

    counts = train[column].value_counts(dropna=False)
    count = int(counts.get(sentinel, 0))

    return {
        "present": count > 0,
        "column": column if count > 0 else None,
        "value": sentinel if count > 0 else None,
        "count": count if count > 0 else None
    }


def detect_duplicate_rows(train):
    count = int(train.duplicated().sum())

    return {
        "present": count > 0,
        "count": count if count > 0 else None
    }


def detect_stuck_sensor(train):
    columns = [
        "cows", "buffaloes", "fodder_price_rs",
        "temp_c", "vet_visits", "feed_kg"
    ]

    data = train.copy()
    data["date"] = pd.to_datetime(data["date"])

    best_run = None

    for column in columns:
        for cluster, group in data.groupby("village_cluster"):

            group = group.sort_values("date").reset_index(drop=True)

            group["run_id"] = (
                group[column].ne(group[column].shift())
            ).cumsum()

            for _, run in group.groupby("run_id"):
                if run[column].nunique(dropna=False) != 1:
                    continue

                start = run["date"].min()
                end = run["date"].max()
                unique_dates = run["date"].nunique()
                duration = (end - start).days

                if duration >= 14 and unique_dates >= 10:
                    candidate = {
                        "column": column,
                        "start_date": start,
                        "end_date": end,
                        "duration": duration
                    }

                    if (
                        best_run is None
                        or duration > best_run["duration"]
                    ):
                        best_run = candidate

    if best_run is None:
        return {
            "present": False,
            "column": None,
            "start_date": None,
            "end_date": None
        }

    return {
        "present": True,
        "column": best_run["column"],
        "start_date": best_run["start_date"].strftime("%Y-%m-%d"),
        "end_date": best_run["end_date"].strftime("%Y-%m-%d")
    }



def detect_target_typos(train):
    target = "litres_per_day"

    values = (
        train[target]
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )

    result = {
        "present": False,
        "count": None
    }

    if len(values) < 2:
        return result

    previous = values.iloc[:-1].to_numpy()
    current = values.iloc[1:].to_numpy()

    ratios = np.divide(
        current,
        previous,
        out=np.zeros_like(current, dtype=float),
        where=previous != 0
    )

    positions = np.where(ratios >= 10)[0]

    if len(positions) == 0:
        return result

    position = positions[0]

    typo_threshold = values.iloc[position + 1]
    count = int((values >= typo_threshold).sum())

    return {
        "present": count > 0,
        "count": count if count > 0 else None
    }



def normalize_text(value):
    value = str(value).strip().lower()
    value = value.translate(
        str.maketrans("", "", string.punctuation)
    )
    return re.sub(r"\s+", " ", value)


def detect_spelling_variants(train, later):
    excluded_columns = ["record_id", "date"]

    text_columns = train.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:
        if column in excluded_columns or column not in later.columns:
            continue

        train_values = train[column].dropna().astype(str)
        later_values = later[column].dropna().astype(str)

        clean_reference = {
            normalize_text(value): value
            for value in later_values.unique()
        }

        normalized_train = train_values.map(normalize_text)

        for normalized_value in normalized_train.unique():
            matching_values = train_values[
                normalized_train == normalized_value
            ].unique()

            if (
                len(matching_values) > 1
                and normalized_value in clean_reference
            ):
                return {
                    "present": True,
                    "column": column,
                    "true_levels": len(clean_reference)
                }

    return {
        "present": False,
        "column": None,
        "true_levels": None
    }


def audit_data_issues(train, later, holdout):
    findings = {
        "leaking_column": detect_leaking_column(train, holdout),
        "unit_mixup": detect_unit_mixup(train),
        "sentinel_value": detect_sentinel_value(train),
        "duplicate_rows": detect_duplicate_rows(train),
        "stuck_sensor": detect_stuck_sensor(train),
        "target_typos": detect_target_typos(train),
        "spelling_variants": detect_spelling_variants(train, later)
    }

    return findings
