import numpy as np

from mp044 import cleaning


def test_remove_duplicates_drops_only_extra_copies(tiny_train):
    out = cleaning.remove_duplicates(tiny_train)
    assert len(tiny_train) - len(out) == 2
    assert not out.duplicated().any()


def test_sentinel_becomes_missing(tiny_train):
    out = cleaning.replace_sentinel(tiny_train, "fodder_price_rs", -999)
    assert (out["fodder_price_rs"] == -999).sum() == 0
    assert out["fodder_price_rs"].isna().sum() == 1


def test_stuck_window_is_blanked_only_inside_dates(tiny_train):
    out = cleaning.blank_stuck_sensor(tiny_train, "cows", "2026-02-10", "2026-02-15")
    inside = (tiny_train["date"] >= "2026-02-10") & (tiny_train["date"] <= "2026-02-15")
    assert out.loc[inside, "cows"].isna().all()
    assert out.loc[~inside, "cows"].notna().all()


def test_clean_training_data_applies_every_rule(tiny_train, params):
    out = cleaning.clean_training_data(tiny_train, params)
    assert len(out) == len(tiny_train) - 2                    # duplicates gone
    assert (out["fodder_price_rs"] == -999).sum() == 0        # sentinel gone
    assert out["fodder_price_rs"].isna().sum() == 1
    assert len(tiny_train) == 62                              # input frame untouched


def test_split_xy_never_returns_the_leaking_column(tiny_train, params):
    X, y = cleaning.split_xy(tiny_train, params)
    assert "litres_paid_in_passbook" not in X.columns
    assert "litres_per_day" not in X.columns
    assert list(X.columns) == params["data"]["numeric"] + params["data"]["categorical"]
    assert y.name == "litres_per_day"


def test_clean_text_strips_and_lowercases():
    import pandas as pd

    df = pd.DataFrame({"breed_mix": [" Desi ", "MIXED"], "village_cluster": ["Cluster A ", "Cluster B"]})
    out = cleaning.clean_text(df, ["breed_mix", "village_cluster"], ["breed_mix"])
    assert out["breed_mix"].tolist() == ["desi", "mixed"]
    assert out["village_cluster"].tolist() == ["Cluster A", "Cluster B"]
