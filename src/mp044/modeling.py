"""Model pipeline and validation helpers."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODELS = {
    "linear_regression": LinearRegression,
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def make_estimator(name, hyperparameters, random_state=42):
    cls = MODELS[name]
    kwargs = dict(hyperparameters or {})
    if name in ("random_forest", "gradient_boosting"):
        kwargs.setdefault("random_state", random_state)
    return cls(**kwargs)


def build_pipeline(estimator, numeric, categorical, scale=False):
    """Imputer (+scaler) for numbers, one-hot for text, then the regressor.

    Everything lives in ONE sklearn Pipeline, so the saved model.joblib accepts a
    DataFrame with the 8 raw API fields and does its own preprocessing.
    """
    steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer(
        [
            ("num", Pipeline(steps), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


def time_split(df, date_column, valid_fraction):
    """Oldest rows train, newest rows validate (no peeking into the future)."""
    df = df.sort_values(date_column, kind="stable").reset_index(drop=True)
    cut = int(len(df) * (1 - valid_fraction))
    return df.iloc[:cut], df.iloc[cut:]


def mae(y_true, y_pred):
    """Mean Absolute Error in litres."""
    return float(mean_absolute_error(y_true, np.asarray(y_pred)))
