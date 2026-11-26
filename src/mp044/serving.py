"""Stage 3: load the saved model and predict. No web code here, so it is easy to test."""
import os
from pathlib import Path

import joblib
import pandas as pd

from mp044.cleaning import clean_features
from mp044.config import ROOT, file_sha256, load_params


class ModelService:
    """The champion pipeline plus the same cleaning that Stage 2 and 4 use."""

    def __init__(self, model_path=None, params=None):
        self.params = params or load_params()
        path = model_path or os.getenv("MODEL_PATH") or ROOT / self.params["paths"]["model_out"]
        self.model_path = Path(path)
        self.model = joblib.load(self.model_path)
        self.model_sha256 = file_sha256(self.model_path)

    @property
    def features(self):
        d = self.params["data"]
        return d["numeric"] + d["categorical"]

    def predict(self, records):
        """records: list of dicts. Returns one float (litres per day) per record."""
        df = pd.DataFrame(records)
        numeric = self.params["data"]["numeric"]
        df[numeric] = df[numeric].astype("float64")  # a null price would otherwise be 'object'
        df = clean_features(df, self.params)
        return [round(float(p), 2) for p in self.model.predict(df[self.features])]
