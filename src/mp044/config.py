"""Reading settings and fingerprinting data files."""
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_params(path=None):
    """Read params.yaml (repo root by default) with yaml.safe_load."""
    path = Path(path) if path else ROOT / "params.yaml"
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def file_sha256(path):
    """SHA-256 of a file's raw bytes. Logged in every MLflow run as data_sha256."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
