"""I/O utilities for loading configs, saving artifacts, and managing paths."""

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[2]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_project_config() -> Dict[str, Any]:
    return load_yaml(ROOT / "configs" / "project.yaml")


def load_model_config() -> Dict[str, Any]:
    return load_yaml(ROOT / "configs" / "models.yaml")


def load_threshold_config() -> Dict[str, Any]:
    return load_yaml(ROOT / "configs" / "thresholds.yaml")


def save_json(data: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str | Path) -> dict:
    with open(path) as f:
        return json.load(f)


def save_model(model: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path) -> Any:
    return joblib.load(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def ensure_dirs() -> None:
    """Create all required project directories."""
    cfg = load_project_config()
    for key, p in cfg["paths"].items():
        full = ROOT / p
        if full.suffix:
            full.parent.mkdir(parents=True, exist_ok=True)
        else:
            full.mkdir(parents=True, exist_ok=True)
