from pathlib import Path

import pandas as pd

from src.utils import io


def test_save_and_load_json_roundtrip(tmp_path: Path):
    target = tmp_path / "nested" / "payload.json"
    payload = {"a": 1, "b": "x", "c": [1, 2, 3]}
    io.save_json(payload, target)
    assert target.exists()
    assert io.load_json(target) == payload


def test_save_csv_creates_parent_dir(tmp_path: Path):
    target = tmp_path / "artifacts" / "sample.csv"
    df = pd.DataFrame({"x": [1, 2], "y": [3, 4]})
    io.save_csv(df, target)
    loaded = io.load_csv(target)
    assert loaded.shape == (2, 2)
    assert list(loaded.columns) == ["x", "y"]


def test_project_config_has_required_paths():
    cfg = io.load_project_config()
    assert isinstance(cfg, dict)
    assert "paths" in cfg
    for key in ("metrics", "figures", "models", "tables", "reports"):
        assert key in cfg["paths"]
