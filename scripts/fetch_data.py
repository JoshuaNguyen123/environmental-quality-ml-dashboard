#!/usr/bin/env python3
"""fetch_data.py — Download or generate the UCI Air Quality dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_project_config, ensure_dirs
from src.data.ingest import load_or_generate

if __name__ == "__main__":
    cfg = load_project_config()
    ensure_dirs()
    raw_path = ROOT / cfg["paths"]["raw_data"]
    df = load_or_generate(raw_path)
    print(f"[fetch_data] Loaded {len(df)} rows → {raw_path}")
