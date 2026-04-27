#!/usr/bin/env python3
"""fetch_data.py — Download or generate the UCI Air Quality dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.ingest import load_or_generate
from src.utils.io import ensure_dirs, load_project_config
from src.utils.logging import setup_logging

if __name__ == "__main__":
    log = setup_logging()
    cfg = load_project_config()
    ensure_dirs()
    raw_path = ROOT / cfg["paths"]["raw_data"]
    df = load_or_generate(raw_path)
    log.info("fetch_data: loaded %d rows -> %s", len(df), raw_path)
