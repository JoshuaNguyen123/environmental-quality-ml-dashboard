#!/usr/bin/env python3
"""run_app.py — Launch the Streamlit dashboard."""

import subprocess
import sys

if __name__ == "__main__":
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app/app.py"], check=True)
