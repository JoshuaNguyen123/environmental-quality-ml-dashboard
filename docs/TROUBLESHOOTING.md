# Troubleshooting

## Installation

### `pip install` fails with "Failed building wheel for numpy/cmake" or SSL errors

You're likely on a Python build that doesn't have pre-built wheels for the
numerical stack — most commonly MSYS2 / MinGW Python on Windows.

**Fix:** Install the official CPython distribution from
[python.org](https://www.python.org/downloads/), then recreate your
virtual environment:

```powershell
# Remove the broken venv first
Remove-Item -Recurse -Force .venv

# Then create a new one with python.org CPython
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you have multiple Python installations, the Windows launcher selects
a specific version: `py -3.11 -m venv .venv`.

### Virtual environment activation differs by shell

| Shell | Command |
|---|---|
| macOS / Linux / Git Bash | `source .venv/bin/activate` |
| Windows PowerShell (Scripts/) | `.\.venv\Scripts\Activate.ps1` |
| Windows PowerShell (bin/) | `.\.venv\bin\Activate.ps1` |
| Windows cmd.exe | `.venv\Scripts\activate.bat` |

Do **not** use `source` in PowerShell.

## Running the dashboard

### "Metric file not found" warnings

The dashboard reads pre-computed artifacts from `artifacts/`. If you
removed them or are running an old commit, regenerate:

```bash
python scripts/train_all.py
```

Pre-built artifacts ship in the repo, so this usually means a previous
`make clean` removed them or the working tree is partially built.

### Streamlit shows "Error: SchemaError"

You placed a real UCI CSV at `data/raw/air_quality.csv` but the column
names don't match what the pipeline expects. The error message lists the
missing columns. Fix the CSV header or delete the file to use the
synthetic fallback.

### "Weather service is temporarily unavailable" on the Live Weather tab

The Open-Meteo API is rate-limited and occasionally times out. Click
**Refresh** after 30 seconds. The keyless free tier permits ~10,000 calls
per day per IP, which is more than enough for development.

### Streamlit shows the topbar but the body is blank

Hard-refresh the browser (Cmd/Ctrl + Shift + R). Streamlit caches
aggressively across reloads.

## Pipeline

### Training is slow on a fresh machine

The 7-stage pipeline runs in ~30–60s on a recent laptop. On underpowered
hardware, the bottlenecks are:

1. t-SNE projection (~5–10s on 9k rows)
2. MLP training (3 architectures × ~30 epochs)

Both are configurable in `configs/models.yaml` if you need a faster turn.

### "DBSCAN: noise_ratio = 1.0"

DBSCAN's `eps=0.85` is tuned for the real UCI distribution. On the
synthetic dataset, the cluster geometry differs slightly and DBSCAN may
flag every point as noise — this is expected behaviour and the dashboard
displays an explanatory note.

## CI

### `ruff check .` fails locally but passes in CI (or vice versa)

Make sure the local ruff version matches the version in
`requirements.txt`. New rules are added in minor releases.

```bash
pip install -U "ruff>=0.5"
```

### Tests pass locally but fail in the GitHub `smoke` job

The smoke job runs only on PRs to `main` and pushes to `main`. The
synthetic data generator is seeded, so any genuine difference points at a
non-deterministic dependency upgrade. Run `pytest -m slow -v` locally
against your venv to reproduce.
