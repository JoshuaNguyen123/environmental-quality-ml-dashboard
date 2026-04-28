.PHONY: setup demo verify data train report app clean all

# ── Demo / presentation ──────────────────────────────────────────────────
setup:
	python -m pip install --upgrade pip
	pip install -r requirements.txt

demo: app

verify:
	python -m ruff check .
	python -m pytest -q -m "not slow"

# ── Pipeline ─────────────────────────────────────────────────────────────
all: data train report

data:
	python scripts/fetch_data.py

train:
	python scripts/train_all.py

report:
	python scripts/build_report.py

app:
	streamlit run app/app.py

clean:
	rm -rf data/interim/* data/processed/* artifacts/models/*
	find . -name "__pycache__" -type d -exec rm -rf {} +
