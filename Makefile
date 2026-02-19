.PHONY: data train report app clean all

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
	rm -rf data/interim/* data/processed/* artifacts/metrics/* artifacts/figures/* artifacts/models/* artifacts/tables/* artifacts/reports/*
	find . -name "__pycache__" -type d -exec rm -rf {} +
