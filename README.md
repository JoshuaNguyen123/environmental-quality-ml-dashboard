# Air Quality Risk Modeling and Environmental Pattern Analysis

A production-grade machine learning pipeline for urban air pollution analysis,
built as an end-to-end demonstration of applied data science methodology.
The project covers statistical analysis, supervised learning (regression and
classification), unsupervised clustering, neural network benchmarking, and
cost-sensitive threshold optimization, with all results served through
a Streamlit dashboard and exported as a standalone HTML report.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Design Decisions](#design-decisions)
- [Assumptions and Limitations](#assumptions-and-limitations)
- [License](#license)

---

## Project Overview

This project implements a complete ML workflow on the UCI Air Quality dataset,
which contains hourly sensor readings from an air quality monitoring station
in an Italian city (March 2004 through February 2005). The pipeline addresses
three core tasks:

- **Regression:** Predict continuous NO2 concentration from sensor and
  meteorological features.
- **Classification:** Detect high-pollution events (NO2 exceeding a
  configurable regulatory threshold) as a binary classification problem.
- **Clustering:** Discover latent environmental regimes (e.g., rush-hour
  pollution peaks, overnight baselines, seasonal patterns) via unsupervised
  learning.

Each task is implemented with multiple model families, evaluated with
appropriate metrics, and benchmarked against a feedforward neural network.

---

## Dataset

**Source:** UCI Machine Learning Repository -- Air Quality Dataset
(https://archive.ics.uci.edu/ml/datasets/Air+Quality)

**Description:** Hourly averaged responses from 5 metal-oxide chemical
sensors deployed at road level in a polluted urban area. Ground-truth
concentrations for CO, NOx, and NO2 are provided by a co-located reference
analyzer.

**Key variables:**

| Variable       | Description                              | Unit      |
|----------------|------------------------------------------|-----------|
| CO(GT)         | True CO concentration                    | mg/m3     |
| NOx(GT)        | True NOx concentration                   | ug/m3 ppb |
| NO2(GT)        | True NO2 concentration (regression target)| ug/m3    |
| PT08.S1--S5    | Metal-oxide sensor responses             | (unitless)|
| Temperature    | Ambient temperature                      | Celsius   |
| Rel_Humidity   | Relative humidity                        | %         |
| Abs_Humidity   | Absolute humidity                        | g/m3      |

**Note on synthetic fallback:** When the real UCI CSV is not present in
`data/raw/`, the pipeline generates a statistically faithful synthetic
dataset that preserves the original's marginal distributions, temporal
patterns, and cross-variable correlations. This allows the full pipeline
to run without external downloads. To use real data, place the UCI CSV at
`data/raw/air_quality.csv`.

---

## Methodology

### Statistical Analysis

- Descriptive statistics (mean, variance, skewness, kurtosis) for all
  numeric features.
- Pearson correlation matrix with lower-triangle heatmap visualization.
- Welch's two-sample t-test comparing NO2 on high- vs low-temperature days.
- One-way ANOVA testing NO2 differences across four seasonal groups.
- Chi-square test of independence between pollution threshold crossings
  and humidity buckets.

All hypothesis tests explicitly document the violation of independence
assumptions inherent in autocorrelated time-series data.

### Supervised Learning

**Regression models (target: NO2 concentration):**

| Model             | Method                               |
|-------------------|--------------------------------------|
| Linear Regression | OLS with StandardScaler              |
| Random Forest     | 200-tree ensemble with max_depth=15  |

Evaluation: RMSE, MAE, R-squared. Diagnostics include residual-vs-fitted
plots, Q-Q plots, Variance Inflation Factor, Shapiro-Wilk normality test,
and Durbin-Watson autocorrelation statistic.

**Classification models (target: NO2 > regulatory threshold):**

| Model             | Method                               |
|-------------------|--------------------------------------|
| Logistic Regression | L2-regularized with LBFGS solver  |
| Random Forest     | 200-tree ensemble with max_depth=15  |

Evaluation: ROC-AUC, PR-AUC, F1, precision, recall, log loss, confusion
matrices, and calibration curves.

### Unsupervised Learning

| Model                 | Objective                           |
|-----------------------|-------------------------------------|
| K-Means (k=4)        | Minimize within-cluster variance    |
| Gaussian Mixture Model| Maximum likelihood for mixture of Gaussians |

Evaluation: Silhouette Score, Davies-Bouldin Index. Visualization via
t-SNE projection. Cluster profiles show normalized feature means per
cluster.

### Deep Learning Benchmark

A feedforward MLP (128-64 hidden units, ReLU activation, Adam optimizer)
is trained for both regression and classification tasks with manual
early stopping on a held-out validation set. Training curves (loss vs
epoch) are recorded and exported.

### Threshold Optimization

Decision thresholds are swept from 0.01 to 0.99 on classification
probabilities. At each threshold, F1 score and a cost-benefit expected
value are computed:

    Expected Value = TP * Benefit - FP * Cost - FN * Cost_miss

The optimal thresholds for F1 and expected value are reported and
visualized.

---

## Results Summary

Results below are from the synthetic dataset. Performance on the real UCI
data will differ.

**Regression (test set):**

| Model             | RMSE  | MAE   | R-squared |
|-------------------|-------|-------|-----------|
| Linear Regression | 25.3  | 20.0  | 0.691     |
| Random Forest     | 25.7  | 20.3  | 0.681     |
| MLP Neural Net    | 25.7  | 20.3  | 0.681     |

**Classification (test set):**

| Model             | ROC-AUC | F1    | Precision | Recall |
|-------------------|---------|-------|-----------|--------|
| Logistic Regression | 0.923 | 0.623 | 0.745     | 0.535  |
| Random Forest     | 0.917   | 0.606 | 0.758     | 0.504  |
| MLP Neural Net    | 0.914   | 0.636 | 0.707     | 0.578  |

**Clustering:**

| Model   | Silhouette Score | Davies-Bouldin Index |
|---------|------------------|----------------------|
| K-Means | 0.154            | 1.72                 |
| GMM     | 0.140            | 1.75                 |

---

## Repository Structure

```
air-quality-ml-dashboard/
|-- README.md
|-- LICENSE
|-- .gitignore
|-- pyproject.toml
|-- requirements.txt
|-- Makefile
|-- scripts/
|   |-- fetch_data.py          # Download or generate dataset
|   |-- train_all.py           # Run full ML pipeline
|   |-- build_report.py        # Generate HTML report
|   +-- run_app.py             # Launch Streamlit dashboard
|-- configs/
|   |-- project.yaml           # Paths, column names, random seed
|   |-- models.yaml            # Hyperparameters for all models
|   +-- thresholds.yaml        # Regulatory thresholds, cost-benefit params
|-- data/
|   |-- raw/                   # Original dataset (gitignored)
|   |-- interim/               # Cleaned intermediate data
|   +-- processed/             # Feature-engineered train/val/test splits
|-- artifacts/
|   |-- metrics/               # JSON metric files
|   |-- figures/               # Publication-quality PNG plots
|   |-- models/                # Serialized model files (joblib)
|   |-- tables/                # CSV comparison tables
|   +-- reports/               # HTML report
|-- src/
|   |-- data/
|   |   |-- ingest.py          # Data loading and synthetic generation
|   |   |-- clean.py           # Missing value handling, datetime parsing
|   |   +-- split.py           # Chronological train/val/test split
|   |-- features/
|   |   |-- engineer.py        # Temporal, lag, and rolling features
|   |   +-- preprocessing.py   # StandardScaler pipeline
|   |-- stats/
|   |   |-- descriptive.py     # Moments and correlations
|   |   |-- hypothesis.py      # t-test, ANOVA, Chi-square
|   |   +-- diagnostics.py     # VIF, normality tests, Durbin-Watson
|   |-- models/
|   |   |-- supervised.py      # Linear/Logistic Regression, Random Forest
|   |   |-- unsupervised.py    # K-Means, GMM, t-SNE
|   |   |-- deep_learning.py   # MLP regressor and classifier
|   |   +-- evaluation.py      # Metrics, ROC/PR curves, calibration
|   |-- business/
|   |   +-- threshold_analysis.py  # Cost-benefit threshold sweep
|   +-- utils/
|       |-- io.py              # Config loading, artifact I/O
|       +-- plotting.py        # Matplotlib/seaborn figure generation
+-- app/
    |-- app.py                 # Streamlit dashboard (6 sections)
    +-- style.css              # Dashboard styling
```

---

## Setup and Installation

**Requirements:** Python 3.10 or later. On Windows, prefer [python.org](https://www.python.org/downloads/) CPython if you hit install or SSL errors (e.g. when using MSYS2/MinGW Python).

```bash
git clone https://github.com/<your-username>/air-quality-ml-dashboard.git
cd air-quality-ml-dashboard

python -m venv .venv
# Activate the venv (pick one; in PowerShell do NOT use "source"):
#   macOS / Linux / Git Bash (or Windows venv with bin/):
source .venv/bin/activate
#   Windows PowerShell with Scripts/:
.\.venv\Scripts\Activate.ps1
#   Windows PowerShell with bin/ (e.g. Unix-style venv):
.\.venv\bin\Activate.ps1

# Install dependencies (the -r flag is required)
pip install -r requirements.txt
```

**If `pip install` fails** (e.g. "Downloading numpy-…tar.gz", SSL errors, or "Failed building wheel for numpy/cmake"): your Python may be one that has no pre-built wheels (e.g. MSYS2/MinGW). Create the venv with **CPython from [python.org](https://www.python.org/downloads/)** instead: remove `.venv`, then run `python -m venv .venv` using the python.org interpreter (or `py -3.10 -m venv .venv` if you use the Windows launcher). Then activate and run `pip install -r requirements.txt` again.

**Optional:** To use the real UCI dataset, download it from the UCI
repository and place the CSV at `data/raw/air_quality.csv`. If this file
is absent, the pipeline will generate synthetic data automatically.

---

## Usage

### Run the full pipeline

```bash
python scripts/train_all.py
```

This executes all seven stages (data ingestion through threshold analysis)
and writes outputs to `artifacts/`.

### Generate the HTML report

```bash
python scripts/build_report.py
```

Opens or serves `artifacts/reports/final_report.html`.

### Launch the Streamlit dashboard

```bash
streamlit run app/app.py
```

The dashboard has seven sections: Executive Summary, Statistical Overview,
Supervised Models, Unsupervised Regimes, Deep Learning Benchmark, Policy
Implications, and Live Weather (API).

#### Live Weather setup

The Live Weather tab uses Open-Meteo (keyless) for hourly and daily forecast data.
You can adjust the refresh cadence (1–5 minutes) directly in the app.

### Makefile shortcuts

```bash
make train    # Run full pipeline
make report   # Generate HTML report
make app      # Launch Streamlit dashboard
make clean    # Remove all generated artifacts
```

---

## Pipeline Stages

1. **Data Ingestion and Cleaning** -- Load the UCI CSV (or generate
   synthetic data), replace sentinel missing values (-200) with NaN,
   drop columns with over 50% missing data, interpolate short gaps,
   and parse timestamps.

2. **Feature Engineering** -- Extract temporal features (hour, day of week,
   month, season, weekend flag), cyclical hour encoding (sin/cos), lag
   features (1h, 3h, 6h), and shifted rolling means (3h, 6h). Define the
   binary classification target based on the configured NO2 threshold.

3. **Chronological Splitting** -- Split data 70/15/15 in temporal order
   (no shuffling) to prevent data leakage from future observations.

4. **Statistical Analysis** -- Compute descriptive statistics and
   correlation matrices. Run three hypothesis tests (t-test, ANOVA,
   Chi-square) with documented assumption violations.

5. **Supervised Model Training** -- Train regression models (Linear
   Regression, Random Forest) and classification models (Logistic
   Regression, Random Forest). Evaluate on the held-out test set.

6. **Unsupervised Clustering** -- Fit K-Means and GMM on sensor and
   environmental features. Evaluate with Silhouette and Davies-Bouldin
   metrics. Visualize with t-SNE.

7. **Deep Learning Benchmark and Threshold Analysis** -- Train MLP models
   for both tasks. Sweep classification thresholds to optimize F1 and
   cost-benefit expected value.

---

## Design Decisions

**Chronological splits over random splits.** Random train/test splitting
on time-series data causes future information to leak into the training
set, producing overly optimistic metrics. This pipeline uses strict
temporal ordering.

**Lag features shifted by 1 step.** Rolling means are shifted by one
timestep to ensure they contain only past information at prediction time,
avoiding target leakage.

**Welch's t-test over Student's t-test.** Welch's variant does not assume
equal variance between groups, making it more robust for environmental
data where variance may differ across conditions.

**Permutation importance over tree-based importance.** Tree-based feature
importance is biased toward high-cardinality features. Permutation
importance (available but not run by default for speed) provides an
unbiased model-agnostic alternative.

**sklearn MLP over PyTorch.** The neural network benchmark uses sklearn's
MLPRegressor/MLPClassifier to minimize dependencies. The architecture
(128-64, ReLU, Adam, early stopping) is functionally equivalent to a
PyTorch implementation.

**All configs externalized to YAML.** Hyperparameters, thresholds, and
path definitions live in `configs/` rather than being hardcoded, making
experiments reproducible and auditable.

---

## Assumptions and Limitations

- **Temporal autocorrelation.** Consecutive hourly observations are not
  independent. This violates assumptions in all three hypothesis tests
  and means that cross-validated classification metrics may be slightly
  optimistic despite chronological splitting. Each test result includes
  an explicit caveat noting this.

- **Single monitoring site.** All data comes from one station in one
  Italian city. Model generalization to other cities, climates, or sensor
  configurations is not established.

- **Sensor drift.** Metal-oxide sensors are known to drift over months
  of deployment. The 12-month observation window may capture drift effects
  that confound temporal patterns.

- **Synthetic data caveat.** When run without the real UCI CSV, the pipeline
  uses generated data that approximates but does not replicate the true
  joint distribution. Results on synthetic data are illustrative, not
  definitive.

- **No hyperparameter tuning.** Model hyperparameters are set via
  configuration rather than searched. A production system would use
  time-series-aware cross-validation (e.g., expanding window) for
  hyperparameter optimization.

---

## License

MIT -- see [LICENSE](LICENSE) for details.
