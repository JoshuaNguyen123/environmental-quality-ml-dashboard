# Model Card — Air Quality Risk Models

This dashboard ships eleven trained models grouped into three tasks:
regression on hourly NO₂ concentration, binary classification of
"high pollution" hours, and unsupervised clustering of environmental
regimes. This card summarises intended use, training data, and headline
performance for the models the dashboard surfaces.

## Intended use

| Audience | Use |
|---|---|
| Public health analysts | Explore trade-offs between false alarms and missed exceedances on hourly NO₂ data |
| ML interviewers / reviewers | Evaluate end-to-end ML methodology — chronological splits, leakage-free features, calibrated classifiers, cost-sensitive thresholds |
| Educators | Walk students through a complete supervised + unsupervised + neural pipeline with statistical diagnostics |

**Out of scope.** These models are **not** intended for operational air-quality decisions, regulatory compliance, or generalisation outside the single Italian monitoring station the data were collected from.

## Training data

- **Source:** UCI Air Quality Dataset (`https://archive.ics.uci.edu/ml/datasets/Air+Quality`) — hourly readings from one station, March 2004 through February 2005, ~9,357 valid hours.
- **Synthetic fallback:** When the real CSV is not present at `data/raw/air_quality.csv`, `src/data/ingest.py:generate_synthetic_air_quality` produces a seeded synthetic dataset that preserves marginal distributions, traffic-driven diurnal patterns, and cross-sensor correlations. **Headline numbers below are computed on the synthetic dataset** (deterministic, seed=42).
- **Split:** Chronological 70 / 15 / 15 (train / validation / test) — no shuffling. Lag features and rolling means are shifted by one step to prevent target leakage.

## Inputs

- **Sensor readings:** PT08.S1 through PT08.S5 metal-oxide responses
- **Reference analyser:** CO_GT, NOx_GT (NO₂_GT is the regression target)
- **Meteorology:** Temperature, Relative Humidity, Absolute Humidity
- **Engineered:** Hour, DayOfWeek, Month, Season, IsWeekend, sin/cos of hour, NO₂ lag (1h, 3h, 6h), NO₂ rolling means (3h, 6h, both shifted by 1)

## Outputs

- **Regression:** Predicted NO₂ concentration (µg/m³)
- **Classification:** Probability that NO₂ exceeds the configured regulatory threshold (default 120 µg/m³, see `configs/thresholds.yaml`)
- **Clustering:** Cluster assignment (0–3) for each hourly observation

## Performance (test set, synthetic data, seed=42)

### Regression — predict NO₂ concentration

| Model | RMSE (µg/m³) | MAE (µg/m³) | R² |
|---|---|---|---|
| Linear Regression | 25.27 | 19.97 | 0.691 |
| Random Forest | 25.66 | 20.31 | 0.682 |
| Extra Trees | 25.47 | 20.23 | 0.686 |
| HistGradientBoosting | 26.15 | 20.63 | 0.670 |
| MLP (128-64) | 25.65 | 20.31 | 0.682 |

### Classification — flag NO₂ > 120 µg/m³

| Model | ROC-AUC | F1 | Precision | Recall |
|---|---|---|---|---|
| Logistic Regression | 0.923 | 0.623 | 0.745 | 0.535 |
| Random Forest | 0.917 | 0.603 | 0.748 | 0.504 |
| Extra Trees | 0.915 | 0.563 | 0.757 | 0.448 |
| HistGradientBoosting | 0.908 | 0.558 | 0.688 | 0.470 |
| MLP (128-64) | 0.915 | 0.640 | 0.709 | 0.583 |

### Clustering — discover environmental regimes

| Model | Silhouette ↑ | Davies-Bouldin ↓ | Clusters |
|---|---|---|---|
| K-Means | 0.154 | 1.72 | 4 |
| Gaussian Mixture | 0.140 | 1.75 | 4 |
| Agglomerative | 0.103 | 1.85 | 4 |
| DBSCAN | n/a (all points labelled noise) | — | 0 |

### Threshold optimisation

Sweeping decision thresholds 0.01 – 0.99 with the cost-benefit weights in
`configs/thresholds.yaml` (TP benefit = 10, FP cost = 3, FN cost = 15)
yields:

- **Best F1 threshold:** 0.337
- **Best expected-value threshold:** 0.178

The dashboard's "Policy Implications" tab visualises this curve.

## Caveats and known limitations

- **Temporal autocorrelation.** Consecutive hourly observations are not independent. This violates assumptions of the t-test, ANOVA, and chi-square tests reported in the dashboard. Each test result includes an explicit caveat.
- **Single monitoring site.** All training data come from one station in one Italian city. Generalisation to other locations, climates, or sensor configurations is not established.
- **Sensor drift.** Metal-oxide sensors drift over months. The 12-month observation window may capture drift that confounds genuine temporal patterns.
- **Synthetic data caveat.** The headline numbers above are on synthetic data. Numbers on the real UCI CSV will differ — re-run `make train` with the real CSV at `data/raw/air_quality.csv` to obtain those.
- **No hyperparameter search.** Hyperparameters are set in `configs/models.yaml` rather than searched. A production deployment would use time-series-aware cross-validation (e.g., expanding-window CV).

## Reproducing these numbers

```bash
make setup
make train          # ~30–60s
cat artifacts/metrics/supervised_metrics.json
```

Random seed = 42 throughout (`configs/project.yaml`, `configs/models.yaml`).
Numbers are stable across runs given the same seed.
