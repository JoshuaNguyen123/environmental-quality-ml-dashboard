#!/usr/bin/env python3
"""build_report.py — Generate HTML report from pipeline artifacts."""

import sys
import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.io import load_json
import pandas as pd


def img_to_base64(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def build_html_report():
    FIGURES = ROOT / "artifacts" / "figures"
    METRICS = ROOT / "artifacts" / "metrics"
    TABLES = ROOT / "artifacts" / "tables"
    OUTPUT = ROOT / "artifacts" / "reports"
    OUTPUT.mkdir(parents=True, exist_ok=True)

    sup = load_json(METRICS / "supervised_metrics.json") if (METRICS / "supervised_metrics.json").exists() else {}
    deep = load_json(METRICS / "deep_metrics.json") if (METRICS / "deep_metrics.json").exists() else {}
    unsup = load_json(METRICS / "unsupervised_metrics.json") if (METRICS / "unsupervised_metrics.json").exists() else {}
    stats = load_json(METRICS / "stats_summary.json") if (METRICS / "stats_summary.json").exists() else {}

    # Model comparison table
    comp_path = TABLES / "model_comparison.csv"
    comp_html = ""
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        comp_html = comp.to_html(index=False, classes="data-table", float_format="%.3f")

    figures = {}
    for name in ["corr_heatmap", "residuals_linear_reg", "roc_curves", "pr_curves",
                  "confusion_matrices", "calibration_curve", "cluster_tsne",
                  "cluster_profiles", "nn_training_curves", "threshold_value_curve"]:
        p = FIGURES / f"{name}.png"
        figures[name] = img_to_base64(p)

    def img_tag(name: str, title: str = "") -> str:
        b64 = figures.get(name, "")
        if not b64:
            return f"<p><em>Figure not available: {name}</em></p>"
        return f'<figure><img src="data:image/png;base64,{b64}" alt="{title}"><figcaption>{title}</figcaption></figure>'

    # Build metrics summary
    lr = sup.get("linear_regression", {})
    rf_reg = sup.get("random_forest_reg", {})
    log = sup.get("logistic_regression", {})
    rf_clf = sup.get("random_forest_clf", {})
    mlp_reg = deep.get("mlp_regression", {})
    mlp_clf = deep.get("mlp_classification", {})
    km = unsup.get("kmeans", {})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Air Quality ML Report</title>
<style>
    body {{ font-family: 'Segoe UI', Tahoma, sans-serif; max-width: 1000px; margin: 2rem auto;
           color: #1F2937; line-height: 1.6; padding: 0 1.5rem; background: #FAFAFA; }}
    h1 {{ color: #111827; border-bottom: 3px solid #2C3E50; padding-bottom: 0.5rem; }}
    h2 {{ color: #2C3E50; margin-top: 2.5rem; border-bottom: 1px solid #D1D5DB; padding-bottom: 0.3rem; }}
    h3 {{ color: #374151; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 1rem; margin: 1.5rem 0; }}
    .metric-card {{ background: white; border: 1px solid #E5E7EB; border-radius: 8px;
                   padding: 1.2rem; text-align: center; }}
    .metric-card .value {{ font-size: 1.8rem; font-weight: 700; color: #2C3E50; }}
    .metric-card .label {{ font-size: 0.85rem; color: #6B7280; margin-top: 0.3rem; }}
    figure {{ margin: 1.5rem 0; text-align: center; }}
    figure img {{ max-width: 100%; border: 1px solid #E5E7EB; border-radius: 6px; }}
    figcaption {{ font-size: 0.85rem; color: #6B7280; margin-top: 0.5rem; }}
    .data-table {{ width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.9rem; }}
    .data-table th {{ background: #2C3E50; color: white; padding: 0.6rem; text-align: left; }}
    .data-table td {{ padding: 0.5rem 0.6rem; border-bottom: 1px solid #E5E7EB; }}
    .data-table tr:nth-child(even) {{ background: #F9FAFB; }}
    .caveat {{ background: #FEF3C7; border-left: 4px solid #F59E0B; padding: 0.8rem 1rem;
              margin: 1rem 0; border-radius: 0 4px 4px 0; font-size: 0.9rem; }}
    .section {{ background: white; padding: 2rem; border-radius: 8px; margin: 1.5rem 0;
               border: 1px solid #E5E7EB; }}
    footer {{ text-align: center; margin-top: 3rem; padding: 1.5rem; color: #9CA3AF;
             font-size: 0.85rem; border-top: 1px solid #E5E7EB; }}
</style>
</head>
<body>

<h1>Air Quality Risk Modeling &amp; Environmental Pattern Analysis</h1>
<p>Comprehensive ML pipeline analysis of urban air pollution data.</p>

<div class="section">
<h2>1. Executive Summary</h2>
<div class="metrics-grid">
    <div class="metric-card"><div class="value">{lr.get('r2', 0):.3f}</div><div class="label">Linear Reg R²</div></div>
    <div class="metric-card"><div class="value">{log.get('roc_auc', 0):.3f}</div><div class="label">Logistic AUC</div></div>
    <div class="metric-card"><div class="value">{km.get('silhouette_score', 0):.3f}</div><div class="label">K-Means Silhouette</div></div>
    <div class="metric-card"><div class="value">{mlp_reg.get('r2', 0):.3f}</div><div class="label">MLP Reg R²</div></div>
</div>

<h3>Model Comparison</h3>
{comp_html}
</div>

<div class="section">
<h2>2. Statistical Analysis</h2>
{img_tag('corr_heatmap', 'Pearson Correlation Matrix')}

<h3>Hypothesis Tests</h3>
<div class="caveat">
<strong>Caveat:</strong> All hypothesis tests assume independent observations.
Hourly time-series data exhibits temporal autocorrelation, so p-values
may be anti-conservative. Results should be interpreted as exploratory.
</div>
</div>

<div class="section">
<h2>3. Regression Models</h2>
<p>Target: NO₂(GT) concentration (µg/m³)</p>
<div class="metrics-grid">
    <div class="metric-card"><div class="value">{lr.get('rmse', 0):.1f}</div><div class="label">Linear Reg RMSE</div></div>
    <div class="metric-card"><div class="value">{rf_reg.get('rmse', 0):.1f}</div><div class="label">RF Reg RMSE</div></div>
    <div class="metric-card"><div class="value">{mlp_reg.get('rmse', 0):.1f}</div><div class="label">MLP Reg RMSE</div></div>
</div>
{img_tag('residuals_linear_reg', 'Linear Regression — Residual Diagnostics')}
</div>

<div class="section">
<h2>4. Classification Models</h2>
<p>Target: High Pollution (NO₂ &gt; threshold)</p>
{img_tag('roc_curves', 'ROC Curves')}
{img_tag('pr_curves', 'Precision-Recall Curves')}
{img_tag('confusion_matrices', 'Confusion Matrices')}
{img_tag('calibration_curve', 'Calibration Curves')}
</div>

<div class="section">
<h2>5. Threshold Optimization</h2>
{img_tag('threshold_value_curve', 'Threshold vs F1 Score & Expected Value')}
</div>

<div class="section">
<h2>6. Unsupervised Clustering</h2>
{img_tag('cluster_tsne', 'Cluster Visualization (t-SNE)')}
{img_tag('cluster_profiles', 'Cluster Feature Profiles')}
</div>

<div class="section">
<h2>7. Deep Learning Benchmark</h2>
{img_tag('nn_training_curves', 'MLP Training Curves')}
</div>

<div class="section">
<h2>8. Limitations</h2>
<ul>
<li><strong>Temporal autocorrelation:</strong> Sequential hourly measurements violate
the independence assumption required by standard statistical tests and naive
cross-validation. We use chronological train/val/test splits to mitigate this.</li>
<li><strong>Single-site generalization:</strong> All data comes from one monitoring
station in an Italian city. Model transferability to other urban environments
is not established.</li>
<li><strong>Sensor drift:</strong> Metal-oxide sensors are known to drift over time.
The 12-month observation window may capture some drift effects.</li>
</ul>
</div>

<footer>
Air Quality ML Dashboard v1.0 — Generated by automated pipeline
</footer>

</body>
</html>"""

    out_path = OUTPUT / "final_report.html"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[report] HTML report → {out_path}")
    return out_path


if __name__ == "__main__":
    build_html_report()
