"""Air Quality Risk Modeling — Streamlit Dashboard.

Run: streamlit run app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import streamlit as st
import pandas as pd
import numpy as np

from src.utils.io import load_json

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ──────────────────────────────────────────────────────
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Paths ────────────────────────────────────────────────────────────────
FIGURES = ROOT / "artifacts" / "figures"
METRICS = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "artifacts" / "tables"


def load_metric_file(name: str) -> dict:
    try:
        return load_json(METRICS / name)
    except FileNotFoundError:
        st.warning(f"Metric file not found: {name}. Run `python scripts/train_all.py` first.")
        return {}


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════
st.sidebar.title("Navigation")
section = st.sidebar.radio(
    "Go to",
    [
        "Executive Summary",
        "Statistical Overview",
        "Supervised Models",
        "Unsupervised Regimes",
        "Deep Learning Benchmark",
        "Policy Implications",
    ],
)


# ═════════════════════════════════════════════════════════════════════════
# 1. EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════
if section == "Executive Summary":
    st.title("Air Quality Risk Modeling & Environmental Pattern Analysis")
    st.markdown("---")

    st.markdown("""
    **Project scope:** End-to-end machine learning pipeline for urban air pollution
    analysis using the UCI Air Quality dataset (hourly sensor readings from an
    Italian city, March 2004 – February 2005).

    **Objectives:**
    - Predict NO₂ concentration (regression) and detect high-pollution events (classification)
    - Discover latent environmental regimes via clustering
    - Benchmark classical ML vs neural network performance
    - Optimize decision thresholds for cost-sensitive pollution alerting
    """)

    # Key metrics
    sup = load_metric_file("supervised_metrics.json")
    deep = load_metric_file("deep_metrics.json")
    unsup = load_metric_file("unsupervised_metrics.json")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        r2 = sup.get("linear_regression", {}).get("r2", "—")
        st.metric("Linear Reg R²", f"{r2:.3f}" if isinstance(r2, float) else r2)
    with col2:
        auc = sup.get("logistic_regression", {}).get("roc_auc", "—")
        st.metric("Logistic AUC", f"{auc:.3f}" if isinstance(auc, float) else auc)
    with col3:
        sil = unsup.get("kmeans", {}).get("silhouette_score", "—")
        st.metric("K-Means Silhouette", f"{sil:.3f}" if isinstance(sil, float) else sil)
    with col4:
        mlp_r2 = deep.get("mlp_regression", {}).get("r2", "—")
        st.metric("MLP Reg R²", f"{mlp_r2:.3f}" if isinstance(mlp_r2, float) else mlp_r2)

    # Model comparison table
    comp_path = TABLES / "model_comparison.csv"
    if comp_path.exists():
        st.subheader("Model Comparison")
        comp = pd.read_csv(comp_path)
        st.dataframe(comp.style.format(precision=3), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# 2. STATISTICAL OVERVIEW
# ═════════════════════════════════════════════════════════════════════════
elif section == "Statistical Overview":
    st.title("Statistical Analysis")
    st.markdown("---")

    stats = load_metric_file("stats_summary.json")

    # Descriptive stats
    if "column_statistics" in stats:
        st.subheader("Descriptive Statistics")
        desc_df = pd.DataFrame(stats["column_statistics"]).T
        st.dataframe(desc_df.style.format(precision=3), use_container_width=True)

    # Correlation heatmap
    fig_path = FIGURES / "corr_heatmap.png"
    if fig_path.exists():
        st.subheader("Pearson Correlation Matrix")
        st.image(str(fig_path), use_container_width=True)

    # Hypothesis tests
    if "hypothesis_tests" in stats:
        st.subheader("Hypothesis Tests")
        ht = stats["hypothesis_tests"]

        for test_name, result in ht.items():
            with st.expander(f"**{result.get('test', test_name)}**"):
                st.write(f"**H₀:** {result.get('null_hypothesis', '')}")
                for k, v in result.items():
                    if k not in ("test", "null_hypothesis", "contingency_table"):
                        st.write(f"- **{k}:** {v}")


# ═════════════════════════════════════════════════════════════════════════
# 3. SUPERVISED MODELS
# ═════════════════════════════════════════════════════════════════════════
elif section == "Supervised Models":
    st.title("Supervised Learning Results")
    st.markdown("---")

    sup = load_metric_file("supervised_metrics.json")

    # Regression
    st.subheader("Regression — NO₂ Prediction")
    reg_models = {k: v for k, v in sup.items() if "regression" in k.lower() and "logistic" not in k.lower()}
    if reg_models:
        cols = st.columns(len(reg_models))
        for col, (name, metrics) in zip(cols, reg_models.items()):
            with col:
                st.markdown(f"**{name.replace('_', ' ').title()}**")
                st.write(f"RMSE: {metrics.get('rmse', '—'):.3f}")
                st.write(f"MAE: {metrics.get('mae', '—'):.3f}")
                st.write(f"R²: {metrics.get('r2', '—'):.3f}")

    fig_path = FIGURES / "residuals_linear_reg.png"
    if fig_path.exists():
        st.subheader("Residual Diagnostics")
        st.image(str(fig_path), use_container_width=True)

    st.markdown("---")

    # Classification
    st.subheader("Classification — High Pollution Detection")
    for fig_name, title in [
        ("roc_curves.png", "ROC Curves"),
        ("pr_curves.png", "Precision-Recall Curves"),
        ("confusion_matrices.png", "Confusion Matrices"),
        ("calibration_curve.png", "Calibration Curves"),
    ]:
        fp = FIGURES / fig_name
        if fp.exists():
            st.subheader(title)
            st.image(str(fp), use_container_width=True)

    # Threshold analysis
    fp = FIGURES / "threshold_value_curve.png"
    if fp.exists():
        st.subheader("Threshold Optimization")
        st.image(str(fp), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# 4. UNSUPERVISED REGIMES
# ═════════════════════════════════════════════════════════════════════════
elif section == "Unsupervised Regimes":
    st.title("Environmental Regime Discovery")
    st.markdown("---")

    unsup = load_metric_file("unsupervised_metrics.json")
    if unsup:
        col1, col2 = st.columns(2)
        for col, (name, m) in zip([col1, col2], unsup.items()):
            with col:
                st.markdown(f"**{name.upper()}**")
                st.write(f"Silhouette Score: {m['silhouette_score']:.3f}")
                st.write(f"Davies-Bouldin Index: {m['davies_bouldin_index']:.3f}")

    for fig_name, title in [
        ("cluster_tsne.png", "Cluster Visualization (t-SNE)"),
        ("cluster_profiles.png", "Cluster Feature Profiles"),
    ]:
        fp = FIGURES / fig_name
        if fp.exists():
            st.subheader(title)
            st.image(str(fp), use_container_width=True)

    cs_path = TABLES / "cluster_summary.csv"
    if cs_path.exists():
        st.subheader("Cluster Summary Table")
        st.dataframe(pd.read_csv(cs_path).style.format(precision=3), use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════
# 5. DEEP LEARNING BENCHMARK
# ═════════════════════════════════════════════════════════════════════════
elif section == "Deep Learning Benchmark":
    st.title("Neural Network Benchmark")
    st.markdown("---")

    deep = load_metric_file("deep_metrics.json")
    if deep:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**MLP Regression**")
            for k, v in deep.get("mlp_regression", {}).items():
                st.write(f"{k.upper()}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")
        with col2:
            st.markdown("**MLP Classification**")
            for k, v in deep.get("mlp_classification", {}).items():
                if k != "confusion_matrix":
                    st.write(f"{k.upper()}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}")

    fp = FIGURES / "nn_training_curves.png"
    if fp.exists():
        st.subheader("Training Curves")
        st.image(str(fp), use_container_width=True)

    st.markdown("""
    **Architecture:** Input → Dense(128, ReLU) → Dense(64, ReLU) → Output

    **Implementation:** sklearn MLPRegressor / MLPClassifier with early stopping.
    For production deployment, a PyTorch implementation with batch normalization
    and learning rate scheduling would be preferred.
    """)


# ═════════════════════════════════════════════════════════════════════════
# 6. POLICY IMPLICATIONS
# ═════════════════════════════════════════════════════════════════════════
elif section == "Policy Implications":
    st.title("Environmental Policy Implications")
    st.markdown("---")

    st.markdown("""
    ### Key Findings

    **1. Predictive Capability:**
    Classical regression models explain a substantial portion of NO₂ variance
    using sensor and meteorological features. This supports real-time
    pollution forecasting from low-cost sensor networks.

    **2. High-Pollution Detection:**
    Classification models achieve strong ROC-AUC, enabling automated alerting
    systems. Threshold optimization reveals the trade-off between false alarms
    and missed events — tunable per regulatory context.

    **3. Environmental Regimes:**
    Unsupervised clustering reveals distinct pollution regimes (e.g., rush-hour
    peaks, overnight lows, seasonal patterns), informing targeted intervention
    strategies.

    ### Limitations

    - **Temporal autocorrelation:** Hourly observations violate independence
      assumptions in classical statistical tests and cross-validated metrics
      may be optimistic.
    - **Single-site data:** Results reflect one Italian city; generalization
      requires multi-site validation.
    - **Synthetic data caveat:** If run with generated data, real-world
      performance will differ.

    ### Recommendations

    - Deploy ensemble models with calibrated probability outputs for
      operational pollution alerting.
    - Use chronological backtesting (as implemented) rather than random
      splits for honest performance estimation.
    - Integrate meteorological forecasts for 24–48 hour pollution prediction.
    """)


# ── Footer ───────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.caption("Air Quality ML Dashboard v1.0")
