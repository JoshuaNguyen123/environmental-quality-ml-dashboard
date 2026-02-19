"""
Air Quality Risk Modeling — Streamlit Dashboard.

Run (from repo root):
    streamlit run app/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

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
CSS_PATH = Path(__file__).parent / "style.css"
if CSS_PATH.exists():
    st.markdown(
    "<style>" + CSS_PATH.read_bytes().decode("utf-8", errors="replace") + "</style>",
    unsafe_allow_html=True,
)

# ── Paths ────────────────────────────────────────────────────────────────
FIGURES = ROOT / "artifacts" / "figures"
METRICS = ROOT / "artifacts" / "metrics"
TABLES = ROOT / "artifacts" / "tables"


# ── Small utilities ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_cached(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def fmt3(x) -> str:
    """Format floats to 3 decimals; return em dash for missing."""
    if isinstance(x, (int, float, np.floating)):
        return f"{float(x):.3f}"
    if x is None or x == "" or x == "—":
        return "—"
    return str(x)


def load_metric_file(name: str) -> dict:
    try:
        return load_json(METRICS / name)
    except FileNotFoundError:
        st.warning(
            f"Metric file not found: {name}. Run `python scripts/train_all.py` (or your pipeline) to generate artifacts."
        )
        return {}
    except Exception as e:
        st.warning(f"Could not load {name}: {e}")
        return {}


def image_if_exists(path: Path, caption: str | None = None):
    if path.exists():
        st.image(str(path), use_container_width=True, caption=caption)


def section_header(title: str, subtitle: str | None = None):
    st.markdown(
        f"""
        <div class="aq-hero">
          <div class="aq-hero-title">{title}</div>
          {f'<div class="aq-hero-sub">{subtitle}</div>' if subtitle else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")


# ═════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### Navigation")
section = st.sidebar.radio(
    "Navigation",
    [
        "Executive Summary",
        "Statistical Overview",
        "Supervised Models",
        "Unsupervised Regimes",
        "Deep Learning Benchmark",
        "Policy Implications",
    ],
    label_visibility="collapsed",
)
st.sidebar.markdown("---")
st.sidebar.caption("Air Quality ML Dashboard v1.0")


# ═════════════════════════════════════════════════════════════════════════
# 1) EXECUTIVE SUMMARY
# ═════════════════════════════════════════════════════════════════════════
if section == "Executive Summary":
    section_header(
        "Air Quality Risk Modeling & Environmental Pattern Analysis",
        "End-to-end ML pipeline for urban pollution forecasting, event detection, and regime discovery.",
    )

    st.markdown(
        """
        <div class="aq-panel">
          <div class="aq-panel-title">Project scope</div>
          <div class="aq-panel-body">
            End-to-end machine learning pipeline for urban air pollution analysis using the
            UCI Air Quality dataset (hourly sensor readings from an Italian city, March 2004 – February 2005).
          </div>
          <div class="aq-panel-title" style="margin-top:12px;">Objectives</div>
          <div class="aq-panel-body">
            <ul style="margin: 0.25rem 0 0.25rem 1.2rem;">
              <li>Predict NO₂ concentration (regression) and detect high-pollution events (classification)</li>
              <li>Discover latent environmental regimes via clustering</li>
              <li>Benchmark classical ML vs neural network performance</li>
              <li>Optimize decision thresholds for cost-sensitive pollution alerting</li>
            </ul>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Key metrics (safe formatting)
    sup = load_metric_file("supervised_metrics.json")
    deep = load_metric_file("deep_metrics.json")
    unsup = load_metric_file("unsupervised_metrics.json")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Linear Reg R²", fmt3(sup.get("linear_regression", {}).get("r2")))
    with col2:
        st.metric("Logistic ROC-AUC", fmt3(sup.get("logistic_regression", {}).get("roc_auc")))
    with col3:
        st.metric("K-Means Silhouette", fmt3(unsup.get("kmeans", {}).get("silhouette_score")))
    with col4:
        st.metric("MLP Reg R²", fmt3(deep.get("mlp_regression", {}).get("r2")))

    st.write("")

    # Model comparison table (stable rendering)
    comp_path = TABLES / "model_comparison.csv"
    st.subheader("Model Comparison")
    if comp_path.exists():
        comp = read_csv_cached(str(comp_path))
        st.dataframe(comp, use_container_width=True, hide_index=True, height=320)
    else:
        st.info("Model comparison table not found yet. Generate artifacts to populate this section.")


# ═════════════════════════════════════════════════════════════════════════
# 2) STATISTICAL OVERVIEW
# ═════════════════════════════════════════════════════════════════════════
elif section == "Statistical Overview":
    section_header("Statistical Overview", "Descriptive statistics, correlations, and hypothesis tests.")

    stats = load_metric_file("stats_summary.json")

    # Descriptive stats
    st.subheader("Descriptive Statistics")
    if isinstance(stats, dict) and "column_statistics" in stats:
        desc_df = pd.DataFrame(stats["column_statistics"]).T
        st.dataframe(desc_df, use_container_width=True, hide_index=False, height=360)
    else:
        st.info("Descriptive statistics not available yet.")

    st.write("")

    # Correlation heatmap
    st.subheader("Pearson Correlation Matrix")
    image_if_exists(FIGURES / "corr_heatmap.png")

    st.write("")

    # Hypothesis tests
    st.subheader("Hypothesis Tests")
    if isinstance(stats, dict) and "hypothesis_tests" in stats:
        ht = stats["hypothesis_tests"]
        if not ht:
            st.info("No hypothesis test results found.")
        for test_name, result in ht.items():
            test_title = result.get("test", test_name) if isinstance(result, dict) else test_name
            with st.expander(test_title):
                if isinstance(result, dict):
                    h0 = result.get("null_hypothesis", "")
                    if h0:
                        st.markdown(f"**H₀:** {h0}")
                    for k, v in result.items():
                        if k in ("test", "null_hypothesis", "contingency_table"):
                            continue
                        st.markdown(f"- **{k}:** {v}")
                else:
                    st.write(result)
    else:
        st.info("Hypothesis test results not available yet.")


# ═════════════════════════════════════════════════════════════════════════
# 3) SUPERVISED MODELS
# ═════════════════════════════════════════════════════════════════════════
elif section == "Supervised Models":
    section_header("Supervised Models", "Regression for NO₂ prediction and classification for high-pollution events.")

    sup = load_metric_file("supervised_metrics.json")

    # --- Regression summary cards (robust + consistent) ---
    st.subheader("Regression — NO₂ Prediction")

    reg_candidates = [
        ("Linear Regression", sup.get("linear_regression", {})),
        ("Random Forest", sup.get("random_forest_regression", sup.get("random_forest", {}))),
        ("MLP (Neural Net)", sup.get("mlp_regression", {})),
    ]
    reg_candidates = [(n, m) for (n, m) in reg_candidates if isinstance(m, dict) and len(m) > 0]

    if reg_candidates:
        cols = st.columns(len(reg_candidates))
        for col, (name, m) in zip(cols, reg_candidates):
            with col:
                st.markdown(f"**{name}**")
                st.markdown(f"- RMSE: **{fmt3(m.get('rmse'))}**")
                st.markdown(f"- MAE: **{fmt3(m.get('mae'))}**")
                st.markdown(f"- R²: **{fmt3(m.get('r2'))}**")
    else:
        st.info("Regression metrics not available yet.")

    st.write("")

    st.subheader("Residual Diagnostics")
    image_if_exists(FIGURES / "residuals_linear_reg.png")
    image_if_exists(FIGURES / "residuals_random_forest.png")
    image_if_exists(FIGURES / "residuals_mlp_reg.png")

    st.write("")

    # --- Classification visuals ---
    st.subheader("Classification — High Pollution Detection")

    for fig_name, title in [
        ("roc_curves.png", "ROC Curves"),
        ("pr_curves.png", "Precision–Recall Curves"),
        ("confusion_matrices.png", "Confusion Matrices"),
        ("calibration_curve.png", "Calibration Curves"),
    ]:
        fp = FIGURES / fig_name
        if fp.exists():
            st.markdown(f"**{title}**")
            st.image(str(fp), use_container_width=True)

    st.write("")

    st.subheader("Threshold Optimization")
    image_if_exists(FIGURES / "threshold_value_curve.png")


# ═════════════════════════════════════════════════════════════════════════
# 4) UNSUPERVISED REGIMES
# ═════════════════════════════════════════════════════════════════════════
elif section == "Unsupervised Regimes":
    section_header("Unsupervised Regimes", "Discover latent environmental regimes via clustering.")

    unsup = load_metric_file("unsupervised_metrics.json")

    # Cards for clustering metrics
    if isinstance(unsup, dict) and len(unsup) > 0:
        # Expect keys like "kmeans", "gmm", etc.
        keys = list(unsup.keys())[:3]  # keep tidy
        cols = st.columns(min(3, len(keys)))
        for col, k in zip(cols, keys):
            m = unsup.get(k, {})
            with col:
                st.markdown(f"**{k.replace('_', ' ').upper()}**")
                st.markdown(f"- Silhouette: **{fmt3(m.get('silhouette_score'))}**")
                st.markdown(f"- Davies–Bouldin: **{fmt3(m.get('davies_bouldin_index'))}**")
    else:
        st.info("Unsupervised metrics not available yet.")

    st.write("")

    for fig_name, title in [
        ("cluster_tsne.png", "Cluster Visualization (t-SNE)"),
        ("cluster_profiles.png", "Cluster Feature Profiles"),
    ]:
        fp = FIGURES / fig_name
        if fp.exists():
            st.subheader(title)
            st.image(str(fp), use_container_width=True)

    st.write("")

    cs_path = TABLES / "cluster_summary.csv"
    st.subheader("Cluster Summary Table")
    if cs_path.exists():
        cs = read_csv_cached(str(cs_path))
        st.dataframe(cs, use_container_width=True, hide_index=True, height=360)
    else:
        st.info("Cluster summary table not found yet.")


# ═════════════════════════════════════════════════════════════════════════
# 5) DEEP LEARNING BENCHMARK
# ═════════════════════════════════════════════════════════════════════════
elif section == "Deep Learning Benchmark":
    section_header("Deep Learning Benchmark", "Neural network baseline vs classical ML performance.")

    deep = load_metric_file("deep_metrics.json")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("MLP Regression")
        m = deep.get("mlp_regression", {}) if isinstance(deep, dict) else {}
        if isinstance(m, dict) and m:
            st.markdown(f"- RMSE: **{fmt3(m.get('rmse'))}**")
            st.markdown(f"- MAE: **{fmt3(m.get('mae'))}**")
            st.markdown(f"- R²: **{fmt3(m.get('r2'))}**")
        else:
            st.info("MLP regression metrics not available yet.")

    with col2:
        st.subheader("MLP Classification")
        m = deep.get("mlp_classification", {}) if isinstance(deep, dict) else {}
        if isinstance(m, dict) and m:
            # Avoid dumping confusion matrices raw
            for k in ("roc_auc", "f1", "precision", "recall", "accuracy"):
                if k in m:
                    st.markdown(f"- {k.upper()}: **{fmt3(m.get(k))}**")
        else:
            st.info("MLP classification metrics not available yet.")

    st.write("")

    st.subheader("Training Curves")
    image_if_exists(FIGURES / "nn_training_curves.png")

    st.markdown(
        """
        <div class="aq-panel">
          <div class="aq-panel-title">Baseline architecture</div>
          <div class="aq-panel-body">
            Input → Dense(128, ReLU) → Dense(64, ReLU) → Output
          </div>
          <div class="aq-panel-title" style="margin-top:12px;">Implementation notes</div>
          <div class="aq-panel-body">
            Implemented with sklearn MLPRegressor / MLPClassifier (early stopping).
            For a production-grade version, a PyTorch implementation with batch normalization,
            learning-rate scheduling, and proper time-series validation would be preferred.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════
# 6) POLICY IMPLICATIONS
# ═════════════════════════════════════════════════════════════════════════
elif section == "Policy Implications":
    section_header("Policy Implications", "How to turn model outputs into actionable decisions.")

    st.markdown(
        """
        <div class="aq-panel">
          <div class="aq-panel-title">Key Findings</div>
          <div class="aq-panel-body">
            <b>1) Predictive capability:</b> Regression models explain a substantial portion of NO₂ variance using sensor
            and meteorological features, supporting real-time forecasting from low-cost sensor networks.<br/><br/>
            <b>2) High-pollution detection:</b> Classification models achieve strong ROC-AUC, enabling automated alerting.
            Threshold tuning exposes the trade-off between false alarms and missed events — adjustable per regulatory context.<br/><br/>
            <b>3) Environmental regimes:</b> Clustering reveals distinct regimes (rush-hour peaks, overnight lows, seasonal effects),
            informing targeted intervention strategies.
          </div>
          <div class="aq-panel-title" style="margin-top:12px;">Limitations</div>
          <div class="aq-panel-body">
            <ul style="margin: 0.25rem 0 0.25rem 1.2rem;">
              <li><b>Temporal autocorrelation:</b> hourly observations violate independence assumptions; random CV can be optimistic.</li>
              <li><b>Single-site data:</b> generalization requires multi-site validation.</li>
              <li><b>Data quality:</b> sensor drift and missingness can meaningfully impact performance.</li>
            </ul>
          </div>
          <div class="aq-panel-title" style="margin-top:12px;">Recommendations</div>
          <div class="aq-panel-body">
            Deploy calibrated probability outputs for alerting, use chronological backtesting,
            and integrate meteorological forecasts for 24–48 hour prediction horizons.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )