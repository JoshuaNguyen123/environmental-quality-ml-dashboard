"""
Air Quality Risk Modeling — Streamlit Dashboard.

Run (from repo root):
    streamlit run app/app.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np

from src.utils.io import load_json, load_project_config

try:
    import altair as alt
except Exception:  # pragma: no cover (optional runtime dependency via Streamlit)
    alt = None

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Air Quality ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load custom CSS ──────────────────────────────────────────────────────
CSS_PATH = Path(__file__).parent / "style_v2.css"
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


@st.cache_data(show_spinner=False, ttl=600)
def http_get_json(url: str) -> dict:
    req = Request(url, headers={"User-Agent": "air-quality-ml-dashboard/1.0"})
    with urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def search_geocode_open_meteo(city: str, *, count: int = 6) -> list[dict]:
    url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
        {"name": city, "count": int(count), "language": "en", "format": "json"}
    )
    data = http_get_json(url)
    results = data.get("results") if isinstance(data, dict) else None
    if not results:
        return []
    out: list[dict] = []
    for r in results:
        if isinstance(r, dict) and "latitude" in r and "longitude" in r:
            out.append(r)
    return out


def refresh_bucket(refresh_minutes: int) -> int:
    m = max(1, int(refresh_minutes))
    return int(time.time() // (m * 60))


def format_geocode_result(r: dict) -> str:
    name = r.get("name") or "Unknown"
    admin1 = r.get("admin1")
    country = r.get("country")
    parts = [p for p in [name, admin1, country] if p]
    lat = r.get("latitude")
    lon = r.get("longitude")
    suffix = ""
    try:
        suffix = f" — ({float(lat):.4f}, {float(lon):.4f})"
    except Exception:
        suffix = ""
    return ", ".join(parts) + suffix


@st.cache_data(show_spinner=False, ttl=600)
def fetch_open_meteo_forecast(
    *,
    latitude: float,
    longitude: float,
    hourly_vars: list[str],
    daily_vars: list[str],
    current_vars: list[str],
    past_days: int = 1,
    forecast_days: int = 3,
    cache_bucket: int = 0,
) -> dict:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }
    if hourly_vars:
        params["hourly"] = ",".join(hourly_vars)
    if daily_vars:
        params["daily"] = ",".join(daily_vars)
    if current_vars:
        params["current"] = ",".join(current_vars)

    url = "https://api.open-meteo.com/v1/forecast?" + urlencode(params)
    data = http_get_json(url)
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(data.get("reason", "Open-Meteo error"))
    return data


def open_meteo_hourly_to_df(payload: dict, vars_: list[str]) -> pd.DataFrame:
    hourly = payload.get("hourly", {}) if isinstance(payload, dict) else {}
    times = hourly.get("time", [])
    if not times:
        return pd.DataFrame()
    df = pd.DataFrame({"time": pd.to_datetime(times)})
    for v in vars_:
        if v in hourly:
            df[v] = hourly[v]
    return df.set_index("time")


def open_meteo_daily_to_df(payload: dict, vars_: list[str]) -> pd.DataFrame:
    daily = payload.get("daily", {}) if isinstance(payload, dict) else {}
    times = daily.get("time", [])
    if not times:
        return pd.DataFrame()
    # pd.to_datetime(list[str]) returns a DatetimeIndex (no .dt accessor)
    df = pd.DataFrame({"date": pd.to_datetime(times).date})
    for v in vars_:
        if v in daily:
            df[v] = daily[v]
    return df.set_index("date")


def trigger_auto_refresh(interval_minutes: int):
    ms = max(1, int(interval_minutes)) * 60 * 1000
    components.html(
        f"""
        <script>
            window.setTimeout(function() {{
                window.parent.location.reload();
            }}, {ms});
        </script>
        """,
        height=0,
    )


LIVE_WEATHER_META: dict[str, dict[str, str]] = {
    "temperature_2m": {
        "label": "Temperature (2m)",
        "meaning": "Air temperature at 2 meters above ground.",
    },
    "relative_humidity_2m": {
        "label": "Relative Humidity (2m)",
        "meaning": "Relative humidity at 2 meters above ground (0–100%).",
    },
    "wind_speed_10m": {
        "label": "Wind Speed (10m)",
        "meaning": "Wind speed at 10 meters above ground.",
    },
    "precipitation": {
        "label": "Precipitation (hourly sum)",
        "meaning": "Total precipitation over the preceding hour (rain + showers + snow).",
    },
    "cloud_cover": {
        "label": "Cloud Cover (total)",
        "meaning": "Total cloud cover as an area fraction (0–100%).",
    },
    "uv_index_max": {
        "label": "UV Index (daily max)",
        "meaning": "Daily maximum UV index (dimensionless). Typical range 0–11+.",
    },
}


def render_single_series_chart(
    *,
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str,
    unit: str | None,
    height: int = 240,
):
    if alt is None:
        st.line_chart(df.set_index(x_col)[[y_col]], height=height)
        return

    y_title = f"{title} ({unit})" if unit else title
    c = (
        alt.Chart(df)
        .mark_line()
        .encode(
            x=alt.X(f"{x_col}:T", title="Time (local)"),
            y=alt.Y(f"{y_col}:Q", title=y_title),
            tooltip=[
                alt.Tooltip(f"{x_col}:T", title="Time"),
                alt.Tooltip(f"{y_col}:Q", title=title, format=".3f"),
            ],
        )
        .properties(height=height)
        .interactive()
    )
    st.altair_chart(c, use_container_width=True)


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
        "Live Weather (API)",
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

    def _pretty(name: str) -> str:
        return name.replace("_", " ").title()

    # --- Regression summary ---
    st.subheader("Regression — NO₂ Prediction")

    reg_candidates = []
    if isinstance(sup, dict):
        for k, m in sup.items():
            if isinstance(m, dict) and "rmse" in m and "r2" in m:
                reg_candidates.append((k, m))

    if reg_candidates:
        reg_df = pd.DataFrame(
            [
                {"Model": _pretty(k), "RMSE": m.get("rmse"), "MAE": m.get("mae"), "R²": m.get("r2")}
                for k, m in reg_candidates
            ]
        ).sort_values("RMSE", ascending=True)
        st.dataframe(reg_df, use_container_width=True, hide_index=True, height=260)
    else:
        st.info("Regression metrics not available yet.")

    st.write("")

    st.subheader("Residual Diagnostics")
    for key, _m in reg_candidates:
        image_if_exists(FIGURES / f"residuals_{key}.png", caption=_pretty(key))

    st.write("")

    # --- Classification summary + visuals ---
    st.subheader("Classification — High Pollution Detection")
    clf_candidates = []
    if isinstance(sup, dict):
        for k, m in sup.items():
            if isinstance(m, dict) and "f1" in m and "precision" in m and "recall" in m:
                clf_candidates.append((k, m))

    if clf_candidates:
        clf_df = pd.DataFrame(
            [
                {
                    "Model": _pretty(k),
                    "ROC-AUC": m.get("roc_auc"),
                    "PR-AUC": m.get("pr_auc"),
                    "F1": m.get("f1"),
                    "Precision": m.get("precision"),
                    "Recall": m.get("recall"),
                }
                for k, m in clf_candidates
            ]
        ).sort_values("F1", ascending=False)
        st.dataframe(clf_df, use_container_width=True, hide_index=True, height=260)

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
        keys = list(unsup.keys())
        cols = st.columns(min(3, len(keys)))
        for idx, k in enumerate(keys):
            m = unsup.get(k, {})
            with cols[idx % len(cols)]:
                st.markdown(f"**{k.replace('_', ' ').upper()}**")
                st.markdown(f"- Silhouette: **{fmt3(m.get('silhouette_score'))}**")
                st.markdown(f"- Davies–Bouldin: **{fmt3(m.get('davies_bouldin_index'))}**")
                if "n_clusters" in m:
                    st.markdown(f"- Clusters: **{m.get('n_clusters')}**")
                if "noise_ratio" in m:
                    st.markdown(f"- Noise ratio: **{fmt3(m.get('noise_ratio'))}**")
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

    reg_rows = []
    clf_rows = []
    if isinstance(deep, dict):
        for k, m in deep.items():
            if not isinstance(m, dict):
                continue
            if k.endswith("_regression") and "rmse" in m:
                reg_rows.append(
                    {"Profile": k.replace("_regression", "").replace("_", " ").title(), "RMSE": m.get("rmse"), "MAE": m.get("mae"), "R²": m.get("r2")}
                )
            if k.endswith("_classification") and "f1" in m:
                clf_rows.append(
                    {
                        "Profile": k.replace("_classification", "").replace("_", " ").title(),
                        "ROC-AUC": m.get("roc_auc"),
                        "PR-AUC": m.get("pr_auc"),
                        "F1": m.get("f1"),
                        "Precision": m.get("precision"),
                        "Recall": m.get("recall"),
                    }
                )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("MLP Regression Profiles")
        if reg_rows:
            st.dataframe(pd.DataFrame(reg_rows).sort_values("RMSE"), use_container_width=True, hide_index=True, height=260)
        else:
            st.info("MLP regression metrics not available yet.")

    with col2:
        st.subheader("MLP Classification Profiles")
        if clf_rows:
            st.dataframe(pd.DataFrame(clf_rows).sort_values("F1", ascending=False), use_container_width=True, hide_index=True, height=260)
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


# ═════════════════════════════════════════════════════════════════════════
# 7) LIVE WEATHER (API)
# ═════════════════════════════════════════════════════════════════════════
elif section == "Live Weather (API)":
    section_header("Live Weather (API)", "Near-live weather view with configurable refresh cadence.")

    try:
        project_cfg = load_project_config()
        live_cfg = project_cfg.get("live_weather", {}) if isinstance(project_cfg, dict) else {}
    except Exception:
        live_cfg = {}

    default_refresh_minutes = int(live_cfg.get("refresh_minutes", 3) or 3)
    default_city = live_cfg.get("default_city", "London")
    default_hourly = live_cfg.get("default_hourly_variables", ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"])
    default_days = int(live_cfg.get("forecast_days", 3) or 3)

    st.markdown(
        """
        <div class="aq-panel">
          <div class="aq-panel-title">Data source</div>
          <div class="aq-panel-body">
            This tab is independent of the ML dataset and demonstrates live external weather integration.
            Live weather data is provided by Open-Meteo (keyless), with a configurable refresh cadence.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    colA, colB = st.columns([1.2, 1.0])
    with colB:
        refresh_minutes = st.select_slider(
            "Auto-refresh interval",
            options=[1, 2, 3, 4, 5],
            value=max(1, min(5, default_refresh_minutes)),
            help="Refresh cadence for reloading weather values.",
        )
        auto_refresh = st.toggle("Auto-refresh", value=True)
        forecast_days = st.slider("Forecast window (days)", min_value=1, max_value=7, value=max(1, min(7, default_days)))
        if st.button("Refresh now", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    with colA:
        st.markdown("#### Location")

        with st.expander("What location inputs work?"):
            st.markdown(
                """
                You have two ways to choose a location:

                - **Search by name (geocoding)**: type a place name and pick the best match from the results list.
                  - Examples: `London`, `Paris`, `Seattle`, `Springfield, IL`, `Cambridge, UK`
                  - If the name is ambiguous, you’ll see **multiple matches** (different countries/states).
                - **Manual coordinates**: directly enter latitude/longitude.

                Notes:
                - If you’re getting no results, try adding a region/country (e.g. `Portland, OR` vs `Portland`).
                """
            )

        if "live_weather_city" not in st.session_state:
            st.session_state["live_weather_city"] = str(default_city)

        def _apply_quick_pick():
            pick = st.session_state.get("live_weather_quick_pick")
            if pick and pick != "(custom)":
                st.session_state["live_weather_city"] = pick

        st.selectbox(
            "Quick picks",
            options=[
                "(custom)",
                "London",
                "New York",
                "Los Angeles",
                "Chicago",
                "Seattle",
                "Paris",
                "Berlin",
                "Tokyo",
                "Delhi",
                "Sydney",
            ],
            key="live_weather_quick_pick",
            on_change=_apply_quick_pick,
            help="Optional. Selecting a quick pick will populate the search box below.",
        )

        city = st.text_input(
            "City / place search",
            key="live_weather_city",
            help="Used for geocoding (lat/lon). Examples: 'Springfield, IL', 'Cambridge, UK'.",
        )

        use_manual = st.toggle("Use manual coordinates", value=False, help="If geocoding is unreliable, use lat/lon directly.")
        if use_manual:
            lat = st.number_input("Latitude", value=51.5074, format="%.6f")
            lon = st.number_input("Longitude", value=-0.1278, format="%.6f")
        else:
            query = city.strip() if city else ""
            matches = search_geocode_open_meteo(query) if query else []
            if not matches:
                st.warning("Could not geocode that city. Try a different spelling or switch to manual coordinates.")
                lat, lon = None, None
            else:
                if len(matches) > 1:
                    st.caption(f"Found **{len(matches)}** matches. Pick the correct one below.")
                idx = st.selectbox(
                    "Geocoding matches",
                    options=list(range(len(matches))),
                    format_func=lambda i: format_geocode_result(matches[i]),
                )
                geo = matches[int(idx)]
                lat = float(geo["latitude"])
                lon = float(geo["longitude"])
                st.caption(f"Using: **{format_geocode_result(geo)}**")

    hourly_options = {
        "temperature_2m": "Temperature (2m)",
        "relative_humidity_2m": "Relative Humidity (2m)",
        "wind_speed_10m": "Wind Speed (10m)",
        "precipitation": "Precipitation (hourly sum)",
        "cloud_cover": "Cloud Cover (total)",
    }
    daily_options = {
        "uv_index_max": "UV Index (daily max)",
    }

    selected_hourly = st.multiselect(
        "Hourly variables",
        options=list(hourly_options.keys()),
        default=[v for v in default_hourly if v in hourly_options] or list(hourly_options.keys())[:3],
        format_func=lambda k: hourly_options.get(k, k),
    )
    selected_daily = st.multiselect(
        "Daily variables",
        options=list(daily_options.keys()),
        default=["uv_index_max"],
        format_func=lambda k: daily_options.get(k, k),
    )

    if auto_refresh:
        trigger_auto_refresh(refresh_minutes)

    if lat is None or lon is None:
        st.info("Select a location to load live weather data.")
        payload = None
    else:
        bucket = refresh_bucket(refresh_minutes)
        try:
            payload = fetch_open_meteo_forecast(
                latitude=float(lat),
                longitude=float(lon),
                hourly_vars=selected_hourly,
                daily_vars=selected_daily,
                current_vars=[v for v in selected_hourly if v in ("temperature_2m", "relative_humidity_2m", "wind_speed_10m")],
                past_days=1,
                forecast_days=int(forecast_days),
                cache_bucket=bucket,
            )
        except Exception as e:
            st.error(f"Weather API request failed: {e}")
            payload = None

    if payload is None:
        st.info("Weather data unavailable.")
        current = {}
        current_units = {}
        hourly_units = {}
        daily_units = {}
        hourly_df = pd.DataFrame()
        daily_df = pd.DataFrame()
        tz = None
    else:
        current = payload.get("current", {}) if isinstance(payload, dict) else {}
        current_units = payload.get("current_units", {}) if isinstance(payload, dict) else {}
        hourly_units = payload.get("hourly_units", {}) if isinstance(payload, dict) else {}
        daily_units = payload.get("daily_units", {}) if isinstance(payload, dict) else {}
        hourly_df = open_meteo_hourly_to_df(payload, selected_hourly)
        daily_df = open_meteo_daily_to_df(payload, selected_daily)
        tz = payload.get("timezone") if isinstance(payload, dict) else None
        st.caption("Cadence: Open-Meteo forecast model data is typically hourly (with daily aggregates).")

    if tz:
        st.caption(f"Timezone: `{tz}` (API returns timestamps in this local timezone)")
    st.caption(
        f"Last refreshed: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        + (f" — next auto-refresh in ~{refresh_minutes} minute(s)." if auto_refresh else "")
    )

    if isinstance(current, dict) and current:
        st.subheader("Current conditions")
        m1, m2, m3 = st.columns(3)
        with m1:
            if "temperature_2m" in current:
                st.metric(
                    "Temperature",
                    f"{current.get('temperature_2m')} {current_units.get('temperature_2m', '')}".strip(),
                )
        with m2:
            if "relative_humidity_2m" in current:
                st.metric(
                    "Humidity",
                    f"{current.get('relative_humidity_2m')} {current_units.get('relative_humidity_2m', '')}".strip(),
                )
        with m3:
            if "wind_speed_10m" in current:
                st.metric(
                    "Wind Speed",
                    f"{current.get('wind_speed_10m')} {current_units.get('wind_speed_10m', '')}".strip(),
                )
        if "time" in current:
            st.caption(f"Current time: `{current.get('time')}` (local to the selected timezone)")

    st.write("")
    st.subheader("Hourly series")
    if hourly_df.empty:
        st.info("No hourly data returned for the selected variables.")
    else:
        now = datetime.now()
        start = now - pd.Timedelta(hours=24)
        end = now + pd.Timedelta(days=int(forecast_days))
        view = hourly_df[(hourly_df.index >= start) & (hourly_df.index <= end)].copy()
        if view.empty:
            st.info("No hourly points in the selected time window.")
        else:
            for v in selected_hourly:
                if v not in view.columns:
                    continue
                meta = LIVE_WEATHER_META.get(v, {})
                label = meta.get("label", v)
                meaning = meta.get("meaning", "")
                unit = hourly_units.get(v, "")

                st.markdown(f"**{label}**")
                if meaning:
                    st.caption(meaning + (f" Unit: `{unit}`" if unit else ""))

                d = view[[v]].reset_index().rename(columns={"time": "time", v: "value"})
                render_single_series_chart(
                    df=d,
                    x_col="time",
                    y_col="value",
                    title=label,
                    unit=unit,
                    height=240,
                )

        with st.expander("Show hourly data table"):
            st.dataframe(view.reset_index(), use_container_width=True, hide_index=True, height=320)

    if selected_daily:
        st.write("")
        st.subheader("Daily series")
        if daily_df.empty:
            st.info("No daily data returned for the selected variables.")
        else:
            for v in selected_daily:
                if v not in daily_df.columns:
                    continue
                meta = LIVE_WEATHER_META.get(v, {})
                label = meta.get("label", v)
                meaning = meta.get("meaning", "")
                unit = daily_units.get(v, "")

                st.markdown(f"**{label}**")
                if meaning:
                    st.caption(meaning + (f" Unit: `{unit}`" if unit else ""))

                d = daily_df[[v]].reset_index().rename(columns={"date": "date", v: "value"})
                d["date"] = pd.to_datetime(d["date"])
                render_single_series_chart(
                    df=d,
                    x_col="date",
                    y_col="value",
                    title=label,
                    unit=unit,
                    height=220,
                )

            with st.expander("Show daily data table"):
                st.dataframe(daily_df.reset_index(), use_container_width=True, hide_index=True, height=240)
