"""Data ingestion: load or generate UCI Air Quality dataset.

The UCI Air Quality dataset contains hourly averaged responses from an array
of 5 metal-oxide chemical sensors embedded in an Air Quality Chemical
Multisensor Device deployed in a significantly polluted area at road level
in an Italian city (March 2004 - February 2005).

When the real dataset is unavailable, this module generates a statistically
faithful synthetic version for demonstration purposes.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_air_quality(n_hours: int = 9357, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic air quality data mimicking the UCI dataset distribution.

    The real dataset has ~9,357 valid hourly records spanning Mar 2004 - Feb 2005.
    We reproduce realistic marginal distributions, correlations, and temporal patterns.
    """
    rng = np.random.default_rng(seed)

    # Time index: hourly from 2004-03-10 to 2005-04-04
    start = pd.Timestamp("2004-03-10 18:00:00")
    dt = pd.date_range(start, periods=n_hours, freq="h")

    hour = dt.hour
    month = dt.month

    # Temperature: seasonal + diurnal cycle (Italian city)
    temp_seasonal = 12 + 10 * np.sin(2 * np.pi * (month - 3) / 12)
    temp_diurnal = 4 * np.sin(2 * np.pi * (hour - 6) / 24)
    temperature = temp_seasonal + temp_diurnal + rng.normal(0, 2.5, n_hours)

    # Relative Humidity: inversely related to temperature
    rh = 55 - 0.6 * (temperature - 15) + rng.normal(0, 8, n_hours)
    rh = np.clip(rh, 9, 90)

    # Absolute Humidity
    abs_humidity = 0.7 + 0.04 * temperature + 0.005 * rh + rng.normal(0, 0.15, n_hours)
    abs_humidity = np.clip(abs_humidity, 0.1, 2.5)

    # Traffic pattern (rush hours)
    traffic = np.where((hour >= 7) & (hour <= 9), 1.5,
              np.where((hour >= 17) & (hour <= 19), 1.4,
              np.where((hour >= 23) | (hour <= 5), 0.4, 1.0)))
    traffic *= (1 + rng.normal(0, 0.15, n_hours))

    # CO(GT) mg/m^3: traffic-driven
    co_gt = 1.5 * traffic + 0.3 * rng.exponential(1.0, n_hours)
    co_gt = np.clip(co_gt, 0.1, 11.9)

    # NOx(GT) ug/m^3: traffic + temperature interaction
    nox_gt = 150 * traffic - 2 * temperature + rng.normal(0, 50, n_hours)
    nox_gt = np.clip(nox_gt, 2, 1479)

    # NO2(GT) ug/m^3: correlated with NOx but distinct
    no2_gt = 0.3 * nox_gt + 40 * traffic + rng.normal(0, 30, n_hours)
    no2_gt = np.clip(no2_gt, 2, 340)

    # Sensor responses (correlated with ground truth + noise)
    pt08_s1 = 600 + 150 * co_gt + rng.normal(0, 80, n_hours)
    pt08_s2 = 700 + 3 * nox_gt + rng.normal(0, 100, n_hours)
    pt08_s3 = 1200 - 1.5 * nox_gt + rng.normal(0, 120, n_hours)
    pt08_s4 = 800 + 2 * no2_gt + rng.normal(0, 100, n_hours)
    pt08_s5 = 900 + 80 * co_gt + rng.normal(0, 150, n_hours)

    # Clip sensor values to realistic ranges
    pt08_s1 = np.clip(pt08_s1, 647, 2040)
    pt08_s2 = np.clip(pt08_s2, 383, 2214)
    pt08_s3 = np.clip(pt08_s3, 322, 2683)
    pt08_s4 = np.clip(pt08_s4, 551, 2775)
    pt08_s5 = np.clip(pt08_s5, 221, 2523)

    # Compute C6H6 before injecting sentinel values
    c6h6_gt = np.clip(np.round(co_gt * 2.5 + rng.normal(0, 1.5, n_hours), 1), 0.1, 40.0)

    # Introduce ~3% missing values (coded as -200 in original)
    miss_mask_co = rng.random(n_hours) < 0.03
    miss_mask_nox = rng.random(n_hours) < 0.03
    miss_mask_no2 = rng.random(n_hours) < 0.03
    co_gt = np.where(miss_mask_co, -200.0, co_gt)
    nox_gt = np.where(miss_mask_nox, -200.0, nox_gt)
    no2_gt = np.where(miss_mask_no2, -200.0, no2_gt)

    df = pd.DataFrame({
        "Date": dt.strftime("%d/%m/%Y"),
        "Time": dt.strftime("%H.%M.%S"),
        "CO_GT": np.round(co_gt, 1),
        "PT08_S1_CO": np.round(pt08_s1).astype(int),
        "NMHC_GT": -200,  # Mostly missing in real dataset
        "C6H6_GT": c6h6_gt,
        "PT08_S2_NMHC": np.round(pt08_s2).astype(int),
        "NOx_GT": np.round(nox_gt).astype(int),
        "PT08_S3_NOx": np.round(pt08_s3).astype(int),
        "NO2_GT": np.round(no2_gt).astype(int),
        "PT08_S4_NO2": np.round(pt08_s4).astype(int),
        "PT08_S5_O3": np.round(pt08_s5).astype(int),
        "Temperature": np.round(temperature, 1),
        "Rel_Humidity": np.round(rh, 1),
        "Abs_Humidity": np.round(abs_humidity, 4),
    })
    return df


def load_or_generate(raw_path: str | Path) -> pd.DataFrame:
    """Load real CSV if available, otherwise generate synthetic data."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        # UCI format uses ';' separator and ',' as decimal
        df = pd.read_csv(raw_path, sep=";", decimal=",")
        # Drop trailing empty columns
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
        # Coerce numeric columns (synthetic CSV is written with '.' decimal, so re-read gives strings when decimal=',')
        for col in df.columns:
            if col not in ("Date", "Time"):
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    print("[ingest] Real dataset not found -- generating synthetic data.")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_air_quality()
    df.to_csv(raw_path, index=False, sep=";")
    return df
