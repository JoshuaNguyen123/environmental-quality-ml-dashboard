"""Unsupervised learning: K-Means and Gaussian Mixture Model clustering."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, davies_bouldin_score
from typing import Dict, Tuple


def train_kmeans(
    X: np.ndarray, params: dict | None = None
) -> Tuple[KMeans, np.ndarray]:
    """K-Means clustering: minimize Sum_k Sum_{x_i in C_k} ||x_i - mu_k||^2

    Assumptions:
        - Spherical, equal-variance clusters
        - Euclidean distance is meaningful
        - Correct k is specified
    """
    if params is None:
        params = {"n_clusters": 4, "random_state": 42, "n_init": 10}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = KMeans(**params)
    labels = model.fit_predict(X_scaled)
    return model, labels


def train_gmm(
    X: np.ndarray, params: dict | None = None
) -> Tuple[GaussianMixture, np.ndarray]:
    """Gaussian Mixture Model: p(x) = Sum pi_k N(x | mu_k, Sum_k)

    Assumptions:
        - Gaussian-distributed clusters
        - Soft probabilistic assignments
    """
    if params is None:
        params = {"n_components": 4, "random_state": 42, "covariance_type": "full"}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = GaussianMixture(**params)
    labels = model.fit_predict(X_scaled)
    return model, labels


def cluster_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    sil = silhouette_score(X_scaled, labels)
    dbi = davies_bouldin_score(X_scaled, labels)
    return {
        "silhouette_score": float(sil),
        "davies_bouldin_index": float(dbi),
    }


def build_cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list,
    target_col: str = "NO2_GT",
) -> pd.DataFrame:
    """Build cluster profiles: mean features and pollution per cluster."""
    df = df.copy()
    df["Cluster"] = labels
    agg = df.groupby("Cluster")[feature_cols + [target_col]].mean()

    # Normalize for profile comparison
    normed = (agg - agg.mean()) / agg.std()
    normed = normed.reset_index()
    return normed


def reduce_for_viz(X: np.ndarray, n_components: int = 2, seed: int = 42) -> np.ndarray:
    """Reduce dimensions with t-SNE for cluster visualization."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    tsne = TSNE(n_components=n_components, random_state=seed, perplexity=30, max_iter=500)
    return tsne.fit_transform(X_scaled)
