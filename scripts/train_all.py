#!/usr/bin/env python3
"""train_all.py — Run the complete ML pipeline end-to-end.

Usage:
    python scripts/train_all.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd

from src.utils.io import (
    load_project_config, load_model_config, load_threshold_config,
    save_json, save_model, save_csv, ensure_dirs,
)
from src.utils.plotting import (
    correlation_heatmap, residual_plot, roc_curves, pr_curves,
    confusion_matrices, calibration_curve_plot, cluster_scatter,
    cluster_profiles as plot_cluster_profiles, training_curves,
    threshold_value_curve,
)
from src.data.ingest import load_or_generate
from src.data.clean import clean
from src.data.split import chronological_split
from src.features.engineer import engineer_features, get_feature_columns
from src.features.preprocessing import prepare_xy
from src.stats.descriptive import compute_descriptive_stats
from src.stats.hypothesis import two_sample_ttest, seasonal_anova, chi_square_test
from src.stats.diagnostics import variance_inflation_factor, residual_diagnostics
from src.models.supervised import (
    train_linear_regression, train_random_forest_reg,
    train_logistic_regression, train_random_forest_clf,
)
from src.models.unsupervised import (
    train_kmeans, train_gmm, cluster_metrics,
    build_cluster_summary, reduce_for_viz,
)
from src.models.deep_learning import train_mlp_regressor, train_mlp_classifier
from src.models.evaluation import (
    regression_metrics, classification_metrics,
    get_roc_data, get_pr_data, get_calibration_data,
    compute_permutation_importance,
)
from src.business.threshold_analysis import threshold_sweep


def main():
    print("=" * 70)
    print("  AIR QUALITY ML PIPELINE — Full Training Run")
    print("=" * 70)

    cfg = load_project_config()
    mcfg = load_model_config()
    tcfg = load_threshold_config()
    ensure_dirs()

    FIGURES = ROOT / cfg["paths"]["figures"]
    METRICS = ROOT / cfg["paths"]["metrics"]
    MODELS = ROOT / cfg["paths"]["models"]
    TABLES = ROOT / cfg["paths"]["tables"]

    # ═════════════════════════════════════════════════════════════════════
    # 1. DATA PIPELINE
    # ═════════════════════════════════════════════════════════════════════
    print("\n[1/7] Data ingestion & cleaning...")
    raw_path = ROOT / cfg["paths"]["raw_data"]
    df_raw = load_or_generate(raw_path)
    df_clean = clean(df_raw, missing_sentinel=cfg["data"]["missing_value_sentinel"])
    save_csv(df_clean, ROOT / cfg["paths"]["cleaned_data"])

    print("[1/7] Feature engineering...")
    threshold = tcfg["no2"]["regulatory_threshold"]
    df_feat = engineer_features(df_clean, regulatory_threshold=threshold)
    save_csv(df_feat, ROOT / cfg["paths"]["features"])

    print("[1/7] Chronological split...")
    train_df, val_df, test_df = chronological_split(
        df_feat,
        train_ratio=mcfg["splits"]["train_ratio"],
        val_ratio=mcfg["splits"]["val_ratio"],
    )
    save_csv(train_df, ROOT / cfg["paths"]["train"])
    save_csv(val_df, ROOT / cfg["paths"]["val"])
    save_csv(test_df, ROOT / cfg["paths"]["test"])

    feature_cols = get_feature_columns(train_df)
    print(f"    Features ({len(feature_cols)}): {feature_cols[:8]}...")

    X_train, y_train_reg = prepare_xy(train_df, feature_cols, "NO2_GT")
    X_val, y_val_reg = prepare_xy(val_df, feature_cols, "NO2_GT")
    X_test, y_test_reg = prepare_xy(test_df, feature_cols, "NO2_GT")

    _, y_train_clf = prepare_xy(train_df, feature_cols, "High_Pollution")
    _, y_val_clf = prepare_xy(val_df, feature_cols, "High_Pollution")
    _, y_test_clf = prepare_xy(test_df, feature_cols, "High_Pollution")

    # Save a standalone fitted preprocessor for reuse
    from sklearn.preprocessing import StandardScaler as _SS
    _preprocessor = _SS().fit(X_train)
    save_model(_preprocessor, MODELS / "preprocessor.joblib")

    print(f"    Class balance (train): {y_train_clf.mean():.2%} positive")

    # ═════════════════════════════════════════════════════════════════════
    # 2. STATISTICAL ANALYSIS
    # ═════════════════════════════════════════════════════════════════════
    print("\n[2/7] Statistical analysis...")
    numeric_cols = [c for c in feature_cols if c in df_feat.columns]
    stats = compute_descriptive_stats(df_feat, numeric_cols)

    # Correlation heatmap
    core_cols = [c for c in cfg["data"]["sensor_columns"] + cfg["data"]["ground_truth_columns"]
                 + cfg["data"]["environmental_columns"] if c in df_feat.columns]
    corr = df_feat[core_cols].corr()
    correlation_heatmap(corr, str(FIGURES / "corr_heatmap.png"))

    # Hypothesis tests
    ttest_result = two_sample_ttest(df_feat)
    anova_result = seasonal_anova(df_feat)
    chi2_result = chi_square_test(
        df_feat,
        bins=tcfg["humidity_buckets"]["bins"],
        labels=tcfg["humidity_buckets"]["labels"],
    )

    stats["hypothesis_tests"] = {
        "two_sample_ttest": ttest_result,
        "seasonal_anova": anova_result,
        "chi_square": chi2_result,
    }

    save_json(stats, METRICS / "stats_summary.json")
    print("    [done] Descriptive stats, hypothesis tests, correlation heatmap")

    # ═════════════════════════════════════════════════════════════════════
    # 3. SUPERVISED — REGRESSION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[3/7] Training regression models...")
    supervised_results = {}

    # Linear Regression
    lr_model = train_linear_regression(X_train, y_train_reg)
    lr_pred = lr_model.predict(X_test)
    lr_metrics = regression_metrics(y_test_reg, lr_pred)
    supervised_results["linear_regression"] = lr_metrics
    save_model(lr_model, MODELS / "linear_regression.joblib")

    # Diagnostics
    residuals = y_test_reg - lr_pred
    residual_plot(y_test_reg, lr_pred, str(FIGURES / "residuals_linear_reg.png"), "Linear Regression")
    diag = residual_diagnostics(residuals)
    vif = variance_inflation_factor(X_train, feature_cols)
    supervised_results["linear_regression"]["diagnostics"] = diag
    supervised_results["linear_regression"]["vif"] = vif
    print(f"    Linear Reg — RMSE: {lr_metrics['rmse']:.3f}, R²: {lr_metrics['r2']:.3f}")

    # Random Forest Regression
    rf_reg_model = train_random_forest_reg(X_train, y_train_reg, mcfg["supervised"]["random_forest_reg"]["params"])
    rf_reg_pred = rf_reg_model.predict(X_test)
    rf_reg_metrics = regression_metrics(y_test_reg, rf_reg_pred)
    supervised_results["random_forest_reg"] = rf_reg_metrics
    save_model(rf_reg_model, MODELS / "random_forest_reg.joblib")
    print(f"    RF Reg     — RMSE: {rf_reg_metrics['rmse']:.3f}, R²: {rf_reg_metrics['r2']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # 4. SUPERVISED — CLASSIFICATION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[4/7] Training classification models...")

    roc_data = {}
    pr_data = {}
    cm_data = {}
    cal_data = {}

    # Logistic Regression
    log_model = train_logistic_regression(X_train, y_train_clf, mcfg["supervised"]["logistic_reg"]["params"])
    log_pred = log_model.predict(X_test)
    log_proba = log_model.predict_proba(X_test)[:, 1]
    log_metrics = classification_metrics(y_test_clf, log_pred, log_proba)
    supervised_results["logistic_regression"] = log_metrics
    save_model(log_model, MODELS / "logistic_reg.joblib")
    roc_data["Logistic Reg"] = get_roc_data(y_test_clf, log_proba)
    pr_data["Logistic Reg"] = get_pr_data(y_test_clf, log_proba)
    cm_data["Logistic Reg"] = np.array(log_metrics["confusion_matrix"])
    frac, mean_p = get_calibration_data(y_test_clf, log_proba)
    cal_data["Logistic Reg"] = (frac, mean_p)
    print(f"    Logistic   — AUC: {log_metrics['roc_auc']:.3f}, F1: {log_metrics['f1']:.3f}")

    # Random Forest Classifier
    rf_clf_model = train_random_forest_clf(X_train, y_train_clf, mcfg["supervised"]["random_forest_clf"]["params"])
    rf_clf_pred = rf_clf_model.predict(X_test)
    rf_clf_proba = rf_clf_model.predict_proba(X_test)[:, 1]
    rf_clf_metrics = classification_metrics(y_test_clf, rf_clf_pred, rf_clf_proba)
    supervised_results["random_forest_clf"] = rf_clf_metrics
    save_model(rf_clf_model, MODELS / "random_forest_clf.joblib")
    roc_data["RF Classifier"] = get_roc_data(y_test_clf, rf_clf_proba)
    pr_data["RF Classifier"] = get_pr_data(y_test_clf, rf_clf_proba)
    cm_data["RF Classifier"] = np.array(rf_clf_metrics["confusion_matrix"])
    frac, mean_p = get_calibration_data(y_test_clf, rf_clf_proba)
    cal_data["RF Classifier"] = (frac, mean_p)
    print(f"    RF Clf     — AUC: {rf_clf_metrics['roc_auc']:.3f}, F1: {rf_clf_metrics['f1']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # 5. UNSUPERVISED LEARNING
    # ═════════════════════════════════════════════════════════════════════
    print("\n[5/7] Unsupervised clustering...")
    cluster_cols = [c for c in cfg["data"]["sensor_columns"] + cfg["data"]["environmental_columns"]
                    if c in df_feat.columns]
    X_cluster = df_feat[cluster_cols].values

    km_model, km_labels = train_kmeans(X_cluster, mcfg["unsupervised"]["kmeans"]["params"])
    km_metrics = cluster_metrics(X_cluster, km_labels)
    save_model(km_model, MODELS / "kmeans.joblib")

    gmm_model, gmm_labels = train_gmm(X_cluster, mcfg["unsupervised"]["gmm"]["params"])
    gmm_metrics = cluster_metrics(X_cluster, gmm_labels)
    save_model(gmm_model, MODELS / "gmm.joblib")

    unsupervised_metrics = {
        "kmeans": km_metrics,
        "gmm": gmm_metrics,
    }
    save_json(unsupervised_metrics, METRICS / "unsupervised_metrics.json")

    # Cluster visualization
    print("    Computing t-SNE projection...")
    X_2d = reduce_for_viz(X_cluster)
    cluster_scatter(X_2d, km_labels, str(FIGURES / "cluster_tsne.png"), method="t-SNE")

    # Cluster profiles
    summary = build_cluster_summary(df_feat, km_labels, cluster_cols)
    save_csv(summary, TABLES / "cluster_summary.csv")
    plot_cluster_profiles(summary, str(FIGURES / "cluster_profiles.png"))

    print(f"    K-Means  — Silhouette: {km_metrics['silhouette_score']:.3f}")
    print(f"    GMM      — Silhouette: {gmm_metrics['silhouette_score']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # 6. DEEP LEARNING BENCHMARK
    # ═════════════════════════════════════════════════════════════════════
    print("\n[6/7] Deep learning benchmark...")
    dl_config = mcfg["deep_learning"]["mlp"]

    # Regression MLP
    mlp_reg, reg_history = train_mlp_regressor(X_train, y_train_reg, X_val, y_val_reg, dl_config)
    mlp_reg_pred = mlp_reg.predict(X_test)
    mlp_reg_metrics = regression_metrics(y_test_reg, mlp_reg_pred)
    training_curves(reg_history, str(FIGURES / "nn_training_curves.png"))

    # Classification MLP
    mlp_clf, clf_history = train_mlp_classifier(X_train, y_train_clf, X_val, y_val_clf, dl_config)
    mlp_clf_pred = mlp_clf.predict(X_test)
    mlp_clf_proba = mlp_clf.predict_proba(X_test)[:, 1]
    mlp_clf_metrics = classification_metrics(y_test_clf, mlp_clf_pred, mlp_clf_proba)

    # Add MLP to ROC/PR/CM plots
    roc_data["MLP Classifier"] = get_roc_data(y_test_clf, mlp_clf_proba)
    pr_data["MLP Classifier"] = get_pr_data(y_test_clf, mlp_clf_proba)
    cm_data["MLP Classifier"] = np.array(mlp_clf_metrics["confusion_matrix"])

    deep_metrics = {
        "mlp_regression": mlp_reg_metrics,
        "mlp_classification": mlp_clf_metrics,
        "training_epochs": len(reg_history["train_loss"]),
    }
    save_json(deep_metrics, METRICS / "deep_metrics.json")
    save_model(mlp_reg, MODELS / "nn_regression.joblib")
    save_model(mlp_clf, MODELS / "nn_classifier.joblib")
    print(f"    MLP Reg  — RMSE: {mlp_reg_metrics['rmse']:.3f}, R²: {mlp_reg_metrics['r2']:.3f}")
    print(f"    MLP Clf  — AUC: {mlp_clf_metrics.get('roc_auc', 0):.3f}, F1: {mlp_clf_metrics['f1']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # 7. THRESHOLD ANALYSIS & FINAL PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n[7/7] Threshold analysis & final artifacts...")
    cb = tcfg["cost_benefit"]
    thresholds, f1s, evs, best_f1_t, best_ev_t = threshold_sweep(
        y_test_clf, rf_clf_proba,
        tp_benefit=cb["true_positive_benefit"],
        fp_cost=cb["false_positive_cost"],
        fn_cost=cb["false_negative_cost"],
        tn_benefit=cb["true_negative_benefit"],
    )
    threshold_value_curve(thresholds, f1s, evs, str(FIGURES / "threshold_value_curve.png"))
    print(f"    Best F1 threshold: {best_f1_t:.3f}")
    print(f"    Best EV threshold: {best_ev_t:.3f}")

    # Generate all classification plots
    roc_curves(roc_data, str(FIGURES / "roc_curves.png"))
    pr_curves(pr_data, str(FIGURES / "pr_curves.png"))
    confusion_matrices(cm_data, str(FIGURES / "confusion_matrices.png"))
    calibration_curve_plot(cal_data, str(FIGURES / "calibration_curve.png"))

    # Save all supervised metrics
    save_json(supervised_results, METRICS / "supervised_metrics.json")

    # Model comparison table
    comparison = []
    comparison.append({"Model": "Linear Regression", "Task": "Regression",
                       "RMSE": lr_metrics["rmse"], "MAE": lr_metrics["mae"], "R²": lr_metrics["r2"]})
    comparison.append({"Model": "Random Forest", "Task": "Regression",
                       "RMSE": rf_reg_metrics["rmse"], "MAE": rf_reg_metrics["mae"], "R²": rf_reg_metrics["r2"]})
    comparison.append({"Model": "MLP (Neural Net)", "Task": "Regression",
                       "RMSE": mlp_reg_metrics["rmse"], "MAE": mlp_reg_metrics["mae"], "R²": mlp_reg_metrics["r2"]})
    comparison.append({"Model": "Logistic Regression", "Task": "Classification",
                       "ROC-AUC": log_metrics.get("roc_auc"), "F1": log_metrics["f1"],
                       "Precision": log_metrics["precision"], "Recall": log_metrics["recall"]})
    comparison.append({"Model": "Random Forest", "Task": "Classification",
                       "ROC-AUC": rf_clf_metrics.get("roc_auc"), "F1": rf_clf_metrics["f1"],
                       "Precision": rf_clf_metrics["precision"], "Recall": rf_clf_metrics["recall"]})
    comparison.append({"Model": "MLP (Neural Net)", "Task": "Classification",
                       "ROC-AUC": mlp_clf_metrics.get("roc_auc"), "F1": mlp_clf_metrics["f1"],
                       "Precision": mlp_clf_metrics["precision"], "Recall": mlp_clf_metrics["recall"]})

    comp_df = pd.DataFrame(comparison)
    save_csv(comp_df, TABLES / "model_comparison.csv")

    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"  Figures:  {FIGURES}")
    print(f"  Metrics:  {METRICS}")
    print(f"  Models:   {MODELS}")
    print(f"  Tables:   {TABLES}")
    print("=" * 70)


if __name__ == "__main__":
    main()
