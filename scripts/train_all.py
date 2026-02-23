#!/usr/bin/env python3
"""train_all.py - Run the complete ML pipeline end-to-end.

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
    train_extra_trees_reg, train_hist_gradient_boosting_reg,
    train_extra_trees_clf, train_hist_gradient_boosting_clf,
)
from src.models.unsupervised import (
    train_kmeans, train_gmm, cluster_metrics,
    build_cluster_summary, reduce_for_viz, train_agglomerative, train_dbscan,
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
    print("  AIR QUALITY ML PIPELINE - Full Training Run")
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
    # 3. SUPERVISED - REGRESSION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[3/7] Training regression models...")
    supervised_results = {}

    reg_trainers = {
        "linear_regression": lambda: train_linear_regression(X_train, y_train_reg),
        "random_forest_reg": lambda: train_random_forest_reg(X_train, y_train_reg, mcfg["supervised"]["random_forest_reg"]["params"]),
        "extra_trees_reg": lambda: train_extra_trees_reg(X_train, y_train_reg, mcfg["supervised"]["extra_trees_reg"]["params"]),
        "hist_gradient_boosting_reg": lambda: train_hist_gradient_boosting_reg(
            X_train, y_train_reg, mcfg["supervised"]["hist_gradient_boosting_reg"]["params"]
        ),
    }

    regression_predictions = {}
    regression_metrics_by_model = {}
    for name, trainer in reg_trainers.items():
        model = trainer()
        pred = model.predict(X_test)
        metrics = regression_metrics(y_test_reg, pred)
        supervised_results[name] = metrics
        save_model(model, MODELS / f"{name}.joblib")
        residual_plot(y_test_reg, pred, str(FIGURES / f"residuals_{name}.png"), name.replace("_", " ").title())
        regression_predictions[name] = pred
        regression_metrics_by_model[name] = metrics
        print(f"    {name:<24} - RMSE: {metrics['rmse']:.3f}, R2: {metrics['r2']:.3f}")

    # Keep detailed diagnostics on linear regression for interpretability continuity
    lr_pred = regression_predictions["linear_regression"]
    residuals = y_test_reg - lr_pred
    diag = residual_diagnostics(residuals)
    vif = variance_inflation_factor(X_train, feature_cols)
    supervised_results["linear_regression"]["diagnostics"] = diag
    supervised_results["linear_regression"]["vif"] = vif

    # ═════════════════════════════════════════════════════════════════════
    # 4. SUPERVISED - CLASSIFICATION
    # ═════════════════════════════════════════════════════════════════════
    print("\n[4/7] Training classification models...")

    roc_data = {}
    pr_data = {}
    cm_data = {}
    cal_data = {}

    clf_trainers = {
        "logistic_regression": lambda: train_logistic_regression(X_train, y_train_clf, mcfg["supervised"]["logistic_reg"]["params"]),
        "random_forest_clf": lambda: train_random_forest_clf(X_train, y_train_clf, mcfg["supervised"]["random_forest_clf"]["params"]),
        "extra_trees_clf": lambda: train_extra_trees_clf(X_train, y_train_clf, mcfg["supervised"]["extra_trees_clf"]["params"]),
        "hist_gradient_boosting_clf": lambda: train_hist_gradient_boosting_clf(
            X_train, y_train_clf, mcfg["supervised"]["hist_gradient_boosting_clf"]["params"]
        ),
    }

    classification_probabilities = {}
    classification_metrics_by_model = {}
    for name, trainer in clf_trainers.items():
        model = trainer()
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test_clf, pred, proba)
        supervised_results[name] = metrics
        save_model(model, MODELS / f"{name}.joblib")

        pretty = name.replace("_", " ").title()
        roc_data[pretty] = get_roc_data(y_test_clf, proba)
        pr_data[pretty] = get_pr_data(y_test_clf, proba)
        cm_data[pretty] = np.array(metrics["confusion_matrix"])
        frac, mean_p = get_calibration_data(y_test_clf, proba)
        cal_data[pretty] = (frac, mean_p)
        classification_probabilities[name] = proba
        classification_metrics_by_model[name] = metrics

        print(f"    {name:<24} - AUC: {metrics['roc_auc']:.3f}, F1: {metrics['f1']:.3f}")

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

    agg_model, agg_labels = train_agglomerative(X_cluster, mcfg["unsupervised"]["agglomerative"]["params"])
    agg_metrics = cluster_metrics(X_cluster, agg_labels)
    save_model(agg_model, MODELS / "agglomerative.joblib")

    dbscan_model, dbscan_labels = train_dbscan(X_cluster, mcfg["unsupervised"]["dbscan"]["params"])
    dbscan_metrics = cluster_metrics(X_cluster, dbscan_labels)
    save_model(dbscan_model, MODELS / "dbscan.joblib")

    unsupervised_metrics = {
        "kmeans": km_metrics,
        "gmm": gmm_metrics,
        "agglomerative": agg_metrics,
        "dbscan": dbscan_metrics,
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

    print(f"    K-Means  - Silhouette: {km_metrics['silhouette_score']:.3f}")
    print(f"    GMM      - Silhouette: {gmm_metrics['silhouette_score']:.3f}")
    print(f"    Agglom   - Silhouette: {agg_metrics['silhouette_score']:.3f}")
    print(f"    DBSCAN   - Silhouette: {dbscan_metrics['silhouette_score']:.3f}")

    # ═════════════════════════════════════════════════════════════════════
    # 6. DEEP LEARNING BENCHMARK
    # ═════════════════════════════════════════════════════════════════════
    print("\n[6/7] Deep learning benchmark...")
    deep_profiles = {k: v for k, v in mcfg["deep_learning"].items() if k.startswith("mlp")}
    deep_metrics = {}

    primary_reg_metrics = None
    primary_clf_metrics = None
    primary_reg_history = None

    for idx, (profile_name, dl_config) in enumerate(deep_profiles.items()):
        reg_model, reg_history = train_mlp_regressor(X_train, y_train_reg, X_val, y_val_reg, dl_config)
        reg_pred = reg_model.predict(X_test)
        reg_metrics = regression_metrics(y_test_reg, reg_pred)
        save_model(reg_model, MODELS / f"{profile_name}_regression.joblib")

        clf_model, clf_history = train_mlp_classifier(X_train, y_train_clf, X_val, y_val_clf, dl_config)
        clf_pred = clf_model.predict(X_test)
        clf_proba = clf_model.predict_proba(X_test)[:, 1]
        clf_metrics = classification_metrics(y_test_clf, clf_pred, clf_proba)
        save_model(clf_model, MODELS / f"{profile_name}_classifier.joblib")

        pretty = f"MLP {profile_name.replace('mlp', '').replace('_', ' ').strip() or 'Baseline'}".strip()
        roc_data[pretty] = get_roc_data(y_test_clf, clf_proba)
        pr_data[pretty] = get_pr_data(y_test_clf, clf_proba)
        cm_data[pretty] = np.array(clf_metrics["confusion_matrix"])

        deep_metrics[f"{profile_name}_regression"] = reg_metrics
        deep_metrics[f"{profile_name}_classification"] = clf_metrics
        deep_metrics[f"{profile_name}_training_epochs"] = len(reg_history["train_loss"])

        if idx == 0:
            primary_reg_metrics = reg_metrics
            primary_clf_metrics = clf_metrics
            primary_reg_history = reg_history
            # Backward-compatible keys consumed by existing report/dashboard
            deep_metrics["mlp_regression"] = reg_metrics
            deep_metrics["mlp_classification"] = clf_metrics
            deep_metrics["training_epochs"] = len(reg_history["train_loss"])
            save_model(reg_model, MODELS / "nn_regression.joblib")
            save_model(clf_model, MODELS / "nn_classifier.joblib")

        print(
            f"    {profile_name:<16} - RMSE: {reg_metrics['rmse']:.3f}, "
            f"AUC: {clf_metrics.get('roc_auc', 0):.3f}, F1: {clf_metrics['f1']:.3f}"
        )

    if primary_reg_history is not None:
        training_curves(primary_reg_history, str(FIGURES / "nn_training_curves.png"))

    save_json(deep_metrics, METRICS / "deep_metrics.json")

    # ═════════════════════════════════════════════════════════════════════
    # 7. THRESHOLD ANALYSIS & FINAL PLOTS
    # ═════════════════════════════════════════════════════════════════════
    print("\n[7/7] Threshold analysis & final artifacts...")
    cb = tcfg["cost_benefit"]
    threshold_proba = classification_probabilities.get("random_forest_clf")
    if threshold_proba is None:
        threshold_proba = next(iter(classification_probabilities.values()))
    thresholds, f1s, evs, best_f1_t, best_ev_t = threshold_sweep(
        y_test_clf, threshold_proba,
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
    for model_name, metrics in regression_metrics_by_model.items():
        comparison.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Task": "Regression",
                "RMSE": metrics["rmse"],
                "MAE": metrics["mae"],
                "R²": metrics["r2"],
            }
        )

    if primary_reg_metrics is not None:
        comparison.append(
            {
                "Model": "MLP (Neural Net)",
                "Task": "Regression",
                "RMSE": primary_reg_metrics["rmse"],
                "MAE": primary_reg_metrics["mae"],
                "R²": primary_reg_metrics["r2"],
            }
        )

    for model_name, metrics in classification_metrics_by_model.items():
        comparison.append(
            {
                "Model": model_name.replace("_", " ").title(),
                "Task": "Classification",
                "ROC-AUC": metrics.get("roc_auc"),
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
            }
        )

    if primary_clf_metrics is not None:
        comparison.append(
            {
                "Model": "MLP (Neural Net)",
                "Task": "Classification",
                "ROC-AUC": primary_clf_metrics.get("roc_auc"),
                "F1": primary_clf_metrics["f1"],
                "Precision": primary_clf_metrics["precision"],
                "Recall": primary_clf_metrics["recall"],
            }
        )

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
