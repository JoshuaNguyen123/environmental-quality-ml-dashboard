"""Publication-quality plotting utilities."""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# -- House style ----------------------------------------------------------
PALETTE = {
    "primary": "#2C3E50",
    "accent": "#E74C3C",
    "secondary": "#3498DB",
    "muted": "#95A5A6",
    "success": "#27AE60",
    "warning": "#F39C12",
    "bg": "#FAFAFA",
    "grid": "#ECEFF1",
}

FIGSIZE_SINGLE = (8, 5)
FIGSIZE_WIDE = (12, 5)
FIGSIZE_SQUARE = (7, 7)
DPI = 150


def set_style() -> None:
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams.update({
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": "white",
        "axes.edgecolor": PALETTE["muted"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.5,
        "text.color": PALETTE["primary"],
        "axes.labelcolor": PALETTE["primary"],
        "xtick.color": PALETTE["primary"],
        "ytick.color": PALETTE["primary"],
        "font.family": "sans-serif",
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.2,
    })


set_style()


def save_fig(fig: plt.Figure, path: str, close: bool = True) -> None:
    from pathlib import Path
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=fig.get_facecolor())
    if close:
        plt.close(fig)


def correlation_heatmap(corr: "pd.DataFrame", path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Pearson r"},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold", pad=15)
    save_fig(fig, path)


def residual_plot(y_true, y_pred, path: str, model_name: str = "Model") -> None:
    residuals = np.array(y_true) - np.array(y_pred)
    fitted = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE_WIDE)

    # Residual vs Fitted
    ax = axes[0]
    ax.scatter(fitted, residuals, alpha=0.3, s=12, color=PALETTE["secondary"])
    ax.axhline(0, color=PALETTE["accent"], linewidth=1.2, linestyle="--")
    ax.set_xlabel("Fitted Values")
    ax.set_ylabel("Residuals")
    ax.set_title(f"{model_name} -- Residuals vs Fitted")

    # Q-Q plot
    from scipy import stats
    ax = axes[1]
    stats.probplot(residuals, plot=ax)
    ax.set_title(f"{model_name} -- Q-Q Plot")
    ax.get_lines()[0].set(color=PALETTE["secondary"], markersize=3, alpha=0.5)
    ax.get_lines()[1].set(color=PALETTE["accent"], linewidth=1.2)

    fig.tight_layout()
    save_fig(fig, path)


def roc_curves(roc_data: dict, path: str) -> None:
    """roc_data: {name: (fpr, tpr, auc)}"""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [PALETTE["secondary"], PALETTE["accent"], PALETTE["success"], PALETTE["warning"]]
    for (name, (fpr, tpr, auc_val)), c in zip(roc_data.items(), colors):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", color=c, linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves -- Classification Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.05])
    save_fig(fig, path)


def pr_curves(pr_data: dict, path: str) -> None:
    """pr_data: {name: (precision, recall, ap)}"""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [PALETTE["secondary"], PALETTE["accent"], PALETTE["success"], PALETTE["warning"]]
    for (name, (prec, rec, ap)), c in zip(pr_data.items(), colors):
        ax.plot(rec, prec, label=f"{name} (AP={ap:.3f})", color=c, linewidth=1.8)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves", fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    save_fig(fig, path)


def confusion_matrices(cm_data: dict, path: str) -> None:
    """cm_data: {name: confusion_matrix_array}"""
    n = len(cm_data)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, (name, cm) in zip(axes, cm_data.items()):
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "High"], yticklabels=["Normal", "High"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=11, fontweight="bold")
    fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, path)


def calibration_curve_plot(cal_data: dict, path: str) -> None:
    """cal_data: {name: (prob_true, prob_pred)}"""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    colors = [PALETTE["secondary"], PALETTE["accent"], PALETTE["success"]]
    for (name, (pt, pp)), c in zip(cal_data.items(), colors):
        ax.plot(pp, pt, "s-", label=name, color=c, markersize=5)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Perfectly Calibrated")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    save_fig(fig, path)


def cluster_scatter(X_2d, labels, path: str, method: str = "t-SNE") -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    unique = sorted(set(labels))
    colors = sns.color_palette("husl", len(unique))
    for lab, c in zip(unique, colors):
        mask = labels == lab
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[c], label=f"Cluster {lab}",
                   alpha=0.5, s=10, edgecolors="none")
    ax.set_xlabel(f"{method} 1")
    ax.set_ylabel(f"{method} 2")
    ax.set_title(f"Cluster Visualization ({method})", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, markerscale=3)
    save_fig(fig, path)


def cluster_profiles(profiles: "pd.DataFrame", path: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    profiles_t = profiles.set_index("Cluster").T
    profiles_t.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.75)
    ax.set_title("Cluster Feature Profiles (Normalized Means)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Standardized Mean")
    ax.legend(title="Cluster", fontsize=9, bbox_to_anchor=(1.02, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, path)


def training_curves(history: dict, path: str) -> None:
    """history: {'train_loss': [...], 'val_loss': [...]}"""
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], label="Train Loss", color=PALETTE["secondary"], linewidth=1.5)
    ax.plot(epochs, history["val_loss"], label="Val Loss", color=PALETTE["accent"], linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Neural Network Training Curves", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    save_fig(fig, path)


def threshold_value_curve(thresholds, f1s, evs, path: str) -> None:
    fig, ax1 = plt.subplots(figsize=FIGSIZE_SINGLE)
    ax1.plot(thresholds, f1s, color=PALETTE["secondary"], linewidth=1.8, label="F1 Score")
    ax1.set_xlabel("Decision Threshold")
    ax1.set_ylabel("F1 Score", color=PALETTE["secondary"])
    ax1.tick_params(axis="y", labelcolor=PALETTE["secondary"])

    ax2 = ax1.twinx()
    ax2.plot(thresholds, evs, color=PALETTE["accent"], linewidth=1.8, linestyle="--", label="Expected Value")
    ax2.set_ylabel("Expected Value", color=PALETTE["accent"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["accent"])

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower center", fontsize=9)
    ax1.set_title("Threshold Optimization -- F1 & Expected Value", fontsize=13, fontweight="bold")
    save_fig(fig, path)
