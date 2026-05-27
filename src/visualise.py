"""
=============================================================================
visualise.py
Generate all result plots:
  1. Actual vs Predicted (all models)
  2. Metrics bar chart comparison
  3. Training & validation loss curves
=============================================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import yaml

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

RESULTS_DIR = config["paths"]["results"]
os.makedirs(RESULTS_DIR, exist_ok=True)

BG_DARK  = "#0d1117"
BG_CARD  = "#161b22"
BORDER   = "#30363d"
COLORS   = ["#00e5ff", "#76ff03", "#ff6d00", "#ea00d9", "#ffe000"]


def _apply_dark(ax):
    ax.set_facecolor(BG_CARD)
    ax.tick_params(colors="gray")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(BORDER)


# ── 1. Actual vs Predicted ────────────────────────────────────────────────────
def plot_predictions(results, model_names):
    n = len(model_names)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), facecolor=BG_DARK)
    fig.suptitle("Ganga River WQI — Actual vs Predicted",
                 color="white", fontsize=15, fontweight="bold")

    for i, name in enumerate(model_names):
        ax  = axes[i]
        m   = results[name]["metrics"]
        _apply_dark(ax)

        y_true = results[name]["y_true"]
        y_pred = results[name]["y_pred"]

        ax.plot(y_true, color="white", linewidth=1.2, label="Actual",    alpha=0.85)
        ax.plot(y_pred, color=COLORS[i], linewidth=1.2, label=f"{name} Predicted")
        ax.fill_between(range(len(y_true)), y_true, y_pred,
                        alpha=0.08, color=COLORS[i])

        title = (f"{name}   |   RMSE={m['RMSE']:.4f}   MSE={m['MSE']:.4f}   "
                 f"MAE={m['MAE']:.4f}   MAPE={m['MAPE']:.2f}%   R²={m['R2']:.4f}")
        ax.set_title(title, color="white", fontsize=9.5, pad=8)
        ax.set_ylabel("WQI", color="gray", fontsize=9)
        ax.legend(facecolor=BG_CARD, labelcolor="white", fontsize=9, loc="upper right")

    axes[-1].set_xlabel("Test Samples (days)", color="gray", fontsize=9)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, "predictions_all_models.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"[visualise] Saved: {path}")


# ── 2. Metrics Bar Chart ─────────────────────────────────────────────────────
def plot_metrics_comparison(results, model_names):
    metric_keys = ["RMSE", "MSE", "MAE", "MAPE", "R2"]
    fig, axes   = plt.subplots(1, 5, figsize=(20, 5), facecolor=BG_DARK)
    fig.suptitle("Metric Comparison — All Models",
                 color="white", fontsize=14, fontweight="bold")

    for j, metric in enumerate(metric_keys):
        ax     = axes[j]
        values = [results[n]["metrics"][metric] for n in model_names]
        _apply_dark(ax)

        bars = ax.bar(model_names, values, color=COLORS, width=0.55, edgecolor=BORDER)
        ax.set_title(metric, color="white", fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=30, ha="right",
                           color="white", fontsize=8)

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.02,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="white", fontsize=7.5)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"[visualise] Saved: {path}")


# ── 3. Loss Curves ───────────────────────────────────────────────────────────
def plot_loss_curves(results, model_names):
    n    = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), facecolor=BG_DARK)
    fig.suptitle("Training vs Validation Loss",
                 color="white", fontsize=13, fontweight="bold")

    for i, name in enumerate(model_names):
        ax   = axes[i]
        hist = results[name]["history"].history
        _apply_dark(ax)

        ax.plot(hist["loss"],     color=COLORS[i], linewidth=1.5, label="Train")
        ax.plot(hist["val_loss"], color="white",   linewidth=1.2,
                linestyle="--", label="Val", alpha=0.8)
        ax.set_title(name, color="white", fontsize=10, fontweight="bold")
        ax.set_xlabel("Epoch", color="gray", fontsize=8)
        ax.set_ylabel("MSE Loss", color="gray", fontsize=8)
        ax.legend(facecolor=BG_CARD, labelcolor="white", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"[visualise] Saved: {path}")


# ── 4. Scatter: Actual vs Predicted ─────────────────────────────────────────
def plot_scatter(results, model_names):
    n    = len(model_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), facecolor=BG_DARK)
    fig.suptitle("Scatter: Actual vs Predicted WQI",
                 color="white", fontsize=13, fontweight="bold")

    for i, name in enumerate(model_names):
        ax     = axes[i]
        y_true = results[name]["y_true"]
        y_pred = results[name]["y_pred"]
        r2     = results[name]["metrics"]["R2"]
        _apply_dark(ax)

        ax.scatter(y_true, y_pred, color=COLORS[i], alpha=0.5, s=10)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        ax.plot([lo, hi], [lo, hi], "w--", linewidth=1, alpha=0.6)

        ax.set_title(f"{name}\nR²={r2:.4f}", color="white", fontsize=9)
        ax.set_xlabel("Actual",    color="gray", fontsize=8)
        ax.set_ylabel("Predicted", color="gray", fontsize=8)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "scatter_actual_vs_predicted.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
    plt.close()
    print(f"[visualise] Saved: {path}")


# ── Generate all plots ────────────────────────────────────────────────────────
def generate_all_plots(results, model_names):
    print("\n[visualise] Generating all plots...")
    plot_predictions(results, model_names)
    plot_metrics_comparison(results, model_names)
    plot_loss_curves(results, model_names)
    plot_scatter(results, model_names)
    print("[visualise] All plots saved to results/")