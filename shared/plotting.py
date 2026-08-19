"""
Shared plotting utilities for HDP property prediction.

Usage:
    from shared.plotting import plot_parity, plot_feature_importance
    from shared.plotting import plot_aggregated_importance, plot_pareto
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ══════════════════════════════════════════════
# Colours
# ══════════════════════════════════════════════

COLORS = {
    "bandgap":  "#2b6c8f",
    "eform":    "#d4883a",
    "ehull":    "#2b6c8f",
    "normal":   "#2b6c8f",
    "corrfilter": "#d4883a",
    "focused":  "#2e7d32",
}


# ══════════════════════════════════════════════
# Parity plots
# ══════════════════════════════════════════════

def plot_parity(y_true, y_pred, r2, title, xlabel, ylabel,
                ax=None, color="#2b6c8f"):
    """Plot true vs predicted values with diagonal reference line.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted values.
    r2 : float
        R² to display in title.
    title : str
        Subplot title.
    xlabel, ylabel : str
        Axis labels.
    ax : matplotlib.Axes or None
        If None, creates a new figure.
    color : str
        Scatter colour.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(y_true, y_pred, alpha=0.4, s=15, color=color,
               edgecolors="none")
    lims = [
        min(np.min(y_true), np.min(y_pred)) - 0.05,
        max(np.max(y_true), np.max(y_pred)) + 0.05,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (R²={r2:.4f})")
    ax.set_aspect("equal")
    return ax


def plot_residuals(y_true_cv, y_pred_cv, y_true_ho, y_pred_ho,
                   xlabel="Residual", ax=None,
                   color_cv="#2b6c8f", color_ho="#d4883a"):
    """Plot residual distributions for CV and holdout sets.

    Parameters
    ----------
    y_true_cv, y_pred_cv : array-like
        CV true and predicted values.
    y_true_ho, y_pred_ho : array-like
        Holdout true and predicted values.
    xlabel : str
        X-axis label.
    ax : matplotlib.Axes or None

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))

    res_cv = np.array(y_true_cv) - np.array(y_pred_cv)
    res_ho = np.array(y_true_ho) - np.array(y_pred_ho)

    ax.hist(res_cv, bins=40, alpha=0.5, color=color_cv,
            label="CV", density=True)
    ax.hist(res_ho, bins=30, alpha=0.5, color=color_ho,
            label="Holdout", density=True)
    ax.axvline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    ax.set_title("Residual distribution")
    ax.legend(fontsize=8)
    return ax


def plot_analysis(results, target_label, unit, save_path,
                  color="#2b6c8f"):
    """Full analysis figure: holdout parity, CV parity, residuals.

    Parameters
    ----------
    results : dict
        Result pkl dict from run_nested_cv_with_holdout.
    target_label : str
        e.g. 'Ehull' for axis labels.
    unit : str
        e.g. 'eV/atom'.
    save_path : str
        Output PNG path.
    color : str
        Plot colour.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    yt_ho = results["holdout_y_true"]
    yp_ho = results["holdout_y_pred"]
    yt_cv = np.concatenate([f["y_test"] for f in results["fold_results"]])
    yp_cv = np.concatenate([f["y_pred"] for f in results["fold_results"]])

    xlabel = f"True {target_label} ({unit})"
    ylabel = f"Predicted {target_label} ({unit})"

    plot_parity(yt_ho, yp_ho, results["holdout_r2"],
                "Holdout parity", xlabel, ylabel,
                ax=axes[0], color=color)

    cv_r2 = results["cv_mean_r2"]
    cv_std = results["cv_std_r2"]
    plot_parity(yt_cv, yp_cv, cv_r2,
                f"CV parity (R²={cv_r2:.4f} ± {cv_std:.4f})",
                xlabel, ylabel, ax=axes[1], color=color)

    plot_residuals(yt_cv, yp_cv, yt_ho, yp_ho,
                   xlabel=f"Residual ({unit})",
                   ax=axes[2], color_cv=color)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Saved to {save_path}", flush=True)
    plt.close()


# ══════════════════════════════════════════════
# Feature importance plots
# ══════════════════════════════════════════════

def _shorten(name, max_len=40):
    """Shorten long feature names for plot readability."""
    if len(name) <= max_len:
        return name
    if "|" in name:
        return name.split("|")[-1][:max_len]
    return name[:max_len]


def plot_fold_importance(fold_result, ax=None, color="#2b6c8f",
                         top_n=15):
    """Plot feature importances for a single fold.

    Parameters
    ----------
    fold_result : dict
        Single entry from results["fold_results"].
    ax : matplotlib.Axes or None
    color : str
    top_n : int
        Number of top features to show.

    Returns
    -------
    matplotlib.Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))

    imp = fold_result["feature_importances"]
    imp_sorted = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    imp_sorted = imp_sorted[:top_n]

    names = [_shorten(n) for n, _ in imp_sorted]
    values = [v for _, v in imp_sorted]
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, color=color, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(
        f"Fold {fold_result['fold']} "
        f"(R²={fold_result['r2']:.4f})",
        fontsize=10,
    )
    return ax


def plot_aggregated_importance(results, save_path, color="#2b6c8f",
                               top_n=25, target_label=""):
    """Plot mean feature importance averaged across all folds with
    error bars and fold-appearance counts.

    Parameters
    ----------
    results : dict
        Result pkl dict.
    save_path : str
        Output PNG path.
    color : str
    top_n : int
    target_label : str
        e.g. 'Eform' for title.
    """
    all_imp = {}
    n_folds = len(results["fold_results"])
    for fr in results["fold_results"]:
        for name, val in fr["feature_importances"].items():
            if name not in all_imp:
                all_imp[name] = []
            all_imp[name].append(val)

    avg_imp = {k: np.mean(v) for k, v in all_imp.items()}
    std_imp = {k: np.std(v) for k, v in all_imp.items()}
    sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1],
                        reverse=True)[:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    names = [_shorten(n) for n, _ in sorted_imp]
    values = [v for _, v in sorted_imp]
    errors = [std_imp[n] for n, _ in sorted_imp]
    y_pos = np.arange(len(names))

    ax.barh(y_pos, values, xerr=errors, color=color, alpha=0.8,
            capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean importance (across folds)")
    ax.set_title(
        f"{target_label} — Top {len(sorted_imp)} features "
        f"(averaged over {n_folds} folds)"
    )

    for i, (name, _) in enumerate(sorted_imp):
        count = len(all_imp[name])
        ax.text(values[i] + errors[i] + 0.001, i,
                f"({count}/{n_folds})",
                va="center", fontsize=7, color="gray")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Saved to {save_path}", flush=True)
    plt.close()


# ══════════════════════════════════════════════
# PySR Pareto front plots
# ══════════════════════════════════════════════

def plot_pareto(equation_csvs, labels, y_var, save_path,
                title="PySR Pareto Fronts",
                colors=None):
    """Plot R² vs complexity Pareto front curves.

    Parameters
    ----------
    equation_csvs : list of str
        Paths to equation CSV files from PySR.
    labels : list of str
        Legend labels for each run.
    y_var : float
        Variance of target (used to compute R² from loss).
    save_path : str
        Output PNG path.
    title : str
    colors : list of str or None
    """
    if colors is None:
        colors = ["#2b6c8f", "#d4883a", "#2e7d32", "#7b1fa2"]

    fig, ax = plt.subplots(figsize=(8, 6))

    for path, label, color in zip(equation_csvs, labels, colors):
        try:
            df = pd.read_csv(path).sort_values("complexity")
            r2s = 1 - df["loss"] / y_var
            ax.plot(df["complexity"], r2s, "o-", label=label,
                    color=color, markersize=3)
        except Exception as e:
            print(f"Warning: could not load {path}: {e}")

    ax.set_xlabel("Complexity")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor="white")
    print(f"Saved to {save_path}", flush=True)
    plt.close()
