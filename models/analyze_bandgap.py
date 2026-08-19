"""Analysis script for bandgap models (RF, GB unfiltered, GB filtered).
Generates parity plots, feature importance, and comparison table.
"""
import pickle, sys
sys.path.insert(0, "..")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from shared.pipeline import CorrelationFilter  # needed for unpickling
from shared.plotting import (
    plot_parity, plot_residuals,
    plot_aggregated_importance, plot_fold_importance,
)

RESULT_FILES = {
    "RF":              "results/results_rf_bandgap.pkl",
    "GB (unfiltered)": "results/results_gb_bandgap.pkl",
    "GB (filtered)":   "results/results_gb_bandgap_filtered.pkl",
}
COLOR = "#2b6c8f"


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def print_summary(res, label):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(f"  CV Mean R²:  {res['cv_mean_r2']:.4f} "
          f"± {res['cv_std_r2']:.4f}")
    print(f"  Holdout R²:  {res['holdout_r2']:.4f}")
    print(f"  Holdout MAE: {res['holdout_mae']:.4f}")
    print(f"  Gap:         {res['holdout_r2']-res['cv_mean_r2']:+.4f}")
    for fr in res["fold_results"]:
        print(f"  Fold {fr['fold']}: R²={fr['r2']:.4f}")


if __name__ == "__main__":
    results = {}
    for label, path in RESULT_FILES.items():
        try:
            results[label] = load(path)
            print_summary(results[label], label)
        except FileNotFoundError:
            print(f"Warning: {path} not found — skipping {label}")

    if not results:
        print("No result files found. Run model scripts first.")
        sys.exit(1)

    # Comparison table
    print(f"\n{'─'*60}")
    print(f"{'Model':<20} {'CV R²':>8} {'±':>5} {'Holdout R²':>12} "
          f"{'Gap':>8}")
    print(f"{'─'*60}")
    for label, res in results.items():
        print(f"{label:<20} {res['cv_mean_r2']:>8.4f} "
              f"{res['cv_std_r2']:>5.4f} {res['holdout_r2']:>12.4f} "
              f"{res['holdout_r2']-res['cv_mean_r2']:>+8.4f}")

    # Feature importance for GB models
    for label in ["GB (unfiltered)", "GB (filtered)"]:
        if label not in results:
            continue
        tag = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_aggregated_importance(
            results[label],
            save_path=f"../figures/feature_importance_{tag}.png",
            color=COLOR,
            target_label=label,
        )
