"""Analysis script for Ehull GB and MODNet models."""
import pickle, sys
sys.path.insert(0, "..")
from shared.pipeline import CorrelationFilter  # needed for unpickling
from shared.plotting import plot_analysis, plot_aggregated_importance

RESULT_FILES = {
    "GB":     "results/results_gb_ehull.pkl",
    "NoDOS":  "results/results_gb_ehull_nodos.pkl",
    "MODNet": "results/results_modnet_ehull.pkl",
}
COLOR = "#2b6c8f"

if __name__ == "__main__":
    for label, path in RESULT_FILES.items():
        try:
            with open(path, "rb") as f:
                res = pickle.load(f)
        except FileNotFoundError:
            print(f"Warning: {path} not found — skipping {label}")
            continue

        print(f"\n{'='*50}\n{label}")
        print(f"CV R²:       {res['cv_mean_r2']:.4f} ± {res['cv_std_r2']:.4f}")
        print(f"Holdout R²:  {res['holdout_r2']:.4f}")
        print(f"Holdout MAE: {res['holdout_mae']:.4f}")
        print(f"Gap:         {res['holdout_r2']-res['cv_mean_r2']:+.4f}")

        tag = label.lower()
        plot_analysis(
            res, target_label=f"Ehull ({label})", unit="eV/atom",
            save_path=f"../figures/parity_ehull_{tag}.png", color=COLOR,
        )
        if "feature_importances" in res.get("fold_results", [{}])[0]:
            plot_aggregated_importance(
                res,
                save_path=f"../figures/feature_importance_ehull_{tag}.png",
                color=COLOR, target_label=f"Ehull ({label})",
            )
