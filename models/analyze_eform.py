"""Analysis script for Eform GB model."""
import pickle, sys
sys.path.insert(0, "..")
from shared.pipeline import CorrelationFilter  # needed for unpickling
from shared.plotting import plot_analysis, plot_aggregated_importance

RESULT_FILE = "results/results_gb_eform.pkl"
COLOR = "#d4883a"

if __name__ == "__main__":
    with open(RESULT_FILE, "rb") as f:
        res = pickle.load(f)

    print(f"CV Mean R²:  {res['cv_mean_r2']:.4f} ± {res['cv_std_r2']:.4f}")
    print(f"Holdout R²:  {res['holdout_r2']:.4f}")
    print(f"Holdout MAE: {res['holdout_mae']:.4f}")
    print(f"Gap:         {res['holdout_r2']-res['cv_mean_r2']:+.4f}")

    for fr in res["fold_results"]:
        top3 = sorted(fr["feature_importances"].items(),
                      key=lambda x: x[1], reverse=True)[:3]
        top3_str = ", ".join(
            f"{n.split('|')[-1][:30]}={v:.3f}" for n, v in top3
        )
        print(f"Fold {fr['fold']}: R²={fr['r2']:.4f} | {top3_str}")

    plot_analysis(
        res, target_label="Eform", unit="eV/atom",
        save_path="../figures/parity_eform.png", color=COLOR,
    )
    plot_aggregated_importance(
        res,
        save_path="../figures/feature_importance_eform.png",
        color=COLOR, target_label="Eform",
    )
