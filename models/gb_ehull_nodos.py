"""GB pipeline for Ehull using NoDOS lobster features (no band descriptors).
Used to test whether DOS band features add predictive value -- they do not.
"""
import pickle, sys
sys.path.insert(0, "..")
from shared.pipeline import load_data, run_nested_cv_with_holdout

PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_bonding_NoDOS.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)

if __name__ == "__main__":
    X, y = load_data(target="Ehull_avg", **PATHS)
    results = run_nested_cv_with_holdout(X, y, n_iter=100)
    with open("results/results_gb_ehull_nodos.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/results_gb_ehull_nodos.pkl")
