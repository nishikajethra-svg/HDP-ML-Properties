"""GB pipeline for formation energy prediction."""
import pickle, sys
sys.path.insert(0, "..")
from shared.pipeline import load_data, run_nested_cv_with_holdout

PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_antibonding.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)

if __name__ == "__main__":
    X, y = load_data(target="Eform_avg", **PATHS)
    results = run_nested_cv_with_holdout(X, y)
    with open("results/results_gb_eform.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/results_gb_eform.pkl")
