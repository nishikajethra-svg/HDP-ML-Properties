"""
MODNet neural network for bandgap prediction.
Replication of De Breuck et al. (2021) on HDP compounds.

Environment: start_modnet
Usage:
    python train_bandgap.py
    python train_bandgap.py fast   # 2 presets only (~30 min)
"""
import warnings
warnings.filterwarnings("ignore")

import sys, pickle
import numpy as np
import pandas as pd
from modnet.preprocessing import MODData
from modnet.models import MODNetModel
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

TARGET = "bandgap"
SELECTED_PKL = f"../data/modnet_selected_{TARGET}.pkl"
SEED = 42
HOLDOUT_FRAC = 0.20


def run(fast=False):
    print(f"Loading selected MODData...", flush=True)
    data = MODData.load(SELECTED_PKL)
    n_features = len(data.get_optimal_descriptors())
    print(f"Features: {n_features}, Compounds: "
          f"{data.df_featurized.shape[0]}", flush=True)

    # Holdout split
    n = data.df_featurized.shape[0]
    rng = np.random.RandomState(SEED)
    indices = rng.permutation(n)
    n_holdout = int(n * HOLDOUT_FRAC)

    holdout_idx = list(indices[:n_holdout])
    cv_idx = list(indices[n_holdout:])
    cv_data, holdout_data = data.split((cv_idx, holdout_idx))

    print(f"Holdout: {n_holdout}, CV: {len(cv_idx)}", flush=True)

    # Nested CV
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        outer_cv.split(np.arange(len(cv_idx))), 1
    ):
        print(f"\nFold {fold_i}/5", flush=True)
        train_data, test_data = cv_data.split(
            (list(train_idx), list(test_idx))
        )
        test_data.num_classes = {TARGET: 0}

        model = MODNetModel(
            targets=[[[TARGET]]],
            weights={TARGET: 1.0},
            n_feat=n_features,
        )
        model.fit_preset(
            train_data, nested=3, fast=fast, refit=True, n_jobs=1,
        )

        y_pred = model.predict(test_data)[TARGET].values
        y_test = test_data.df_targets[TARGET].values
        r2 = r2_score(y_test, y_pred)
        print(f"Fold {fold_i}: R²={r2:.4f}", flush=True)

        fold_results.append({
            "fold": fold_i, "r2": r2,
            "mse": mean_squared_error(y_test, y_pred),
            "mae": mean_absolute_error(y_test, y_pred),
            "y_test": y_test, "y_pred": y_pred,
        })

    r2s = [f["r2"] for f in fold_results]
    print(f"\nCV Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}",
          flush=True)

    # Final model
    print("\nTraining final model...", flush=True)
    final = MODNetModel(
        targets=[[[TARGET]]], weights={TARGET: 1.0}, n_feat=n_features,
    )
    final.fit_preset(cv_data, nested=3, fast=fast, refit=True, n_jobs=1)

    holdout_data.num_classes = {TARGET: 0}
    y_ho_pred = final.predict(holdout_data)[TARGET].values
    y_ho_true = holdout_data.df_targets[TARGET].values
    ho_r2 = r2_score(y_ho_true, y_ho_pred)

    print(f"\n{'='*50}")
    print(f"HOLDOUT: R²={ho_r2:.4f}, "
          f"MAE={mean_absolute_error(y_ho_true, y_ho_pred):.4f}")
    print(f"Gap: {ho_r2 - np.mean(r2s):+.4f}")
    print(f"{'='*50}")

    results = {
        "fold_results": fold_results,
        "cv_mean_r2": np.mean(r2s), "cv_std_r2": np.std(r2s),
        "holdout_r2": ho_r2,
        "holdout_mse": mean_squared_error(y_ho_true, y_ho_pred),
        "holdout_mae": mean_absolute_error(y_ho_true, y_ho_pred),
        "holdout_y_true": y_ho_true, "holdout_y_pred": y_ho_pred,
    }
    with open("results/results_modnet_bandgap.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/results_modnet_bandgap.pkl", flush=True)


if __name__ == "__main__":
    run(fast="fast" in sys.argv)
