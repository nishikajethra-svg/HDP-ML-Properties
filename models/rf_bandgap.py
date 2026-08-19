"""Random Forest baseline for bandgap prediction."""
import pickle
import sys
import numpy as np
sys.path.insert(0, "..")

from shared.pipeline import load_data, run_nested_cv_with_holdout
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.pipeline import CorrelationFilter
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import GradientBoostingRegressor

PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_antibonding.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)

PARAM_DIST = {
    "rf__n_estimators": [100, 300, 500],
    "rf__max_depth": [5, 10, 20, None],
    "rf__min_samples_split": [2, 5, 10],
    "rf__min_samples_leaf": [1, 2, 4],
    "rf__max_features": ["sqrt", "log2", 0.5],
}


def build_rf_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("corrfilter", CorrelationFilter(threshold=0.9)),
        ("selector", SelectFromModel(
            GradientBoostingRegressor(
                n_estimators=500, max_depth=3, random_state=42
            ),
            threshold="mean",
        )),
        ("rf", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])


if __name__ == "__main__":
    X, y = load_data(target="bandgap", **PATHS)

    # Use shared nested CV but with RF pipeline
    # Simpler: run a single 5-fold CV for RF baseline
    from collections import Counter
    from sklearn.model_selection import KFold, RandomizedSearchCV

    n = X.shape[0]
    rng = np.random.RandomState(42)
    indices = rng.permutation(n)
    n_holdout = int(n * 0.20)
    holdout_idx = indices[:n_holdout]
    cv_idx = indices[n_holdout:]

    X_holdout = X.values[holdout_idx]
    y_holdout = y[holdout_idx]
    X_cv = X.values[cv_idx]
    y_cv = y[cv_idx]

    print(f"Holdout: {len(holdout_idx)}, CV: {len(cv_idx)}", flush=True)

    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_cv), 1
    ):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

        pipeline = build_rf_pipeline()
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        search = RandomizedSearchCV(
            pipeline, PARAM_DIST, n_iter=50,
            cv=inner_cv, scoring="r2",
            n_jobs=-1, refit=True, random_state=42,
        )
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        print(f"Fold {fold_i}: R²={r2:.4f}, MAE={mae:.4f}, "
              f"params={search.best_params_}", flush=True)

        fold_results.append({
            "fold": fold_i, "r2": r2,
            "mse": mean_squared_error(y_test, y_pred), "mae": mae,
            "best_params": search.best_params_,
            "y_test": y_test, "y_pred": y_pred,
        })

    r2s = [f["r2"] for f in fold_results]
    print(f"\nCV Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}",
          flush=True)

    # Final model
    param_counts = Counter(
        tuple(sorted(f["best_params"].items())) for f in fold_results
    )
    best_params = dict(param_counts.most_common(1)[0][0])
    final = build_rf_pipeline()
    final.set_params(**best_params)
    final.fit(X_cv, y_cv)

    y_ho_pred = final.predict(X_holdout)
    ho_r2 = r2_score(y_holdout, y_ho_pred)
    print(f"\nHoldout R²: {ho_r2:.4f}", flush=True)
    print(f"Gap: {ho_r2 - np.mean(r2s):+.4f}", flush=True)

    results = {
        "fold_results": fold_results,
        "cv_mean_r2": np.mean(r2s), "cv_std_r2": np.std(r2s),
        "holdout_r2": ho_r2,
        "holdout_mse": mean_squared_error(y_holdout, y_ho_pred),
        "holdout_mae": mean_absolute_error(y_holdout, y_ho_pred),
        "holdout_y_true": y_holdout, "holdout_y_pred": y_ho_pred,
        "final_params": best_params, "final_pipeline": final,
        "feature_names": list(X.columns),
    }

    with open("results/results_rf_bandgap.pkl", "wb") as f:
        pickle.dump(results, f)
    print("Saved to results/results_rf_bandgap.pkl", flush=True)
