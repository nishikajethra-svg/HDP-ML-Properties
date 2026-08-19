"""
Shared ML pipeline for HDP property prediction.

Contains all reusable components:
- MODData pickle loading (without modnet dependency)
- Data loading and merging (MODNet + lobster + asymmetry features)
- CorrelationFilter transformer
- Pipeline building and nested CV with holdout validation

Usage:
    from shared.pipeline import load_data, run_nested_cv_with_holdout
    X, y = load_data(target="Ehull_avg")
    results = run_nested_cv_with_holdout(X, y)
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import types
import pickle
import importlib.abc
import importlib.machinery
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ══════════════════════════════════════════════
# MODData pickle loader (no modnet dependency)
# ══════════════════════════════════════════════

class _FlexClass:
    """Accepts any constructor args — stands in for pymatgen/matminer
    classes during unpickling."""
    def __init__(self, *a, **kw):
        pass


class _FlexUnpickler(pickle.Unpickler):
    """Unpickler that auto-stubs any missing modnet/matminer/pymatgen
    classes."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            return type(name.replace(".", "_"), (_FlexClass,), {})


class _MockFinder(importlib.abc.MetaPathFinder):
    """Import hook that stubs modnet/matminer/pymatgen/tensorflow."""
    _PREFIXES = (
        "matminer", "pymatgen", "modnet", "tensorflow", "keras", "tf_keras",
    )

    def find_spec(self, name, path, target=None):
        for p in self._PREFIXES:
            if name == p or name.startswith(p + "."):
                return importlib.machinery.ModuleSpec(
                    name, _MockLoader(), is_package=True
                )
        return None


class _MockLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return types.ModuleType(spec.name)

    def exec_module(self, module):
        module.__path__ = []


_mock_installed = False


def load_moddata_pkl(path):
    """Load a MODData pickle file without requiring modnet to be installed.

    Parameters
    ----------
    path : str
        Path to the .pkl file saved by MODData.save().

    Returns
    -------
    object
        The unpickled MODData object with .df_featurized, .df_targets,
        and .optimal_features attributes.
    """
    global _mock_installed
    if not _mock_installed:
        sys.meta_path.insert(0, _MockFinder())
        _mock_installed = True

    with open(path, "rb") as f:
        return _FlexUnpickler(f).load()


# ══════════════════════════════════════════════
# CorrelationFilter
# ══════════════════════════════════════════════

class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one feature from each highly-correlated pair, keeping
    whichever has higher correlation with the target.

    Parameters
    ----------
    threshold : float, default=0.9
        Absolute correlation threshold above which one feature is dropped.
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        corr = np.abs(np.corrcoef(X, rowvar=False))
        n = X.shape[1]
        drop = set()

        if y is not None:
            target_corr = np.abs(np.array(
                [np.corrcoef(X[:, i], y)[0, 1] for i in range(n)]
            ))
        else:
            target_corr = np.zeros(n)

        for i in range(n):
            if i in drop:
                continue
            for j in range(i + 1, n):
                if j in drop:
                    continue
                if corr[i, j] >= self.threshold:
                    if target_corr[i] >= target_corr[j]:
                        drop.add(j)
                    else:
                        drop.add(i)
                        break

        self.keep_mask_ = np.array([i not in drop for i in range(n)])
        return self

    def transform(self, X):
        return X[:, self.keep_mask_]


# ══════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════

# Default paths — override these when calling load_data()
DEFAULT_PKL_PATH = "modnet_selected.pkl"
DEFAULT_LOBDATA_PATH = "hdp_lobfeats_antibonding.csv"
DEFAULT_ORIGINAL_CSV_PATH = "HDP_CombinedInfo_WithStructures.csv"
DEFAULT_TARGET_CSV_PATH = "MLIP_Ehullform_averaged.csv"

ASYMMETRY_COLS = [
    "Icohp.B1.axial_asym_index",
    "Icohp.B1.directional_asym_index",
    "Icobi.B1.axial_asym_index",
    "Icobi.B1.directional_asym_index",
    "Icohp.B2.axial_asym_index",
    "Icohp.B2.directional_asym_index",
    "Icobi.B2.axial_asym_index",
    "Icobi.B2.directional_asym_index",
]


def _fix_vacancy_names(orig_index, modnet_index):
    """Map vacancy-ordered compound names between CSV (Cs2TeVacCl6)
    and MODData (Cs2TeCl6) naming conventions.

    Returns a dict mapping CSV names to MODData names.
    """
    csv_names = set(orig_index)
    mod_names = set(modnet_index)
    missing = mod_names - csv_names

    if not missing:
        return {}

    csv_no_vac = {
        name.replace("Vac", ""): name
        for name in csv_names if "Vac" in name
    }
    name_map = {}
    for m in missing:
        if m in csv_no_vac:
            name_map[m] = csv_no_vac[m]

    return {v: k for k, v in name_map.items()}


def load_data(
    target="bandgap",
    pkl_path=DEFAULT_PKL_PATH,
    lobdata_path=DEFAULT_LOBDATA_PATH,
    original_csv_path=DEFAULT_ORIGINAL_CSV_PATH,
    target_csv_path=DEFAULT_TARGET_CSV_PATH,
    n_features=300,
    bandgap_filter=None,
):
    """Load and merge features from MODNet + LobsterPy + asymmetry indices.

    Parameters
    ----------
    target : str
        Target property name. Use "bandgap" to load from MODData targets,
        or "Ehull_avg" / "Eform_avg" to load from target_csv_path.
    pkl_path : str
        Path to MODData pickle with NMI-selected features.
    lobdata_path : str
        Path to LobsterPy features CSV.
    original_csv_path : str
        Path to HDP_CombinedInfo_WithStructures.csv for asymmetry indices.
    target_csv_path : str
        Path to MLIP_Ehullform_averaged.csv for Ehull/Eform targets.
    n_features : int
        Number of MODNet optimal features to use.
    bandgap_filter : float or None
        If set, filter out compounds with bandgap <= this value.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    """
    # 1. MODNet features
    data = load_moddata_pkl(pkl_path)
    selected_features = data.optimal_features[:n_features]
    X_modnet = data.df_featurized[selected_features].copy()
    print(f"MODNet selected features: {X_modnet.shape[1]} "
          f"({X_modnet.shape[0]} compounds)", flush=True)

    # 2. Target
    if target == "bandgap":
        y_series = data.df_targets["bandgap"].copy()
    else:
        if not os.path.exists(target_csv_path):
            raise FileNotFoundError(
                f"{target_csv_path} not found. This file contains "
                f"MLIP-averaged Ehull and Eform values."
            )
        target_df = pd.read_csv(target_csv_path, index_col=0)
        target_df.index = X_modnet.index
        y_series = target_df[target].copy()
        print(f"Target: {target}", flush=True)

    # 3. Drop NaN targets
    valid_mask = ~y_series.isna()
    n_dropped = (~valid_mask).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped} compounds with NaN target, "
              f"{valid_mask.sum()} remain", flush=True)

    # 4. Bandgap filter (optional)
    if bandgap_filter is not None and target == "bandgap":
        bg_mask = data.df_targets["bandgap"].values > bandgap_filter
        valid_mask = valid_mask & bg_mask
        print(f"After filtering bandgap <= {bandgap_filter} eV: "
              f"{valid_mask.sum()} compounds remain", flush=True)

    X_modnet = X_modnet[valid_mask]
    y_series = y_series[valid_mask]

    # 5. Lobster features
    if not os.path.exists(lobdata_path):
        raise FileNotFoundError(
            f"{lobdata_path} not found. This file contains LobsterPy "
            f"bonding descriptors."
        )
    lob = pd.read_csv(lobdata_path, index_col=0)
    lob.index = data.df_featurized.index
    lob = lob[valid_mask]
    print(f"Lobdata features: {lob.shape[1]}", flush=True)

    # 6. Asymmetry indices
    if not os.path.exists(original_csv_path):
        raise FileNotFoundError(
            f"{original_csv_path} not found. Download from: "
            f"[source repository URL] and place in the data/ directory."
        )
    orig = pd.read_csv(
        original_csv_path, usecols=["comp_name_full"] + ASYMMETRY_COLS
    )
    orig = orig.set_index("comp_name_full")

    # Fix vacancy naming (Cs2TeCl6 <-> Cs2TeVacCl6)
    reverse_map = _fix_vacancy_names(orig.index, X_modnet.index)
    if reverse_map:
        orig = orig.rename(index=reverse_map)
        print(f"Mapped {len(reverse_map)} vacancy-ordered compounds",
              flush=True)

    asym = orig.loc[X_modnet.index, ASYMMETRY_COLS]
    print(f"Asymmetry features: {asym.shape[1]}", flush=True)

    # 7. Merge
    X = pd.concat([X_modnet, lob, asym], axis=1)
    y = y_series.values
    print(f"Total features before NaN handling: {X.shape[1]}", flush=True)

    # 8. Impute NaNs
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"Imputing {nan_count} NaN values with column means",
              flush=True)
        X = X.fillna(X.mean())

    # 9. Drop all-NaN columns
    all_nan = X.columns[X.isna().all()]
    if len(all_nan) > 0:
        print(f"Dropping {len(all_nan)} all-NaN columns", flush=True)
        X = X.drop(columns=all_nan)

    print(f"Final feature matrix: {X.shape[0]} compounds x "
          f"{X.shape[1]} features", flush=True)
    return X, y


# ══════════════════════════════════════════════
# Pipeline building
# ══════════════════════════════════════════════

def build_pipeline():
    """Build the standard GB pipeline with CorrelationFilter and
    SelectFromModel."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("corrfilter", CorrelationFilter(threshold=0.9)),
        ("selector", SelectFromModel(
            GradientBoostingRegressor(
                n_estimators=500, max_depth=3, random_state=42
            ),
            threshold="mean",
        )),
        ("gb", GradientBoostingRegressor(random_state=42)),
    ])


def get_param_distributions():
    """Return hyperparameter distributions for RandomizedSearchCV."""
    return {
        "gb__n_estimators": [500, 1000],
        "gb__max_depth": [3, 4, 5, 6],
        "gb__learning_rate": [0.03, 0.05, 0.1],
        "gb__min_samples_leaf": [1, 4],
        "gb__subsample": [0.6, 0.8, 1.0],
    }


# ══════════════════════════════════════════════
# Nested CV with holdout validation
# ══════════════════════════════════════════════

def _extract_fold_importances(best_pipe, feature_names):
    """Trace feature importances through the pipeline back to original
    feature names."""
    mask_corr = best_pipe.named_steps["corrfilter"].keep_mask_
    names_after_corr = [
        f for f, k in zip(feature_names, mask_corr) if k
    ]
    mask_sel = best_pipe.named_steps["selector"].get_support()
    names_after_sel = [
        f for f, k in zip(names_after_corr, mask_sel) if k
    ]
    gb_importances = best_pipe.named_steps["gb"].feature_importances_
    return dict(zip(names_after_sel, gb_importances))


def run_nested_cv_with_holdout(
    X, y,
    holdout_frac=0.20,
    n_outer=5,
    n_inner=3,
    n_iter=100,
    seed=42,
):
    """Run nested cross-validation with holdout validation.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : np.ndarray
        Target values.
    holdout_frac : float
        Fraction of data to hold out before CV.
    n_outer : int
        Number of outer CV folds.
    n_inner : int
        Number of inner CV folds for hyperparameter tuning.
    n_iter : int
        Number of random hyperparameter combinations to try.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        Results including fold-level metrics, holdout metrics,
        feature importances, and the final trained pipeline.
    """
    feature_names = list(X.columns)
    n = X.shape[0]
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    # Holdout split
    n_holdout = int(n * holdout_frac)
    holdout_idx = indices[:n_holdout]
    cv_idx = indices[n_holdout:]

    X_holdout = X.values[holdout_idx]
    y_holdout = y[holdout_idx]
    X_cv = X.values[cv_idx]
    y_cv = y[cv_idx]

    print(f"\nHoldout set: {len(holdout_idx)} compounds "
          f"({holdout_frac*100:.0f}%)", flush=True)
    print(f"CV set: {len(cv_idx)} compounds "
          f"({(1-holdout_frac)*100:.0f}%)", flush=True)

    # Nested CV
    outer_cv = KFold(n_splits=n_outer, shuffle=True, random_state=seed)
    param_dist = get_param_distributions()

    fold_results = []

    for fold_i, (train_idx, test_idx) in enumerate(
        outer_cv.split(X_cv), 1
    ):
        X_train, X_test = X_cv[train_idx], X_cv[test_idx]
        y_train, y_test = y_cv[train_idx], y_cv[test_idx]

        pipeline = build_pipeline()
        inner_cv = KFold(
            n_splits=n_inner, shuffle=True, random_state=seed
        )
        search = RandomizedSearchCV(
            pipeline, param_dist,
            n_iter=n_iter, cv=inner_cv, scoring="r2",
            n_jobs=-1, refit=True, random_state=seed,
        )
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        best_pipe = search.best_estimator_
        n_after_corr = best_pipe.named_steps["corrfilter"].keep_mask_.sum()
        n_after_sel = best_pipe.named_steps["selector"].get_support().sum()

        fold_importance = _extract_fold_importances(
            best_pipe, feature_names
        )

        print(
            f"Fold {fold_i}: CorrelationFilter kept {n_after_corr} of "
            f"{X_train.shape[1]}, SelectFromModel kept {n_after_sel}",
            flush=True,
        )
        print(
            f"Fold {fold_i}: R2={r2:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, "
            f"best_params={search.best_params_}",
            flush=True,
        )

        fold_results.append({
            "fold": fold_i,
            "r2": r2, "mse": mse, "mae": mae,
            "best_params": search.best_params_,
            "n_after_corrfilter": n_after_corr,
            "n_after_selectfrommodel": n_after_sel,
            "y_test": y_test,
            "y_pred": y_pred,
            "feature_importances": fold_importance,
        })

    # CV summary
    r2s = [f["r2"] for f in fold_results]
    print(f"\nCV Mean R2: {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}",
          flush=True)

    # Final model on all CV data
    print("\nTraining final model on full CV set...", flush=True)
    from collections import Counter
    param_counts = Counter(
        tuple(sorted(f["best_params"].items()))
        for f in fold_results
    )
    best_params = dict(param_counts.most_common(1)[0][0])
    print(f"Most common best params: {best_params}", flush=True)

    final_pipeline = build_pipeline()
    final_pipeline.set_params(**best_params)
    final_pipeline.fit(X_cv, y_cv)

    y_holdout_pred = final_pipeline.predict(X_holdout)
    holdout_r2 = r2_score(y_holdout, y_holdout_pred)
    holdout_mse = mean_squared_error(y_holdout, y_holdout_pred)
    holdout_mae = mean_absolute_error(y_holdout, y_holdout_pred)

    print(f"\n{'='*60}", flush=True)
    print(f"HOLDOUT VALIDATION ({len(holdout_idx)} compounds)", flush=True)
    print(f"  R2:  {holdout_r2:.4f}", flush=True)
    print(f"  MSE: {holdout_mse:.4f}", flush=True)
    print(f"  MAE: {holdout_mae:.4f}", flush=True)
    print(f"  CV Mean R2 was: {np.mean(r2s):.4f} +/- {np.std(r2s):.4f}",
          flush=True)
    print(f"  Difference (holdout - CV mean): "
          f"{holdout_r2 - np.mean(r2s):+.4f}", flush=True)
    print(f"{'='*60}", flush=True)

    results = {
        "fold_results": fold_results,
        "cv_mean_r2": np.mean(r2s),
        "cv_std_r2": np.std(r2s),
        "holdout_r2": holdout_r2,
        "holdout_mse": holdout_mse,
        "holdout_mae": holdout_mae,
        "holdout_y_true": y_holdout,
        "holdout_y_pred": y_holdout_pred,
        "final_params": best_params,
        "final_pipeline": final_pipeline,
        "feature_names": feature_names,
    }

    return results
