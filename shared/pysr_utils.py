"""
Shared PySR utilities for HDP property prediction.

Contains all reusable PySR components:
- Feature sanitisation
- Feature set builders (normal, corrfilter, focused)
- PySR model running
- Pareto front printing and saving

Usage:
    from shared.pysr_utils import run_pysr, print_pareto_front
    from shared.pysr_utils import get_normal_features, apply_corrfilter
"""

import re
import pickle
import numpy as np
import pandas as pd
from pysr import PySRRegressor


# ══════════════════════════════════════════════
# Feature name sanitisation
# ══════════════════════════════════════════════

def sanitize_columns(df):
    """Replace all non-alphanumeric/underscore characters in column
    names with underscores. Required for Julia symbol compatibility."""
    df = df.copy()
    df.columns = [re.sub(r'[^a-zA-Z0-9_]', '_', c) for c in df.columns]
    return df


# ══════════════════════════════════════════════
# Feature set builders
# ══════════════════════════════════════════════

def get_normal_features(X, results_pkl):
    """Return the union of features surviving SelectFromModel across
    all GB folds (normal / GB-informed strategy).

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    results_pkl : str
        Path to GB result pkl file.

    Returns
    -------
    pd.DataFrame
        Sanitised feature matrix with GB-selected features only.
    """
    with open(results_pkl, "rb") as f:
        res = pickle.load(f)

    all_features = set()
    for fr in res["fold_results"]:
        all_features.update(fr["feature_importances"].keys())

    available = [f for f in sorted(all_features) if f in X.columns]
    print(f"Normal: {len(available)} features from GB SelectFromModel",
          flush=True)

    return sanitize_columns(X[available].copy())


def apply_corrfilter(X, y, threshold=0.9):
    """Apply CorrelationFilter to X and return filtered DataFrame.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    y : np.ndarray
        Target values (used to break ties in correlated pairs).
    threshold : float
        Correlation threshold.

    Returns
    -------
    pd.DataFrame
        Sanitised, correlation-filtered feature matrix.
    """
    Xv = X.values
    corr = np.abs(np.corrcoef(Xv, rowvar=False))
    n = Xv.shape[1]
    drop = set()

    target_corr = np.abs(np.array(
        [np.corrcoef(Xv[:, i], y)[0, 1] for i in range(n)]
    ))

    for i in range(n):
        if i in drop:
            continue
        for j in range(i + 1, n):
            if j in drop:
                continue
            if corr[i, j] >= threshold:
                if target_corr[i] >= target_corr[j]:
                    drop.add(j)
                else:
                    drop.add(i)
                    break

    keep = [i for i in range(n) if i not in drop]
    X_filtered = X.iloc[:, keep].copy()
    print(f"CorrelationFilter: {len(keep)} of {n} features kept",
          flush=True)

    return sanitize_columns(X_filtered)


def get_recurring_features(results_pkl, min_folds=3):
    """Extract features recurring across GB folds for use in focused
    PySR strategy.

    Parameters
    ----------
    results_pkl : str
        Path to GB result pkl.
    min_folds : int
        Minimum number of folds a feature must appear in.

    Returns
    -------
    core : list
        Features appearing in >= min_folds folds.
    secondary : list
        Features appearing in min_folds - 1 folds.
    """
    with open(results_pkl, "rb") as f:
        res = pickle.load(f)

    n_folds = len(res["fold_results"])
    feature_counts = {}
    for fr in res["fold_results"]:
        for name in fr["feature_importances"]:
            feature_counts[name] = feature_counts.get(name, 0) + 1

    core = [f for f, c in feature_counts.items() if c >= min_folds]
    secondary = [f for f, c in feature_counts.items()
                 if c == min_folds - 1]

    print(f"Core features ({min_folds}+/{n_folds} folds): {len(core)}",
          flush=True)
    print(f"Secondary features ({min_folds-1}/{n_folds} folds): "
          f"{len(secondary)}", flush=True)

    return core, secondary


def build_focused_features(X, core, secondary, extra_ratios=None):
    """Build focused feature set with optional pre-computed ratios.

    Parameters
    ----------
    X : pd.DataFrame
        Full feature matrix.
    core : list
        Core recurring feature names.
    secondary : list
        Secondary recurring feature names.
    extra_ratios : dict or None
        Dict of {name: pd.Series} for pre-computed ratio features.
        Series must be indexed like X.

    Returns
    -------
    pd.DataFrame
        Sanitised focused feature matrix.
    """
    use = [f for f in core + secondary if f in X.columns]
    X_f = X[use].copy()
    print(f"Base focused features: {X_f.shape[1]}", flush=True)

    if extra_ratios:
        ratio_df = pd.DataFrame(extra_ratios, index=X_f.index)
        X_f = pd.concat([X_f, ratio_df], axis=1)
        print(f"Added {len(extra_ratios)} pre-computed ratios",
              flush=True)

    print(f"Total focused features: {X_f.shape[1]}", flush=True)
    return sanitize_columns(X_f)


# ══════════════════════════════════════════════
# PySR model
# ══════════════════════════════════════════════

def run_pysr(X, y, maxsize=40, niterations=500, tag="run",
             tempdir=None):
    """Run PySR symbolic regression.

    Parameters
    ----------
    X : pd.DataFrame
        Sanitised feature matrix (column names must be valid Julia symbols).
    y : np.ndarray
        Target values.
    maxsize : int
        Maximum expression tree size.
    niterations : int
        Number of evolutionary iterations.
    tag : str
        Label for the tempdir folder name.
    tempdir : str or None
        If None, uses ./pysr_{tag}.

    Returns
    -------
    PySRRegressor
        Fitted model with .equations_ Pareto front.
    """
    if tempdir is None:
        tempdir = f"./pysr_{tag}"

    model = PySRRegressor(
        binary_operators=["+", "-", "*", "/"],
        unary_operators=[
            "sqrt", "log", "exp", "square", "cube",
            "inv(x) = 1/x",
        ],
        extra_sympy_mappings={"inv": lambda x: 1 / x},
        maxsize=maxsize,
        niterations=niterations,
        populations=40,
        population_size=60,
        ncycles_per_iteration=400,
        parsimony=0.001,
        elementwise_loss="loss(prediction, target) = "
                         "(prediction - target)^2",
        temp_equation_file=True,
        tempdir=tempdir,
        progress=True,
    )
    model.fit(X, y)
    return model


# ══════════════════════════════════════════════
# Results output
# ══════════════════════════════════════════════

def print_pareto_front(model, y, top_n=30):
    """Print the Pareto front of equations sorted by complexity.

    Parameters
    ----------
    model : PySRRegressor
        Fitted model.
    y : np.ndarray
        Target values (used to compute R²).
    top_n : int
        Maximum number of equations to print.
    """
    eqs = model.equations_
    if eqs is None or len(eqs) == 0:
        print("No equations found.")
        return

    y_var = np.var(y)
    print(f"\n{'='*80}")
    print("PARETO FRONT: simplest to most complex")
    print(f"{'='*80}")

    eqs_sorted = eqs.sort_values("complexity").head(top_n)
    for _, row in eqs_sorted.iterrows():
        r2 = 1 - row["loss"] / y_var
        print(f"\nComplexity {int(row['complexity']):>2d} | "
              f"MSE={row['loss']:.6f} | R2~{r2:.4f}")
        print(f"  {row['equation']}")

    best = model.get_best()
    print(f"\n{'='*80}")
    print("BEST (auto-selected):")
    print(f"  {best['equation']}")
    print(f"  Complexity: {int(best['complexity'])}, "
          f"MSE: {best['loss']:.6f}, "
          f"R2~{1 - best['loss'] / y_var:.4f}")
    print(f"{'='*80}")


def save_results(model, y, out_prefix):
    """Save PySR model and equation CSV.

    Parameters
    ----------
    model : PySRRegressor
        Fitted model.
    y : np.ndarray
        Target values.
    out_prefix : str
        Prefix for output files (e.g. 'equations/pysr_eform_normal').
    """
    with open(f"{out_prefix}.pkl", "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {out_prefix}.pkl", flush=True)

    if model.equations_ is not None:
        model.equations_.to_csv(
            f"{out_prefix}_equations.csv", index=False
        )
        print(f"Saved equations to {out_prefix}_equations.csv",
              flush=True)
