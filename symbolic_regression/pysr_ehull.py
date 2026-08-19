"""
PySR symbolic regression for Ehull prediction.

Usage:
    python pysr_ehull.py normal
    python pysr_ehull.py corrfilter
    python pysr_ehull.py focused
"""
import sys, os, numpy as np
sys.path.insert(0, "..")

from shared.pipeline import load_data, CorrelationFilter
from shared.pysr_utils import (
    get_normal_features, apply_corrfilter,
    get_recurring_features, build_focused_features,
    run_pysr, print_pareto_front, save_results,
)

TARGET = "Ehull_avg"
PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_antibonding.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)
GB_PKL = "results/results_gb_ehull.pkl"


def build_ehull_focused_ratios(X, core, secondary):
    """Pre-compute ratios PySR kept discovering for Ehull."""
    use = [f for f in core + secondary if f in X.columns]
    Xf = X[use].copy()

    ratios = {}

    def _add(name, a, b, op="div"):
        if a not in Xf.columns or b not in Xf.columns:
            return
        if op == "div":
            ratios[name] = Xf[a] / (Xf[b].abs() + 0.01)
        elif op == "mul":
            ratios[name] = Xf[a] * Xf[b]
        elif op == "sub":
            ratios[name] = Xf[a] - Xf[b]
        elif op == "add":
            ratios[name] = Xf[a] + Xf[b]

    # antibond_max x p_center^2 (C10 focused)
    if "antibonding_orb_perc_max" in Xf.columns and \
       "p_band_center" in Xf.columns:
        ratios["antibond_max_times_p_center_sq"] = (
            Xf["antibonding_orb_perc_max"] * Xf["p_band_center"] ** 2
        )

    # ICOHP_min x log(s_width) (normal C14)
    if "ICOHP_mean_min" in Xf.columns and "s_band_width" in Xf.columns:
        ratios["ICOHP_min_times_log_swidth"] = (
            Xf["ICOHP_mean_min"]
            * np.log(Xf["s_band_width"].clip(lower=0.001))
        )

    # GaussSymm / min_EN_diff (normal C19)
    _add("GaussSymm_div_minEN",
         "GaussianSymmFunc|std_dev G2_0.05",
         "ElectronegativityDiff|minimum EN difference", op="div")

    # p_center / Ionicity_Loew (corrfilter C25)
    _add("p_center_div_Ionicity",
         "p_band_center", "Ionicity_Loew", op="div")

    # ICOHP_antibndg / (min_Number - 3.6) (corrfilter C25)
    if "ICOHP_antibndg_orb_sum_max" in Xf.columns and \
       "ElementProperty|MagpieData minimum Number" in Xf.columns:
        ratios["ICOHP_antibndg_div_minNumber"] = (
            Xf["ICOHP_antibndg_orb_sum_max"]
            / (Xf["ElementProperty|MagpieData minimum Number"]
               - 3.6).abs().clip(lower=0.01)
        )

    # p_center - 1/(min_Number - 4.5)^2 (corrfilter C12)
    if "p_band_center" in Xf.columns and \
       "ElementProperty|MagpieData minimum Number" in Xf.columns:
        ratios["p_center_minus_inv_minNum_sq"] = (
            Xf["p_band_center"]
            - 1.0
            / (Xf["ElementProperty|MagpieData minimum Number"]
               - 4.5).clip(lower=0.01) ** 2
        )

    # rangeEN / minEN (corrfilter C26)
    _add("rangeEN_div_minEN",
         "ElectronegativityDiff|range EN difference",
         "ElectronegativityDiff|minimum EN difference", op="div")

    # exp(d_band_upperband_edge) (corrfilter C26)
    if "d_band_upperband_edge" in Xf.columns:
        ratios["exp_d_upperband"] = np.exp(
            Xf["d_band_upperband_edge"].clip(upper=5)
        )

    return build_focused_features(X, core, secondary, ratios)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    assert mode in ("normal", "corrfilter", "focused"), \
        "Usage: python pysr_ehull.py [normal|corrfilter|focused]"

    print(f"\n{'#'*60}\n  PySR Ehull — mode: {mode}\n{'#'*60}\n",
          flush=True)

    X, y = load_data(target=TARGET, **PATHS)
    print(f"Full data: {X.shape}", flush=True)

    if mode == "normal":
        X_pysr = get_normal_features(X, GB_PKL)
    elif mode == "corrfilter":
        X_pysr = apply_corrfilter(X, y)
    elif mode == "focused":
        core, secondary = get_recurring_features(GB_PKL, min_folds=3)
        X_pysr = build_ehull_focused_ratios(X, core, secondary)

    print(f"PySR input: {X_pysr.shape[1]} features", flush=True)

    model = run_pysr(X_pysr, y, maxsize=40, niterations=500,
                     tag=f"ehull_{mode}")
    print_pareto_front(model, y)

    os.makedirs("equations", exist_ok=True)
    save_results(model, y, f"equations/pysr_ehull_{mode}")
