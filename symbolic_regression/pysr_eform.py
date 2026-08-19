"""
PySR symbolic regression for Eform prediction.

Usage:
    python pysr_eform.py normal       # GB-informed features
    python pysr_eform.py corrfilter   # after CorrelationFilter
    python pysr_eform.py focused      # recurring features + ratios
"""
import sys, os
sys.path.insert(0, "..")

from shared.pipeline import load_data, CorrelationFilter
from shared.pysr_utils import (
    get_normal_features, apply_corrfilter,
    get_recurring_features, build_focused_features,
    run_pysr, print_pareto_front, save_results,
)
import numpy as np
import pandas as pd

TARGET = "Eform_avg"
PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_antibonding.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)
GB_PKL = "results/results_gb_eform.pkl"


def build_eform_focused_ratios(X, core, secondary):
    """Pre-compute ratios PySR kept discovering across normal + corrfilter."""
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

    # NsUnfilled - gap_AO (corrfilter C12 numerator)
    _add("NsUnfilled_minus_gapAO",
         "ElementProperty|MagpieData mean NsUnfilled",
         "AtomicOrbitals|gap_AO", op="sub")

    # antibond_min x CN6_oct (corrfilter C16)
    _add("antibond_min_times_CN6",
         "antibonding_orb_perc_min",
         "CrystalNNFingerprint|mean octahedral CN_6", op="mul")

    # gap_AO - (LUMO - NsUnfilled)^2 (corrfilter C22)
    if all(c in Xf.columns for c in [
        "AtomicOrbitals|gap_AO",
        "AtomicOrbitals|LUMO_energy",
        "ElementProperty|MagpieData mean NsUnfilled",
    ]):
        ratios["gapAO_minus_sq_LUMO_NsUnf"] = (
            Xf["AtomicOrbitals|gap_AO"]
            - (Xf["AtomicOrbitals|LUMO_energy"]
               - Xf["ElementProperty|MagpieData mean NsUnfilled"]) ** 2
        )

    # N_unf - Madelung (focused C8)
    _add("NUnfilled_minus_Madelung",
         "ElementProperty|MagpieData range NUnfilled",
         "Madelung_Mull", op="sub")

    # Madelung / EN_std (focused C15)
    _add("Madelung_div_EN_std",
         "Madelung_Mull",
         "ElectronegativityDiff|std_dev EN difference", op="div")

    # EN_min x ICOHP_antibndg_mean (focused C25)
    _add("minEN_times_ICOHP_ab_mean",
         "ElectronegativityDiff|minimum EN difference",
         "ICOHP_antibndg_orb_mean_avg", op="mul")

    # exp(Madelung/15) - ICOHP_sum (focused C25)
    if "Madelung_Mull" in Xf.columns and \
       "ICOHP_antibndg_orb_sum_max" in Xf.columns:
        ratios["exp_Madelung_minus_ICOHP_sum"] = (
            np.exp(Xf["Madelung_Mull"] / 15.0)
            - Xf["ICOHP_antibndg_orb_sum_max"]
        )

    # avg_dev_EN + Voronoi_std (corrfilter C21)
    _add("avgdevEN_plus_Voronoi",
         "ElementProperty|MagpieData avg_dev Electronegativity",
         "VoronoiFingerprint|std_dev Voro_dist_std_dev", op="add")

    # EN_std / Loewdin_max (normal C8)
    _add("EN_std_div_Loewdin",
         "ElectronegativityDiff|std_dev EN difference",
         "Loewdin_max", op="div")

    return build_focused_features(X, core, secondary, ratios)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    assert mode in ("normal", "corrfilter", "focused"), \
        "Usage: python pysr_eform.py [normal|corrfilter|focused]"

    print(f"\n{'#'*60}\n  PySR Eform — mode: {mode}\n{'#'*60}\n",
          flush=True)

    X, y = load_data(target=TARGET, **PATHS)
    print(f"Full data: {X.shape}", flush=True)

    if mode == "normal":
        X_pysr = get_normal_features(X, GB_PKL)
    elif mode == "corrfilter":
        X_pysr = apply_corrfilter(X, y)
    elif mode == "focused":
        core, secondary = get_recurring_features(GB_PKL, min_folds=3)
        X_pysr = build_eform_focused_ratios(X, core, secondary)

    print(f"PySR input: {X_pysr.shape[1]} features", flush=True)

    model = run_pysr(X_pysr, y, maxsize=40, niterations=500,
                     tag=f"eform_{mode}")
    print_pareto_front(model, y)

    os.makedirs("equations", exist_ok=True)
    save_results(model, y, f"equations/pysr_eform_{mode}")
