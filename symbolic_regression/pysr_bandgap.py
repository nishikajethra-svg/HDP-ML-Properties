"""
PySR symbolic regression for bandgap prediction.
 
Covers all four strategies run during the internship:
  1. normal      - GB-informed features (union across 5 folds)
  2. corrfilter  - all features after CorrelationFilter (discovered charge_A)
  3. focused     - recurring features + pre-computed ratios (best result R2~0.65)
 
The corrfilter run is particularly important: it discovered charge_A
(charge on the A-site cation) which never appeared in GB importance rankings
but showed up consistently from complexity 14 onwards in every PySR run.
This illustrates how symbolic regression and tree-based importance find
complementary structure.
 
Usage:
    python pysr_bandgap.py normal
    python pysr_bandgap.py corrfilter
    python pysr_bandgap.py focused
 
Environment: add_bonding
 
Note: Run GB first (gb_bandgap.py) before running focused mode,
as it reads the GB result pkl to extract recurring features.
"""
 
import sys
import os
import re
import numpy as np
sys.path.insert(0, "..")
 
from shared.pipeline import load_data, CorrelationFilter
from shared.pysr_utils import (
    get_normal_features,
    apply_corrfilter,
    get_recurring_features,
    build_focused_features,
    run_pysr,
    print_pareto_front,
    save_results,
)
 
TARGET = "bandgap"
PATHS = dict(
    pkl_path="../data/modnet_selected.pkl",
    lobdata_path="../data/hdp_lobfeats_antibonding.csv",
    original_csv_path="../data/HDP_CombinedInfo_WithStructures.csv",
    target_csv_path="../data/MLIP_Ehullform_averaged.csv",
)
GB_PKL = "results/results_gb_bandgap.pkl"
 
 
def build_bandgap_focused_ratios(X, core, secondary):
    """Pre-compute ratios PySR kept discovering across normal + corrfilter runs.
 
    Key ratios identified from Pareto front analysis:
    - sigma_col x HOMO_energy: column variation times HOMO energy
    - HOMO / Icobi_B1_asym: orbital energy normalised by B1 asymmetry
    - Delta_chi - B12_asym: EN difference minus B1+B2 asymmetry sum
    - exp(Delta_chi - B12_asym): exponential of the above
    - Delta_chi^3 / density: EN cubed over density
    - charge_A - charge_spin: A-site charge minus spin charge
    - q_B2 + gap_AO: B2 charge plus atomic orbital gap
 
    Note: charge_A was discovered by the corrfilter run at complexity 14.
    It never appeared in GB importance rankings — a clear example of
    symbolic regression finding structure that tree importance misses.
    """
    use = [f for f in core + secondary if f in X.columns]
    Xf = X[use].copy()
 
    ratios = {}
 
    def _col(name):
        """Find column containing name (case-insensitive partial match)."""
        matches = [c for c in Xf.columns if name.lower() in c.lower()]
        return matches[0] if matches else None
 
    def _add(ratio_name, col_a, col_b, op="mul"):
        a, b = _col(col_a), _col(col_b)
        if a is None or b is None:
            return
        if op == "mul":
            ratios[ratio_name] = Xf[a] * Xf[b]
        elif op == "div":
            ratios[ratio_name] = Xf[a] / (Xf[b].abs() + 1e-6)
        elif op == "sub":
            ratios[ratio_name] = Xf[a] - Xf[b]
        elif op == "add":
            ratios[ratio_name] = Xf[a] + Xf[b]
 
    # sigma_col x HOMO_energy (appears at C4 in normal run)
    _add("sigma_col_times_HOMO", "avg_dev Column", "HOMO_energy", "mul")
 
    # HOMO / Icobi_B1_directional_asym (C5 in normal run)
    _add("HOMO_div_B1_asym", "HOMO_energy", "Icobi.B1.directional", "div")
 
    # B1 + B2 directional asymmetry sum (recurring from C5)
    b1 = _col("Icobi.B1.directional")
    b2 = _col("Icobi.B2.directional")
    if b1 and b2:
        ratios["B12_asym_sum"] = Xf[b1] + Xf[b2]
 
    # Delta_chi - B12_asym (C10 exponential structure)
    dchi = _col("minimum EN difference")
    if dchi and "B12_asym_sum" in ratios:
        ratios["dchi_minus_B12"] = Xf[dchi] - ratios["B12_asym_sum"]
        ratios["exp_dchi_minus_B12"] = np.exp(
            ratios["dchi_minus_B12"].clip(upper=5)
        )
 
    # Delta_chi^3 / density (C40 best formula)
    dens = _col("density")
    if dchi and dens:
        ratios["dchi_cubed_div_density"] = Xf[dchi] ** 3 / (
            Xf[dens].abs() + 1e-6
        )
 
    # charge_A - charge_spin (corrfilter discovery, C40)
    _add("charge_A_minus_spin", "charge_A", "charge_spin", "sub")
 
    # q_B2 + gap_AO (C40 best formula)
    _add("qB2_plus_gapAO", "charge_B2", "gap_AO", "add")
 
    # HOMO x gap_AO (recurring from C3)
    _add("HOMO_times_gapAO", "HOMO_energy", "gap_AO", "mul")
 
    # Madelung / EN_diff (corrfilter mid-complexity)
    _add("Madelung_div_EN", "MadelungEnergy", "minimum EN difference", "div")
 
    # avg_dev_Column + avg_dev_EN (simple additive structure)
    _add("col_plus_EN", "avg_dev Column", "avg_dev Electronegativity", "add")
 
    print(f"Pre-computed {len(ratios)} ratios for focused run", flush=True)
    return build_focused_features(X, core, secondary, ratios)
 
 
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "normal"
    assert mode in ("normal", "corrfilter", "focused"), \
        "Usage: python pysr_bandgap.py [normal|corrfilter|focused]"
 
    print(f"\n{'#'*60}\n  PySR Bandgap -- mode: {mode}\n{'#'*60}\n",
          flush=True)
 
    X, y = load_data(target=TARGET, **PATHS)
    print(f"Full data: {X.shape}", flush=True)
 
    if mode == "normal":
        X_pysr = get_normal_features(X, GB_PKL)
 
    elif mode == "corrfilter":
        # The corrfilter run is the one that discovered charge_A.
        # It searches ~200 features after correlation removal -- a much
        # wider space than the GB-informed set, which is why it can find
        # features that GB importance never ranked highly.
        X_pysr = apply_corrfilter(X, y)
        print(
            "Note: this is the run that discovered charge_A at C14+.\n"
            "charge_A never appeared in GB top-18 importance rankings\n"
            "but shows up consistently in corrfilter PySR formulas.",
            flush=True
        )
 
    elif mode == "focused":
        # Two-stage: recurring features from normal + corrfilter runs,
        # plus pre-computed ratios PySR kept building internally.
        # This is the strategy that achieved the best result (R2~0.65).
        core, secondary = get_recurring_features(GB_PKL, min_folds=3)
        X_pysr = build_bandgap_focused_ratios(X, core, secondary)
 
    print(f"PySR input: {X_pysr.shape[1]} features", flush=True)
 
    model = run_pysr(X_pysr, y, maxsize=40, niterations=500,
                     tag=f"bandgap_{mode}")
    print_pareto_front(model, y)
 
    os.makedirs("equations", exist_ok=True)
    save_results(model, y, f"equations/pysr_bandgap_{mode}")
 