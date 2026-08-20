"""
NMI-based feature selection for a given target property.

Loads the featurized MODData checkpoint, swaps the target if needed,
runs MODNet's relevance-redundancy feature selection, and saves
a new checkpoint with 300 selected features.

Runtime: ~40-60 minutes per target (pairwise NMI computation).
Environment: start_modnet

Usage:
    python feature_selection.py --target bandgap
    python feature_selection.py --target Ehull_avg
    python feature_selection.py --target Eform_avg
"""
import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from modnet.preprocessing import MODData

FEATURIZED_PKL = "../data/modnet_featurized.pkl"
TARGET_CSV = "../data/MLIP_Ehullform_averaged.csv"
N_FEATURES = 300


def main(target):
    print(f"Loading featurized MODData...", flush=True)
    data = MODData.load(FEATURIZED_PKL)
    print(f"Loaded: {data.df_featurized.shape}", flush=True)

    # Swap target if not bandgap
    if target != "bandgap":
        print(f"Loading target: {target}", flush=True)
        target_df = pd.read_csv(TARGET_CSV, index_col=0)
        target_df.index = data.df_featurized.index  # positional align

        valid_mask = ~target_df[target].isna()
        n_dropped = (~valid_mask).sum()
        print(f"Dropping {n_dropped} NaN target compounds", flush=True)

        keep_idx = [i for i, v in enumerate(valid_mask) if v]
        drop_idx = [i for i, v in enumerate(valid_mask) if not v]
        data, _ = data.split((keep_idx, drop_idx))

        data.df_targets = pd.DataFrame(
            {target: target_df.loc[valid_mask, target].values},
            index=data.df_featurized.index,
        )
        data.num_classes = {target: 0}  # 0 = regression

    print(f"\nRunning NMI feature selection (n={N_FEATURES})...",
          flush=True)
    print("Step 1: Pairwise NMI between features (~40 min)...",
          flush=True)

    data.feature_selection(
        n=N_FEATURES,
        use_precomputed_cross_nmi=False,
    )

    selected = data.get_optimal_descriptors()
    print(f"\nSelected {len(selected)} features", flush=True)
    print(f"Top 10: {selected[:10]}", flush=True)

    out_path = f"../data/modnet_selected_{target}.pkl"
    data.save(out_path)
    print(f"\nSaved to {out_path}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target", default="bandgap",
        help="Target property: bandgap, Ehull_avg, or Eform_avg"
    )
    args = parser.parse_args()
    main(args.target)
