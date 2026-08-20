"""
Matminer2023 featurization for HDP compounds.

Reads crystal structures from the HDP dataset CSV and computes
2325 Matminer2023 features. Saves a MODData checkpoint.

Runtime: ~4-6 hours on CPU.
Environment: start_modnet

Usage:
    python featurize.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from modnet.preprocessing import MODData
from modnet.featurizers.presets import Matminer2023Featurizer
from pymatgen.core import Structure
import ast

DATA_CSV = "../data/HDP_CombinedInfo_WithStructures.csv"
TARGET_COL = "BandGap"
OUTPUT_PKL = "../data/modnet_featurized.pkl"


def load_structures(csv_path):
    """Load structures and targets from the HDP dataset CSV."""
    print("Loading dataset...", flush=True)
    df = pd.read_csv(csv_path, index_col="comp_name_full")

    # Parse structure column (stored as dict string)
    print("Parsing structures...", flush=True)
    structures = {}
    for name, row in df.iterrows():
        try:
            struct_dict = ast.literal_eval(row["structure"])
            structures[name] = Structure.from_dict(struct_dict)
        except Exception as e:
            print(f"  Warning: could not parse structure for {name}: {e}")

    targets = df[[TARGET_COL]].dropna()
    print(f"Loaded {len(structures)} structures, "
          f"{len(targets)} with {TARGET_COL}", flush=True)
    return structures, targets


if __name__ == "__main__":
    structures, targets = load_structures(DATA_CSV)

    # Build MODData object
    data = MODData(
        materials=list(structures.values()),
        targets=targets.values.tolist(),
        target_names=[TARGET_COL],
        structure_ids=list(structures.keys()),
    )

    # Featurize
    print("\nStarting featurization...", flush=True)
    data.featurize(
        featurizer=Matminer2023Featurizer(),
        n_jobs=4,
    )
    print(f"Featurized: {data.df_featurized.shape}", flush=True)

    # Save checkpoint
    data.save(OUTPUT_PKL)
    print(f"\nSaved to {OUTPUT_PKL}", flush=True)
