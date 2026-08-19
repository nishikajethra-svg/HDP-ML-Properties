"""Plot PySR Pareto front curves for all targets and modes."""
import sys
sys.path.insert(0, "..")
import numpy as np
from shared.plotting import plot_pareto

# Y variances (from the training data — MSE at complexity 1)
Y_VAR = {
    "eform": 0.454758,
    "ehull": 0.009542,
}

MODES = ["normal", "corrfilter", "focused"]
LABELS = ["Normal", "CorrelationFilter", "Focused"]

if __name__ == "__main__":
    for target, y_var in Y_VAR.items():
        csvs = [
            f"equations/pysr_{target}_{mode}_equations.csv"
            for mode in MODES
        ]
        plot_pareto(
            equation_csvs=csvs,
            labels=LABELS,
            y_var=y_var,
            save_path=f"../figures/pareto_{target}.png",
            title=f"{target.capitalize()}: PySR Pareto Fronts",
        )
    print("Done.")
