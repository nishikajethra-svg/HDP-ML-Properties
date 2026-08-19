# ML Prediction of Electronic Properties of Halide Double Perovskites

Predicting band gap, formation energy ($E_\text{form}$), and energy above hull ($E_\text{hull}$) of halide double perovskite (HDP) compounds using gradient boosted trees, MODNet neural networks, and PySR symbolic regression.

**BAM Internship, Berlin, 2026** | Nishika Jethra

---

## Results

| Target | Model | CV R² | Holdout R² |
|--------|-------|-------|------------|
| Band gap | GB | 0.896 ± 0.014 | **0.911** |
| Band gap | MODNet NN | 0.910 | — |
| $E_\text{form}$ | GB | 0.983 ± 0.004 | **0.985** |
| $E_\text{form}$ | PySR (best) | — | 0.959 |
| $E_\text{hull}$ | GB | 0.823 ± 0.017 | **0.733** |
| $E_\text{hull}$ | MODNet NN | 0.748 ± 0.036 | 0.698 |
| $E_\text{hull}$ | PySR (best) | — | 0.536 |

**Key finding:** Adding LobsterPy bonding descriptors reverses the GB vs. MODNet ranking for band gap (MODNet leads without bonding; GB leads with bonding). For $E_\text{hull}$, GB outperforms MODNet — likely because GB benefits from `center_COBI` bonding descriptors that MODNet does not use.

**Key finding:** $E_\text{form}$ and $E_\text{hull}$ require fundamentally different descriptor sets. A single electronegativity feature captures 81% of $E_\text{form}$ variance. $E_\text{hull}$ depends on band structure features (p-band center, Yang ω).

---

## Repository Structure

```
HDP-ML-Properties/
├── docs/                        # SOP documentation (PDF)
│   ├── 00_project_overview.pdf
│   ├── 01_featurization.pdf
│   ├── 02_models.pdf
│   ├── 03_symbolic_regression.pdf
│   ├── 04_modnet_replication.pdf
│   └── 05_results_summary.pdf
├── data/                        # Feature CSVs (see data/README.md)
├── shared/                      # Shared pipeline code
│   ├── pipeline.py              # CorrelationFilter, data loading, nested CV
│   ├── pysr_utils.py            # PySR feature strategies and utilities
│   └── plotting.py              # Parity, feature importance, Pareto plots
├── models/                      # GB and RF training scripts
├── symbolic_regression/         # PySR scripts and equation CSVs
├── modnet_replication/          # Standalone MODNet paper replication
├── figures/                     # Output plots
└── notebooks/
    └── results_overview.ipynb   # Visual results browser
```

---

## Dataset

The primary HDP dataset must be downloaded separately:

> Naik et al. (2024) — *[citation]* — [source repository URL]

Place `HDP_CombinedInfo_WithStructures.csv` in `data/`. All derived feature files (`hdp_lobfeats_antibonding.csv`, `hdp_lobfeats_bonding_NoDOS.csv`, `MLIP_Ehullform_averaged.csv`) are included in `data/`.

Pre-computed MODNet featurization checkpoints (`modnet_featurized.pkl`, `modnet_selected.pkl`) are available at: [Zenodo DOI link]

---

## Quickstart

```bash
# Clone
git clone https://github.com/[username]/HDP-ML-Properties
cd HDP-ML-Properties

# Set up environment (GB, PySR, analysis)
micromamba create -n add_bonding python=3.10 numpy"<2" \
    pandas scikit-learn matplotlib pysr -c conda-forge
micromamba activate add_bonding

# Run GB for formation energy
cd models
python gb_eform.py

# Run PySR (after GB)
cd ../symbolic_regression
python pysr_eform.py normal
python pysr_eform.py corrfilter
python pysr_eform.py focused

# Analyse results
python analyze_eform.py
```

For MODNet, use the `start_modnet` environment (see `requirements_modnet.txt`).

---

## Feature Engineering

Three feature sources combined (393 total):

| Source | Features | Description |
|--------|----------|-------------|
| MODNet / Matminer2023 | 300 (NMI-selected) | Composition, structure, site-based |
| LobsterPy bonding | 85 | ICOHP/ICOBI stats, orbital-resolved bonding |
| Asymmetry indices | 8 | ICOHP/ICOBI B1/B2 directional/axial asymmetry |

---

## Documentation

Full SOP documents in `docs/`:

- **[00_project_overview.pdf](docs/00_project_overview.pdf)** — Full methodology, all results, physical interpretation
- **[01_featurization.pdf](docs/01_featurization.pdf)** — Matminer2023, NMI selection, LobsterPy features
- **[02_models.pdf](docs/02_models.pdf)** — RF, GB, MODNet — architecture and three-way comparison
- **[03_symbolic_regression.pdf](docs/03_symbolic_regression.pdf)** — PySR strategies, all formulas, recurring features
- **[04_modnet_replication.pdf](docs/04_modnet_replication.pdf)** — MODNet paper replication details
- **[05_results_summary.pdf](docs/05_results_summary.pdf)** — All numbers, tables, and formulas

---

## References

- De Breuck et al., *npj Computational Materials* **7**, 83 (2021) — MODNet: https://doi.org/10.1038/s41524-021-00552-2
- Naik et al. (2024) — HDP dataset and LobsterPy descriptors
- Cranmer (2023) — PySR: https://arxiv.org/abs/2305.01582
