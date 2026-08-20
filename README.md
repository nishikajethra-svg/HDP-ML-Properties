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
├── README.md
├── LICENSE
├── .gitignore
├── requirements_bonding.txt       # add_bonding environment
├── requirements_modnet.txt        # start_modnet environment
│
├── data/                          # Feature CSVs (see data/README.md)
│   ├── README.md
│   ├── hdp_lobfeats_antibonding.csv
│   ├── hdp_lobfeats_bonding_NoDOS.csv
│   └── MLIP_Ehullform_averaged.csv
│
├── shared/                        # Shared pipeline code
│   ├── pipeline.py                # CorrelationFilter, data loading, nested CV
│   ├── pysr_utils.py              # PySR feature strategies and utilities
│   └── plotting.py                # Parity, feature importance, Pareto plots
│
├── featurization/                 # Matminer featurization + NMI selection
│   ├── featurize.py               # 2325 Matminer2023 features (~4-6 hours)
│   └── feature_selection.py       # NMI selection → 300 features per target
│
├── models/                        # GB and RF training + analysis
│   ├── rf_bandgap.py
│   ├── gb_bandgap.py
│   ├── gb_bandgap_filtered.py     # bandgap > 0.1 eV only
│   ├── gb_eform.py
│   ├── gb_ehull.py
│   ├── gb_ehull_nodos.py          # Ehull without DOS band descriptors
│   ├── analyze_bandgap.py
│   ├── analyze_eform.py
│   ├── analyze_ehull.py
│   └── results/                   # result pkl files (not tracked)
│
├── symbolic_regression/           # PySR scripts and equation CSVs
│   ├── pysr_bandgap.py            # normal / corrfilter / focused
│   ├── pysr_eform.py
│   ├── pysr_ehull.py
│   ├── plot_pareto.py
│   └── equations/                 # Pareto front CSVs (committed)
│       ├── pysr_eform_normal_equations.csv
│       ├── pysr_eform_corrfilter_equations.csv
│       ├── pysr_eform_focused_equations.csv
│       ├── pysr_ehull_normal_equations.csv
│       ├── pysr_ehull_corrfilter_equations.csv
│       └── pysr_ehull_focused_equations.csv
│
├── modnet_replication/            # Standalone MODNet paper replication
│   ├── featurize.py
│   ├── feature_selection.py
│   ├── train_bandgap.py
│   └── train_ehull.py             # fast mode: python train_ehull.py fast
│
├── notebooks/
│   └── results_overview.ipynb     # Visual results browser (no re-running needed)
│
├── figures/                       # Output plots
├── presentations/                 # PowerPoint updates
└── docs/                          # SOP documentation PDFs
    ├── 00_project_overview.pdf
    ├── 01_featurization.pdf
    ├── 02_models.pdf
    ├── 03_symbolic_regression.pdf
    ├── 04_modnet_replication.pdf
    └── 05_results_summary.pdf
```

---

## Dataset

The primary HDP dataset must be downloaded separately from NOMAD before running any scripts.

**Preprint:** L. Walterbos, A. McEwan, R. Shinde, J. George, L. Leppert,
"Spin-Polarized Electronic Structure and Chemical Bonding Data for 2,500+ Halide Double Perovskites,"
*under review* (2026). [arXiv:2606.11928](https://arxiv.org/abs/2606.11928)

**NOMAD dataset:** https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/nomad.wb9y-b8j7

**Workflow analysis code:** https://github.com/Luccerboi/HDP_WorkFLow_Analysis

Download `HDP_CombinedInfo_WithStructures.csv` from the NOMAD link above and place it in `data/`.
All derived feature files (`hdp_lobfeats_antibonding.csv`, `hdp_lobfeats_bonding_NoDOS.csv`,
`MLIP_Ehullform_averaged.csv`) are already included in `data/`.

Pre-computed MODNet featurization checkpoints (`modnet_featurized.pkl`, `modnet_selected.pkl`)
are large files not tracked by git. Download from: [Zenodo DOI — to be added]

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

**Dataset & bonding descriptors:**
- L. Walterbos, A. McEwan, R. Shinde, J. George, L. Leppert, "Spin-Polarized Electronic Structure and Chemical Bonding Data for 2,500+ Halide Double Perovskites," *under review* (2026). [arXiv:2606.11928](https://arxiv.org/abs/2606.11928)
- NOMAD dataset: https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/nomad.wb9y-b8j7
- Workflow code: https://github.com/Luccerboi/HDP_WorkFLow_Analysis

**MODNet:**
- P.-P. De Breuck, G. Hautier, G.-M. Rignanese, "Materials property prediction for limited datasets enabled by feature selection and joint learning with MODNet," *npj Computational Materials* **7**, 83 (2021). https://doi.org/10.1038/s41524-021-00552-2

**Symbolic regression:**
- M. Cranmer, "Interpretable machine learning for science with PySR and SymbolicRegression.jl," *arXiv*:2305.01582 (2023). https://arxiv.org/abs/2305.01582

**Featurization:**
- L. Ward et al., "Matminer: An open source toolkit for materials data mining," *Computational Materials Science* **152**, 60–69 (2018). https://doi.org/10.1016/j.commatsci.2018.05.018

**Bonding analysis:**
- J. George et al., "Automated bonding analysis with crystal orbital Hamilton populations," *ChemPlusChem* **87**, e202200123 (2022). https://doi.org/10.1002/cplu.202200123 (LobsterPy)
- S. Maintz et al., "LOBSTER: A tool to extract chemical bonding from plane-wave based DFT," *Journal of Computational Chemistry* **37**, 1030–1035 (2016). https://doi.org/10.1002/jcc.24300

**Materials informatics:**
- S. P. Ong et al., "Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis," *Computational Materials Science* **68**, 314–319 (2013). https://doi.org/10.1016/j.commatsci.2012.10.028
- F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research* **12**, 2825–2830 (2011).