# data/

Feature CSV files included here. The primary HDP dataset must be downloaded separately.

See [docs/01_featurization.pdf](../docs/01_featurization.pdf) for full details.

## Dataset

**Dataset:** Walterbos, L. (2026). HalideDoublePerovskite_HSE06_Lobster.
https://nomad-lab.eu/prod/v1/gui/dataset/doi/10.17172/nomad.wb9y-b8j7

**Preprint:** L. Walterbos, A. McEwan, R. Shinde, J. George, L. Leppert,
"Spin-Polarized Electronic Structure and Chemical Bonding Data for 2,500+
Halide Double Perovskites," *under review* (2026). arXiv:2606.11928
https://arxiv.org/abs/2606.11928

**Workflow analysis code:** https://github.com/Luccerboi/HDP_WorkFLow_Analysis

Download `HDP_CombinedInfo_WithStructures.csv` from the NOMAD link above
and place it in this `data/` directory before running any scripts.

## Included files

- `hdp_lobfeats_antibonding.csv` — 85 LobsterPy bonding features
- `hdp_lobfeats_bonding_NoDOS.csv` — 65 features (no DOS band descriptors)
- `MLIP_Ehullform_averaged.csv` — MLIP-averaged Ehull and Eform values