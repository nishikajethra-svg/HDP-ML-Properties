# symbolic_regression/

PySR scripts for all targets. Run GB models first (focused mode reads GB results).

See [docs/03_symbolic_regression.pdf](../docs/03_symbolic_regression.pdf) for full details.

## Run order
```bash
micromamba activate add_bonding
python pysr_eform.py normal
python pysr_eform.py corrfilter
python pysr_eform.py focused
python pysr_ehull.py normal
python pysr_ehull.py corrfilter
python pysr_ehull.py focused
python plot_pareto.py
```
Equation CSVs saved to `equations/`.
