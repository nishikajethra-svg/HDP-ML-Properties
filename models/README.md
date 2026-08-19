# models/

GB and RF training scripts. All import from `../shared/pipeline.py`.

See [docs/02_models.pdf](../docs/02_models.pdf) for full details.

## Run order
```bash
micromamba activate add_bonding
python rf_bandgap.py
python gb_bandgap.py
python gb_bandgap_filtered.py
python gb_eform.py
python gb_ehull.py
python gb_ehull_nodos.py
```
Results saved to `results/`.
