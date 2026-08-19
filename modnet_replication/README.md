# modnet_replication/

Standalone replication of De Breuck et al. (2021) MODNet framework.

See [docs/04_modnet_replication.pdf](../docs/04_modnet_replication.pdf) for full details.

Requires `start_modnet` environment (TensorFlow 2.14, MODNet 0.4.5).

```bash
micromamba activate start_modnet
python featurize.py          # one-time, ~4-6 hours
python feature_selection.py --target bandgap
python feature_selection.py --target Ehull_avg
python train_bandgap.py
python train_ehull.py fast   # sanity check
python train_ehull.py        # full run (overnight)
```
