# R022-aligned architecture experiment harness

This directory contains an executable training harness aligned to the documented R022 recipe and configs for a baseline plus proposed architecture comparisons. The B config is a fresh implementation of the written recipe; it is not a bitwise reproduction of the remote R022 checkpoint. C01–C12 and their controls are proposals; they have not been run as part of this package.

## Data and environment

The competition dataset and trained weights are not distributed in this repository. Prepare authorized data locally as `Data/{train,valid,test}/{images,masks}` (test has images only), or pass another path with `--data`. The expected split is 70 train pairs, 20 validation pairs, and 10 unlabeled test images. See the repository's [data and reproducibility note](../../docs/DATA_AND_REPRODUCIBILITY_KO.md).

Use the team's Python 3.10 / PyTorch 2.7.1 + CUDA 12.8 environment where available. Install the project extras with `pip install -r requirements.txt`; C09 also needs `requirements-swin.txt`, and C10 needs the CUDA selective-scan dependency documented in the configs.

## Run the baseline and first comparisons

```bash
python run.py list
python run.py check --config configs/B.json --device cuda --size 448 --backward --data Data
python run.py train --config configs/B.json --data Data --output runs/B_reference --device cuda
python run.py train --config configs/C01.json --data Data --output runs/C01_seed42 --device cuda
python run.py train --config configs/C02.json --data Data --output runs/C02_seed42 --device cuda
```

Run B first as the code-matched baseline. Compare C01/C02 before deciding whether to run C03. Training starts only when a `train` command is explicitly invoked.

R022 reference metrics and actual completed-run summaries are in the repository's [experiment history](../../docs/EXPERIMENT_RESULTS_KO.md). The package config/code was assembled after those runs and does not contain their checkpoints.
