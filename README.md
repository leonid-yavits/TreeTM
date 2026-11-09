# TreeTM: Taxonomy-Guided Trees of Tsetlin Machine Experts

This repository accompanies the manuscript *Taxonomy-Guided Trees of Tsetlin Machine Experts for Efficient and Interpretable Hierarchical Classification*.
It holds the code, configurations, and per-seed results behind every number, table, and figure of the paper.

TreeTM places one multiclass Tsetlin Machine (TM), an *expert*, at every internal node of a class taxonomy.
Each expert is trained only on the samples of its own subtree, with class-balanced resampling.
Hard routing evaluates only the experts on one root-to-leaf path.
Two training mechanisms act on the routing experts: leaf-balanced resampling and a budget-matched root.

## Configuration reported in the paper

| Dataset | TreeTM | Flat TM, same C | Flat TM, validation-selected C |
|---|---|---|---|
| NSL-KDD (5 seeds) | `results/nslkdd/leafbal/`: leaf-balanced resampling at the root and the attack expert, C = 500 | `results/nslkdd/main/` | `results/nslkdd/flattuned/`, C = 1000 |
| ALeRCE (3 seeds) | `results/alerce/budget/`: budget-matched root (P_root = 20000), C = 1000 | `results/alerce/main/` | validation selects C = 1000, the C of the main runs |

All paths are relative to `treetm/`.
`gen_eaai.py` and `figs_eaai.py` regenerate every number, table, and figure of the paper from these files.

## Repository layout

| Path | Content |
|---|---|
| `treetm/` | TreeTM code: data preparation, training, evaluation, cost model, analyses, paper tables and figure |
| `treetm/run_paper.sh` | Every run and analysis behind the paper, in order |
| `treetm/work/<dataset>/meta.json` | Taxonomy, leaf classes, class counts, Boolean feature names, and split of each dataset |
| `treetm/results/<dataset>/` | Per-seed results (see below) |
| `alerce_prep/` | ALeRCE preprocessing and booleanization, and the training code of ALeRCE seed 0 (`run_tms.py`) |

### Results per dataset (`treetm/results/<dataset>/`)

| Folder | Content |
|---|---|
| `main/seed<s>/` | Flat TM and TreeTM with child-balanced resampling at every expert and P_root = P (the "Child-balanced" rows of Table 3): training histories, validation and test class sums, final clause masks, evaluation, predictions |
| `leafbal/seed<s>/` (NSL-KDD) | TreeTM. Root and attack expert trained with leaf-balanced resampling. The other experts and the flat TM are copies of the files in `main/` |
| `budget/seed<s>/` (ALeRCE) | TreeTM. Budget-matched root. The other experts and the flat TM are copies of the files in `main/` |
| `sweep/C<clauses>/` | Flat TM with other clause budgets (seed 0), used to select C on the validation set |
| `flattuned/seed<s>/` (NSL-KDD) | Flat TM with the validation-selected C = 1000 and its evaluation (`flat_eval.json`) |
| `probe/` (NSL-KDD) | Short T and s probes |

Each run folder holds `treetm_eval.json` (test metrics) and `treetm_predictions.npz`.
`leafbal/seed0` and `budget/seed0` also hold `treetm_cost.json` (literal tests, CPU time, and the AP(TM) model).
`results/analysis.json` holds the NSL-KDD misrouting counts by leaf.
The NSL-KDD runs also hold the class sums of the test records of unseen attack types (`*_novel_uscores.npy`).
The repository contains no symbolic links, so a downloaded archive runs as is.
TM checkpoints are not included. The final clause masks of every expert are.

## Datasets

The datasets are public and are not redistributed here.

- **NSL-KDD.** Canadian Institute for Cybersecurity. Place `KDDTrain+.txt`, `KDDTest+.txt`, and `training_attack_types.txt` (KDD Cup 1999, UCI Machine Learning Repository, DOI 10.24432/C51C7N) in one directory.
- **ALeRCE light-curve classifier labeled set and features.** Zenodo, DOI 10.5281/zenodo.4279623.

## Requirements

Python 3 with pyTsetlinMachine 0.6.6, scikit-learn, numpy, scipy, pandas, and matplotlib.

## Reproducing the results

Run all commands from `treetm/`.

```bash
# 0. Prepare the data (writes work/<dataset>/prepared.npz)
cd ../alerce_prep && python prepare.py <ALeRCE data dir> && cd ../treetm
python datasets.py alerce ../alerce_prep
python datasets.py nslkdd <NSL-KDD dir>

# 1-6. Training, evaluation, cost, analyses, and the paper tables and figure
./run_paper.sh <paper dir>
```

## Notes on reproducibility

- Training is resumable. The TM state and the random-number state are checkpointed every epoch.
- The clause-feedback generator of pyTsetlinMachine is process-wide with a fixed initial state, and four expert runs were resumed from checkpoints. Runs are therefore repeatable in distribution but not bit for bit.
- `results/alerce/main/seed0` was produced with `alerce_prep/run_tms.py` and the same configuration as the other ALeRCE seeds.
- CPU timings (`treetm_cost.json`) come from a bit-packed NumPy implementation on a shared two-core machine. Repeated measurements varied by about 15%.
- Latency and energy on the associative processor are predictions of an analytical model, not measurements.

## License

MIT License, see `LICENSE`.
