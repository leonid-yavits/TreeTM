# TreeTM: Taxonomy-Guided Trees of Tsetlin Machine Experts

This repository accompanies the manuscript *Taxonomy-Guided Trees of Tsetlin Machine Experts for Efficient and Interpretable Hierarchical Classification*.
It holds the code, configurations, and per-seed results behind every number, table, and figure of the paper.

TreeTM places one multiclass Tsetlin Machine (TM), an *expert*, at every internal node of a class taxonomy.
Each expert is trained only on the samples of its own subtree, with class-balanced resampling.
With hard routing, only the experts on one root-to-leaf path are evaluated.
Soft routing and confidence-gated routing are also provided.

## Repository layout

| Path | Content |
|---|---|
| `treetm/` | TreeTM framework: data preparation, training, evaluation, baselines, cost model, analysis, paper tables and figures |
| `treetm/work/<dataset>/meta.json` | Taxonomy, leaf classes, class counts, Boolean feature names, and split description of each dataset |
| `treetm/results/<dataset>/` | Per-seed results (see below) |
| `alerce_prep/` | ALeRCE preprocessing, booleanization, and the balanced hierarchical random forest (BHRF) reference |
| `alerce_prep/results/` | BHRF predictions and results, and the earlier ALeRCE analyses |

### Results per dataset (`treetm/results/<dataset>/`)

| Folder | Content |
|---|---|
| `main/seed<s>/` | TreeTM-C (child-balanced resampling) and the flat TM: per-epoch histories, validation and test scores, final clause masks, evaluation, predictions, cost |
| `leafbal/seed<s>/` | TreeTM-L (leaf-balanced resampling). Experts identical to TreeTM-C are symbolic links to `main/` |
| `budget/seed<s>/` | TreeTM-B (budget-matched root). Non-root experts are symbolic links to `main/` |
| `baselines/seed<s>/` | RF, LCPN-RF (hard and soft), HGB, and MLP results and predictions |
| `sweep/C<clauses>/` | Clause-budget sweep (seed 0) |
| `probe/` | Short T and s probes |
| `summary.json`, `summary.txt` | Means, standard deviations, Welch t-tests, and McNemar tests over seeds |

`treetm/results/analysis.json` holds the validation, misrouting, false-alarm, and conditional-error analyses.
TM checkpoints are not included. The final clause masks of every expert are.

## Datasets

The datasets are public and are not redistributed here.

- **NSL-KDD.** Canadian Institute for Cybersecurity. Place `KDDTrain+.txt`, `KDDTest+.txt`, and `training_attack_types.txt` (KDD Cup 1999, UCI Machine Learning Repository, DOI 10.24432/C51C7N) in one directory.
- **ALeRCE light-curve classifier labeled set and features.** Zenodo, DOI 10.5281/zenodo.4279623.
- **Web of Science WOS-46985.** Mendeley Data, DOI 10.17632/9rw3vkcfy4.6, folder `WOS46985` inside `WebOfScience.zip`.

## Requirements

Python 3 with pyTsetlinMachine 0.6.6, scikit-learn 1.9.1, imbalanced-learn 0.14.2, numpy, scipy, pandas, and matplotlib.

## Reproducing the results

Run all commands from `treetm/`.

```bash
# 1. Prepare the data (writes work/<dataset>/prepared.npz)
python ../alerce_prep/prepare.py <ALeRCE data dir> --out ../alerce_prep/work
python datasets.py alerce ../alerce_prep
python datasets.py nslkdd <NSL-KDD dir>
python datasets.py wos    <WOS46985 dir>

# 2. TreeTM-C and the flat TM, one call per seed
python train.py nslkdd --run results/nslkdd/main/seed<s> --clauses 500  --T 50  --s 4 --epochs 50 --per_class 4000 --seed <s>
python train.py alerce --run results/alerce/main/seed<s> --clauses 1000 --T 100 --s 4 --epochs 50 --per_class 4000 --seed <s>
python train.py wos    --run results/wos/main/seed<s>    --clauses 100  --T 10  --s 5 --epochs 30 --per_class 600  --seed <s>

# 3. TreeTM-L and TreeTM-B
./leafbal.sh <dataset> <seed> <nodes> C T s E P    # nodes: root (NSL-KDD: root,attack)
./budget.sh  <dataset> <seed> <P_root> C T s E     # P_root = P*L/K_root: 46000 (NSL-KDD), 20000 (ALeRCE), 11400 (WOS)

# 4. Baselines
python baselines.py nslkdd <s>
python baselines.py alerce <s>
python baselines.py wos <s> --models flat_rf,lcpn_rf,flat_mlp --trees 100

# 5. Evaluation, statistics, cost, and analyses
./eval_all.sh
python aggregate.py <dataset>
python analysis.py
python cost.py <dataset> results/<dataset>/main/seed0
python gated_cost.py wos budget
python interpret.py nslkdd results/nslkdd/main/seed0 dos neptune 2

# 6. Paper tables and figures
python gen_eaai.py <paper dir>
python figs_eaai.py <paper dir>
```

The shell scripts `run_*.sh` and `queue*.sh` list the exact runs and their order.
NSL-KDD uses five seeds (0 to 4). ALeRCE and WOS use three seeds (0 to 2).

## Notes on reproducibility

- Training is resumable. The TM state and the random-number state are checkpointed every epoch.
- The clause-feedback generator of pyTsetlinMachine is process-wide with a fixed initial state, and four expert runs were resumed from checkpoints. Runs are therefore repeatable in distribution but not bit for bit.
- `results/alerce/main/seed0` is the first ALeRCE run, produced with `alerce_prep/run_tms.py`.
- CPU timings (`treetm_cost.json`) come from a bit-packed NumPy implementation on a shared two-core machine. Repeated measurements varied by about 15%.
- Latency and energy on the associative processor are predictions of an analytical model, not measurements.
