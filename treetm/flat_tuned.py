"""Flat TM with the number of clauses per class selected on the validation set
(unclipped class sums). Evaluates results/<ds>/flattuned/seed<s>/flat and writes
results/<ds>/baselines/seed<s>/flat_tm_tuned.json.
Usage: python flat_tuned.py <dataset> <run_dir> [<run_dir> ...]"""
import json, os, sys
import numpy as np
from taxonomy import Taxonomy
from evaluate import load_scores, full_metrics

ds = sys.argv[1]
tax = Taxonomy(json.load(open(f'work/{ds}/meta.json'))['taxonomy'])
P = np.load(f'work/{ds}/prepared.npz')
for run in sys.argv[2:]:
    seed = run.rstrip('/').split('seed')[-1]
    p = load_scores(run, 'flat', 'test').argmax(1)
    r = full_metrics(tax, P['yte'], p)
    r['experts_per_input'] = 1.0
    r['clauses_per_class'] = json.load(open(f'{run}/flat_hist.json'))['cfg']['clauses']
    r['val_accuracy'] = float((load_scores(run, 'flat', 'val').argmax(1) == P['yva']).mean())
    od = f'results/{ds}/baselines/seed{seed}'; os.makedirs(od, exist_ok=True)
    json.dump(r, open(f'{od}/flat_tm_tuned.json', 'w'), indent=1)
    print(run, 'acc %.4f hF %.4f val %.4f' % (r['accuracy'], r['hF'], r['val_accuracy']))
