"""NSL-KDD: detection of unseen attack types by the non-TM baselines.
Retrains the baselines of baselines.py with the same settings and seed, checks
that the test predictions reproduce the saved ones, and reports the share of
the novel test records (attack types absent from training) that each model
assigns to the attack branch, the detection rate of known attacks, and the
false-alarm rate on normal records.
Usage: python novel_baselines.py <seed>  -> results/nslkdd/baselines/seed<s>/novel.json"""
import json, sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from taxonomy import Taxonomy

seed = int(sys.argv[1])
tax = Taxonomy(json.load(open('work/nslkdd/meta.json'))['taxonomy'])
P = np.load('work/nslkdd/prepared.npz')
Ftr, Fte, Fnov, ytr, yte = P['Ftr'], P['Fte'], P['Fnov'], P['ytr'], P['yte']
od = f'results/nslkdd/baselines/seed{seed}'
lv1 = lambda y: np.array(tax.level_label(y, 1))
normal = yte == tax.leaf_index['normal']
rf = lambda: RandomForestClassifier(n_estimators=300, class_weight='balanced_subsample', n_jobs=2, random_state=seed)
out = {}


def record(name, pt, pn_attack):
    saved = np.load(f'{od}/{name}_pred.npy')
    out[name] = {'novel_detected_as_attack': float(np.mean(pn_attack)),
                 'known_detected_as_attack': float((lv1(pt[~normal]) == 'attack').mean()),
                 'false_alarm': float((lv1(pt[normal]) == 'attack').mean()),
                 'reproduces_saved_test_predictions': bool(np.array_equal(pt, saved))}
    print(seed, name, out[name], flush=True)


c = rf().fit(Ftr, ytr)
record('flat_rf', c.classes_[c.predict_proba(Fte).argmax(1)],
       lv1(c.classes_[c.predict_proba(Fnov).argmax(1)]) == 'attack')
c = HistGradientBoostingClassifier(max_iter=200, class_weight='balanced', early_stopping=True, random_state=seed).fit(Ftr, ytr)
record('flat_hgb', c.predict(Fte), lv1(c.predict(Fnov)) == 'attack')
sc = StandardScaler().fit(Ftr)
c = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=200, random_state=seed).fit(sc.transform(Ftr), ytr)
record('flat_mlp', c.predict(sc.transform(Fte)), lv1(c.predict(sc.transform(Fnov))) == 'attack')
# LCPN-RF: the root forest decides normal versus attack for hard routing
keep, lab = tax.node_labels('root', ytr)
c = rf().fit(Ftr[keep], lab[keep])
k = tax.children['root'].index('attack')
pr = np.zeros((len(Fnov), len(tax.children['root']))); pr[:, c.classes_] = c.predict_proba(Fnov)
saved = np.load(f'{od}/lcpn_rf_hard_pred.npy')
out['lcpn_rf_hard'] = {'novel_detected_as_attack': float((pr.argmax(1) == k).mean()),
                       'known_detected_as_attack': float((lv1(saved[~normal]) == 'attack').mean()),
                       'false_alarm': float((lv1(saved[normal]) == 'attack').mean())}
print(seed, 'lcpn_rf_hard', out['lcpn_rf_hard'], flush=True)
json.dump(out, open(f'{od}/novel.json', 'w'), indent=1)
