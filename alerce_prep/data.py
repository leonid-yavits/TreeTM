"""ALeRCE light-curve classifier data loading and booleanization.

Preprocessing replicates the authors' training notebook
(alercebroker/light_curve_classifier_SanchezSaez_2020,
training_deployed_BRF_alerts_20200609.ipynb):
  * keep the 15 taxonomy classes,
  * drop the columns listed in rm_nd_cols, plus Mean_1 and Mean_2,
  * replace +/-inf by NaN, then fill NaN with -999.
"""
import os
import tarfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

LEAVES = ['SNIa', 'SNIbc', 'SNII', 'SLSN',
          'QSO', 'AGN', 'Blazar', 'YSO', 'CV/Nova',
          'LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other']
TOP = ['Transient', 'Stochastic', 'Periodic']
TREE = {
    'Transient': ['SNIa', 'SNIbc', 'SNII', 'SLSN'],
    'Stochastic': ['QSO', 'AGN', 'Blazar', 'YSO', 'CV/Nova'],
    'Periodic': ['LPV', 'E', 'DSCT', 'RRL', 'CEP', 'Periodic-Other'],
}
PARENT = {leaf: top for top, leaves in TREE.items() for leaf in leaves}

RM_COLS = ['n_det_1', 'n_det_2', 'n_pos_1', 'n_pos_2', 'n_neg_1', 'n_neg_2',
           'first_mag_1', 'first_mag_2', 'MHPS_non_zero_1', 'MHPS_non_zero_2',
           'MHPS_PN_flag_1', 'MHPS_PN_flag_2', 'mean_mag_1', 'mean_mag_2',
           'min_mag_1', 'min_mag_2', 'W1', 'W2', 'W3', 'W4', 'iqr_1', 'iqr_2',
           'delta_mjd_fid_1', 'delta_mjd_fid_2', 'last_mjd_before_fid_1',
           'last_mjd_before_fid_2', 'g-r_ml', 'MHAOV_Period_1', 'MHAOV_Period_2',
           'Mean_1', 'Mean_2']
MISSING = -999.0


def _find(root, name):
    for d, _, files in os.walk(root):
        if name in files:
            return os.path.join(d, name)
    return None


def load_raw(path):
    """Return a DataFrame indexed by oid with features and column 'classALeRCE'.

    `path` may be a parquet file holding the joined table, a directory holding
    the two Zenodo CSVs, or the Zenodo tar.gz archive.
    """
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    if path.endswith('.tar.gz') or path.endswith('.tgz'):
        out = os.path.join(os.path.dirname(path), 'zenodo_extracted')
        os.makedirs(out, exist_ok=True)
        with tarfile.open(path) as tf:
            members = [m for m in tf.getmembers()
                       if m.name.endswith(('labeled_set_lc_classifier_SanchezSaez_2020.csv',
                                           'features_for_lc_classifier_20200609.csv'))]
            tf.extractall(out, members=members, filter='data')
        path = out
    lab_f = _find(path, 'labeled_set_lc_classifier_SanchezSaez_2020.csv')
    feat_f = _find(path, 'features_for_lc_classifier_20200609.csv')
    lab = pd.read_csv(lab_f, index_col='oid')
    feat = pd.read_csv(feat_f, index_col='oid')
    return feat.join(lab[['classALeRCE']], how='inner')


def preprocess(df):
    df = df[df['classALeRCE'].isin(LEAVES)].copy()
    y_leaf = np.asarray(df['classALeRCE'].astype(str).tolist(), dtype=object)
    X = df.drop(columns=['classALeRCE'] + [c for c in RM_COLS if c in df.columns])
    X = X.replace([np.inf, -np.inf], np.nan).fillna(MISSING)
    X = X.select_dtypes(include=[np.number])
    return X, y_leaf


def split(X, y_leaf, seed=0, test_size=0.2, val_size=0.1):
    """Stratified 80/20 train/test split (as in the notebook), then a
    stratified validation split carved out of the training part."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_leaf, test_size=test_size, stratify=y_leaf, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


class Thermometer:
    """Per-feature thermometer encoding with quantile thresholds plus one
    'missing' bit. Missing values (-999) set the missing bit and leave all
    thermometer bits at 0."""

    def __init__(self, n_bins=8):
        self.n_bins = n_bins

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.thresholds = []
        qs = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        for j in range(X.shape[1]):
            v = X[:, j]
            v = v[v != MISSING]
            th = np.unique(np.quantile(v, qs)) if v.size else np.array([])
            self.thresholds.append(th)
        self.has_missing = (X == MISSING).any(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        cols = []
        for j, th in enumerate(self.thresholds):
            v = X[:, j]
            miss = v == MISSING
            if th.size:
                b = (v[:, None] >= th[None, :]) & ~miss[:, None]
                cols.append(b)
            if self.has_missing[j]:
                cols.append(miss[:, None])
        return np.ascontiguousarray(np.hstack(cols).astype(np.uint32))

    @property
    def n_bits(self):
        return int(sum(t.size for t in self.thresholds) + self.has_missing.sum())
