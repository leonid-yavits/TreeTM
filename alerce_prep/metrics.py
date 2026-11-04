import numpy as np
from sklearn import metrics as M


def report(y, p, k):
    labels = list(range(k))
    return {
        'accuracy': float(M.accuracy_score(y, p)),
        'balanced_accuracy': float(M.balanced_accuracy_score(y, p)),
        'macro_precision': float(M.precision_score(y, p, labels=labels, average='macro', zero_division=0)),
        'macro_recall': float(M.recall_score(y, p, labels=labels, average='macro', zero_division=0)),
        'macro_f1': float(M.f1_score(y, p, labels=labels, average='macro', zero_division=0)),
        'per_class_recall': M.recall_score(y, p, labels=labels, average=None, zero_division=0).tolist(),
        'per_class_precision': M.precision_score(y, p, labels=labels, average=None, zero_division=0).tolist(),
    }
