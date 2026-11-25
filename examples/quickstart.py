"""FWSE on synthetic data shaped like an omics study: many features, few samples.

20 of 2,000 features carry signal. We check how many FWSE recovers in its
top 20, and how stable that top 20 is across cross-validation folds.

Run from the repository root:  python examples/quickstart.py
"""

import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

from fwse import FWSE, snr_score

N_INFORMATIVE, TOP = 20, 20

# 120 samples, 2,000 features. The first 20 differ between the two classes.
rng = np.random.RandomState(0)
y = np.repeat([0, 1], 60)
X = rng.normal(size=(120, 2000))
X[:, :N_INFORMATIVE] += 0.8 * y[:, None]


def make_selector():
    return FWSE(
        filter_estimators=[f_classif, snr_score],
        wrapper_estimators=[
            RFE(SVC(kernel="linear"), n_features_to_select=1, step=0.2),
            RFE(LogisticRegression(max_iter=1000), n_features_to_select=1, step=0.2),
        ],
        n_bootstraps=10,
        pruning_factor=0.5,
        n_features_to_select=TOP,
    )


selector = make_selector().fit(X, y)
top = selector.feature_ranking_[:TOP]
print(f"Top {TOP} features: {sorted(top.tolist())}")
print(f"True signal features recovered: {np.sum(top < N_INFORMATIVE)} of {N_INFORMATIVE}")

# Stability: average Jaccard index between the top sets chosen on each fold.
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
top_sets = [
    set(make_selector().fit(X[train], y[train]).feature_ranking_[:TOP].tolist())
    for train, _ in folds.split(X, y)
]
jaccard = [len(a & b) / len(a | b) for a, b in combinations(top_sets, 2)]
print(f"Stability across 5 folds (mean Jaccard): {np.mean(jaccard):.2f}")
