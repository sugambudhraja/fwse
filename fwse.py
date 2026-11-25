"""Filter and Wrapper Stacking Ensemble (FWSE) feature selection.

Reference:
    Budhraja et al., "Filter and Wrapper Stacking Ensemble (FWSE): a robust
    approach for reliable biomarker discovery in high-dimensional omics data",
    Briefings in Bioinformatics 24(6), 2023. https://doi.org/10.1093/bib/bbad382
"""

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, clone
from sklearn.feature_selection import SelectorMixin
from sklearn.utils import check_random_state, check_X_y, resample
from sklearn.utils.validation import check_is_fitted

__all__ = ["FWSE", "snr_score"]
__version__ = "1.0.0"


def snr_score(X, y):
    """Signal-to-noise ratio of each feature for a two-class problem.

    SNR = |mean_a - mean_b| / (std_a + std_b). Higher means more discriminative.
    """
    X, y = check_X_y(X, y)
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError("snr_score needs exactly two classes, got %d." % len(classes))
    a, b = X[y == classes[0]], X[y == classes[1]]
    spread = a.std(axis=0) + b.std(axis=0)
    diff = np.abs(a.mean(axis=0) - b.mean(axis=0))
    return np.divide(diff, spread, out=np.zeros_like(diff, dtype=float), where=spread > 0)


def _rank_from_scores(scores):
    """Turn importance scores into ranks, where rank 1 is the most important."""
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    return rankdata(-scores, method="average")


def _rank_features(method, X, y):
    """Rank every column of X with one filter or wrapper method (1 = best)."""
    if callable(method) and not hasattr(method, "fit"):
        result = method(X, y)
        # Score functions such as sklearn's f_classif return (scores, p-values).
        scores = result[0] if isinstance(result, tuple) else result
        return _rank_from_scores(scores)

    fitted = clone(method).fit(X, y)
    if hasattr(fitted, "ranking_"):  # RFE, RFECV
        return rankdata(fitted.ranking_, method="average")
    if hasattr(fitted, "scores_"):  # SelectKBest and friends
        return _rank_from_scores(fitted.scores_)
    if hasattr(fitted, "coef_"):  # linear models: size of the coefficient
        coef = np.abs(np.atleast_2d(fitted.coef_))
        return _rank_from_scores(coef.mean(axis=0))
    if hasattr(fitted, "feature_importances_"):  # tree ensembles
        return _rank_from_scores(fitted.feature_importances_)
    raise TypeError(
        "%s exposes none of ranking_, scores_, coef_ or feature_importances_ "
        "after fitting, so FWSE cannot rank features with it." % type(fitted).__name__
    )


def _aggregate(rankings):
    """Rank aggregation: sum the ranks per feature, then rank the sums."""
    return rankdata(np.sum(rankings, axis=0), method="average")


class FWSE(SelectorMixin, BaseEstimator):
    """Filter and Wrapper Stacking Ensemble feature selector.

    Stage 1 runs every filter method on bootstrap samples of the data,
    aggregates the rankings, and prunes the weakest features. Stage 2 runs
    every wrapper method on fresh bootstrap samples of the surviving features
    and aggregates again to give the final ranking.

    Parameters
    ----------
    filter_estimators : list
        Filter methods. Each is either a score function ``f(X, y)`` returning
        scores or ``(scores, pvalues)``, such as ``f_classif`` or ``snr_score``,
        or an unfitted estimator that exposes ``scores_``, ``coef_``,
        ``feature_importances_`` or ``ranking_`` after ``fit``.
    wrapper_estimators : list
        Wrapper methods, typically ``RFE`` instances. Same rules as above.
    n_bootstraps : int, default=10
        Bootstrap samples drawn per method in each stage.
    pruning_factor : float in [0, 1), default=0.5
        Share of features removed after the filter stage.
    n_features_to_select : int or None, default=None
        Features kept by ``transform``. ``None`` keeps all of them, reordered
        from most to least important.
    stratify : bool, default=True
        Keep class proportions in each bootstrap sample. This avoids samples
        with a single class when the data set is small.
    random_state : int, RandomState instance or None, default=0

    Attributes
    ----------
    ranking_ : ndarray of shape (n_features,)
        Rank of each feature, where 1 is the most important.
    feature_ranking_ : ndarray of shape (n_features,)
        Feature indices ordered from most to least important.
    filter_support_ : ndarray of bool, shape (n_features,)
        Features that survived the filter stage.
    """

    def __init__(
        self,
        filter_estimators,
        wrapper_estimators,
        n_bootstraps=10,
        pruning_factor=0.5,
        n_features_to_select=None,
        stratify=True,
        random_state=0,
    ):
        self.filter_estimators = filter_estimators
        self.wrapper_estimators = wrapper_estimators
        self.n_bootstraps = n_bootstraps
        self.pruning_factor = pruning_factor
        self.n_features_to_select = n_features_to_select
        self.stratify = stratify
        self.random_state = random_state

    def _stage(self, methods, X, y, rng):
        """Rank features with each method over bootstrap samples and aggregate."""
        per_method = []
        for method in methods:
            rankings = []
            for _ in range(self.n_bootstraps):
                X_boot, y_boot = resample(
                    X,
                    y,
                    stratify=y if self.stratify else None,
                    random_state=rng.randint(np.iinfo(np.int32).max),
                )
                rankings.append(_rank_features(method, X_boot, y_boot))
            per_method.append(_aggregate(rankings))
        return _aggregate(per_method)

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        if not self.filter_estimators or not self.wrapper_estimators:
            raise ValueError("FWSE needs at least one filter and one wrapper method.")
        if self.n_bootstraps < 1:
            raise ValueError("n_bootstraps must be at least 1.")
        if not 0 <= self.pruning_factor < 1:
            raise ValueError("pruning_factor must be in [0, 1).")

        rng = check_random_state(self.random_state)
        n_features = X.shape[1]
        self.n_features_in_ = n_features

        # Stage 1: filters on all features, then prune the weakest.
        filter_rank = self._stage(self.filter_estimators, X, y, rng)
        filter_order = np.argsort(filter_rank, kind="stable")
        n_keep = max(1, int(round(n_features * (1 - self.pruning_factor))))
        kept, pruned = filter_order[:n_keep], filter_order[n_keep:]

        # Stage 2: wrappers on the surviving features only.
        wrapper_rank = self._stage(self.wrapper_estimators, X[:, kept], y, rng)
        kept_order = kept[np.argsort(wrapper_rank, kind="stable")]

        # Survivors in wrapper order, then pruned features in filter order.
        self.feature_ranking_ = np.concatenate([kept_order, pruned])
        self.ranking_ = np.empty(n_features, dtype=int)
        self.ranking_[self.feature_ranking_] = np.arange(1, n_features + 1)
        self.filter_support_ = np.zeros(n_features, dtype=bool)
        self.filter_support_[kept] = True
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "ranking_")
        k = self.n_features_in_ if self.n_features_to_select is None else self.n_features_to_select
        return self.ranking_ <= k

    def transform(self, X):
        """Keep the top features, ordered from most to least important."""
        check_is_fitted(self, "feature_ranking_")
        k = self.n_features_in_ if self.n_features_to_select is None else self.n_features_to_select
        return np.asarray(X)[:, self.feature_ranking_[:k]]
