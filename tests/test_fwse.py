import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from fwse import FWSE, snr_score


@pytest.fixture(scope="module")
def data():
    # 80 samples, 200 features; the first 8 carry a mean shift between classes.
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1], 40)
    X = rng.normal(size=(80, 200))
    X[:, :8] += 1.5 * y[:, None]
    return X, y


def make(**kwargs):
    params = dict(
        filter_estimators=[f_classif, snr_score],
        wrapper_estimators=[RFE(LogisticRegression(max_iter=500), n_features_to_select=1, step=0.25)],
        n_bootstraps=5,
    )
    params.update(kwargs)
    return FWSE(**params)


def test_ranking_is_a_permutation(data):
    X, y = data
    fwse = make().fit(X, y)
    assert sorted(fwse.feature_ranking_.tolist()) == list(range(X.shape[1]))
    assert sorted(fwse.ranking_.tolist()) == list(range(1, X.shape[1] + 1))
    assert fwse.ranking_[fwse.feature_ranking_[0]] == 1


def test_recovers_informative_features(data):
    X, y = data
    top = make().fit(X, y).feature_ranking_[:8]
    assert np.sum(top < 8) >= 6


def test_pruned_features_rank_below_survivors(data):
    X, y = data
    fwse = make(pruning_factor=0.75).fit(X, y)
    assert fwse.filter_support_.sum() == 50
    assert fwse.ranking_[fwse.filter_support_].max() == 50


def test_transform_and_support(data):
    X, y = data
    fwse = make(n_features_to_select=10).fit(X, y)
    Xt = fwse.transform(X)
    assert Xt.shape == (80, 10)
    assert np.array_equal(Xt, X[:, fwse.feature_ranking_[:10]])
    assert fwse.get_support().sum() == 10
    assert make().fit(X, y).transform(X).shape == X.shape


def test_reproducible_and_seed_sensitive(data):
    X, y = data
    a = make(random_state=1).fit(X, y).feature_ranking_
    b = make(random_state=1).fit(X, y).feature_ranking_
    c = make(random_state=2).fit(X, y).feature_ranking_
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_bootstraps_differ(data):
    X, y = data
    one = make(n_bootstraps=1).fit(X, y).feature_ranking_
    many = make(n_bootstraps=8).fit(X, y).feature_ranking_
    assert not np.array_equal(one, many)


def test_negative_coefficients_count_as_important():
    rng = np.random.RandomState(0)
    y = np.repeat([0, 1], 50)
    X = rng.normal(size=(100, 20))
    X[:, 3] -= 3.0 * y  # strongly informative, negative coefficient
    fwse = FWSE([LogisticRegression()], [LogisticRegression()], n_bootstraps=3).fit(X, y)
    assert fwse.feature_ranking_[0] == 3


def test_accepts_estimator_filters_and_tree_wrappers(data):
    X, y = data
    fwse = FWSE(
        filter_estimators=[SelectKBest(f_classif, k="all")],
        wrapper_estimators=[RandomForestClassifier(n_estimators=30, random_state=0)],
        n_bootstraps=2,
    ).fit(X, y)
    assert len(fwse.feature_ranking_) == X.shape[1]


def test_does_not_mutate_inputs(data):
    X, y = data
    lr = LogisticRegression(max_iter=500)
    FWSE([f_classif], [lr], n_bootstraps=2).fit(X, y)
    assert not hasattr(lr, "coef_")


def test_works_in_a_pipeline(data):
    X, y = data
    pipe = make_pipeline(make(n_features_to_select=10), LogisticRegression(max_iter=500))
    assert pipe.fit(X, y).score(X, y) > 0.8


def test_rejects_bad_input(data):
    X, y = data
    with pytest.raises(ValueError):
        make(pruning_factor=1.0).fit(X, y)
    with pytest.raises(ValueError):
        FWSE([], [LogisticRegression()]).fit(X, y)
    with pytest.raises(TypeError):
        FWSE([f_classif], [make_pipeline(LogisticRegression())], n_bootstraps=1).fit(X, y)
    with pytest.raises(ValueError):
        snr_score(X, np.arange(len(y)) % 3)
