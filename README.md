# FWSE: Filter and Wrapper Stacking Ensemble

Stable feature selection for data with far more features than samples, such as gene expression or proteomics. FWSE is a scikit-learn compatible selector, and the reference implementation of:

> Budhraja S, Doborjeh M, Singh B, Tan S, Doborjeh Z, Lai E, Merkin A, Lee J, Goh W, Kasabov N.
> **Filter and Wrapper Stacking Ensemble (FWSE): a robust approach for reliable biomarker discovery in high-dimensional omics data.**
> *Briefings in Bioinformatics*, 24(6), 2023. https://doi.org/10.1093/bib/bbad382

## Why

Biomarker lists from small omics studies often change when the samples change, or when a different selection method is used. FWSE combines several methods over many resamples of the data, so the features it ranks highly are the ones that keep turning up.

## How it works

1. **Filter stage.** Each filter method (for example ANOVA F and signal-to-noise ratio) scores every feature on bootstrap samples of the data. The ranks are summed across samples and methods.
2. **Prune.** The weakest features are dropped. The default `pruning_factor=0.5` drops half.
3. **Wrapper stage.** Each wrapper method (for example recursive feature elimination with an SVM or logistic regression) ranks the surviving features on fresh bootstrap samples. The ranks are summed again to give the final ranking.

Cheap filters remove the bulk of irrelevant features, so the slower wrappers only run on the features worth their time.

## Install

```bash
pip install git+https://github.com/sugambudhraja/fwse
```

Requires Python 3.9+, NumPy, SciPy and scikit-learn.

## Usage

```python
from sklearn.feature_selection import RFE, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from fwse import FWSE, snr_score

selector = FWSE(
    filter_estimators=[f_classif, snr_score],
    wrapper_estimators=[
        RFE(SVC(kernel="linear"), n_features_to_select=1, step=0.2),
        RFE(LogisticRegression(max_iter=1000), n_features_to_select=1, step=0.2),
    ],
    n_bootstraps=10,
    pruning_factor=0.5,
    n_features_to_select=20,
)
selector.fit(X, y)

selector.feature_ranking_[:20]  # indices of the top 20 features, best first
selector.ranking_               # rank of every feature, 1 = best
X_top = selector.transform(X)   # the top 20 columns
```

It works anywhere a scikit-learn transformer does, including inside a `Pipeline`. Fit it inside your cross-validation loop, not on the full data set, or the test folds will leak into the selection.

### Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `filter_estimators` | required | Filter methods: score functions such as `f_classif` and `snr_score`, or estimators |
| `wrapper_estimators` | required | Wrapper methods, typically `RFE` instances |
| `n_bootstraps` | `10` | Bootstrap samples per method in each stage |
| `pruning_factor` | `0.5` | Share of features removed after the filter stage |
| `n_features_to_select` | `None` | Features kept by `transform`. `None` keeps all, reordered best first |
| `stratify` | `True` | Keep class proportions in each bootstrap sample |
| `random_state` | `0` | Seed for the bootstrap samples |

A method can be any score function `f(X, y)`, or any estimator that exposes `ranking_`, `scores_`, `coef_` or `feature_importances_` after fitting.

## Example

[`examples/quickstart.py`](examples/quickstart.py) runs FWSE on synthetic data with 120 samples and 2,000 features, 20 of which carry signal. It reports how many of the 20 are recovered and how stable the selection is across cross-validation folds:

```
True signal features recovered: 16 of 20
Stability across 5 folds (mean Jaccard): 0.59
```

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Changes in 1.0.0

This release fixes three bugs in the file first published with the paper, so rankings will differ from that version:

- Every bootstrap sample used the same seed, so all samples were identical and `n_bootstraps` had no effect.
- Linear models were ranked by signed coefficient, which put features with large negative coefficients last. They are now ranked by absolute value.
- The pruned half of `feature_ranking_` held rank values where it should have held feature indices, so some features appeared twice and others not at all.

It also adds `pruning_factor`, `n_features_to_select`, `stratify`, the `snr_score` filter, support for score functions as filters, and `ranking_` and `get_support()`. Estimators passed in are now cloned and left unfitted.

## Citation

```bibtex
@article{budhraja2023filter,
  title={Filter and Wrapper Stacking Ensemble (FWSE): a robust approach for reliable biomarker discovery in high-dimensional omics data},
  author={Budhraja, Sugam and Doborjeh, Maryam and Singh, Balkaran and Tan, Samuel and Doborjeh, Zohreh and Lai, Edmund and Merkin, Alexander and Lee, Jimmy and Goh, Wilson and Kasabov, Nikola},
  journal={Briefings in Bioinformatics},
  volume={24},
  number={6},
  pages={bbad382},
  year={2023},
  publisher={Oxford University Press},
  doi={10.1093/bib/bbad382}
}
```
