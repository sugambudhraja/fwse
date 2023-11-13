import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import resample

class FWSE(BaseEstimator, TransformerMixin):
    def __init__(self, filter_estimators, wrapper_estimators, n_bootstraps=10, random_state=0):
        self.filter_estimators = filter_estimators
        self.wrapper_estimators = wrapper_estimators
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.feature_ranking_ = None

    def fit(self, X, y):
        
        filter_rankings = []
            
        for estimator in self.filter_estimators:

            estimator_rankings = []
        
            for _ in range(self.n_bootstraps):           
                
                X_bootstrapped, y_bootstrapped = resample(X, y, random_state=self.random_state)
                fitted_estimator = estimator.fit(X_bootstrapped, y_bootstrapped)

                if hasattr(fitted_estimator, 'coef_'):
                    bootstrap_importances = fitted_estimator.coef_.flatten()
                    bootstrap_ranking = np.argsort(np.argsort(-1*bootstrap_importances))

                elif hasattr(fitted_estimator, 'feature_importances_'):
                    bootstrap_importances = fitted_estimator.feature_importances_
                    bootstrap_ranking = np.argsort(np.argsort(-1*bootstrap_importances))

                elif hasattr(fitted_estimator, 'ranking_'):
                    bootstrap_ranking = fitted_estimator.ranking_
        
                estimator_rankings.append(bootstrap_ranking)

            filter_rankings.append(self.aggregate_rankings(estimator_rankings))
        
        aggregated_ranking_filter = self.aggregate_rankings(filter_rankings)
        filtered_features = np.argsort(aggregated_ranking_filter)[:len(aggregated_ranking_filter) // 2]

        wrapper_rankings = []
            
        for estimator in self.wrapper_estimators:

            estimator_rankings = []
        
            for _ in range(self.n_bootstraps):           
                
                X_bootstrapped, y_bootstrapped = resample(X[:, filtered_features], y, random_state=self.random_state)
                fitted_estimator = estimator.fit(X_bootstrapped, y_bootstrapped)

                if hasattr(fitted_estimator, 'coef_'):
                    bootstrap_importances = fitted_estimator.coef_.flatten()
                    bootstrap_ranking = np.argsort(np.argsort(-1*bootstrap_importances))

                elif hasattr(fitted_estimator, 'feature_importances_'):
                    bootstrap_importances = fitted_estimator.feature_importances_
                    bootstrap_ranking = np.argsort(np.argsort(-1*bootstrap_importances))

                elif hasattr(fitted_estimator, 'ranking_'):
                    bootstrap_ranking = fitted_estimator.ranking_

                estimator_rankings.append(bootstrap_ranking)

            wrapper_rankings.append(self.aggregate_rankings(estimator_rankings))
            
        aggregated_ranking_wrapper = self.aggregate_rankings(wrapper_rankings)
        final_ranking = np.array(filtered_features[np.argsort(aggregated_ranking_wrapper)].tolist() + aggregated_ranking_filter[len(aggregated_ranking_filter) // 2:].tolist())
        self.feature_ranking_ = final_ranking 

        return self

    def transform(self, X):
        if self.feature_ranking_ is None:
            raise ValueError("FWSE has not been fitted yet.")
        return X[:, self.feature_ranking_]

    def aggregate_rankings(self, rankings):
        aggregate_ranking = np.argsort(np.argsort(np.sum(rankings, axis=0)))
        return aggregate_ranking
