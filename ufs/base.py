import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

class BaseEstimatorUFS(BaseEstimator, TransformerMixin):
    
    def __init__(self, nFeaturesOut=None):
        self.nFeaturesOut = nFeaturesOut

    def fit(self,X,y=None):
        X = self._check_X(X)
        self._core(X)
        self.selected_index = self._order()
        return self

    def transform(self,X):
        check_is_fitted(self, "selected_index_")
        X = self._check_X(X)
        return X[:, self.selected_index]

    def _core(self,X):
        raise NotImplementedError
    
    def _order(self):
        k = self.nFeaturesOut or len(self.scores_)
        return np.argsort(self.scores_)[::-1][:k]

    @staticmethod
    def _check_X(X):
        X = np.asarray(X, dtype=float)

        if X.ndim != 2:
            raise ValueError("X debe ser 2-D con formato (n_samples, n_features)")
        
        return X




