from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import sklearn
sklearn.set_config(transform_output="pandas")


class DatatypeToCategorical(BaseEstimator, TransformerMixin):
    """Encode features into a categorical pandas features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Series(X[col], dtype="category")
        return X

class FeaturesToSelect(BaseEstimator, TransformerMixin):
    """Return only the selected features. 
    If features to select is not specified will return all features."""
    def __init__(self,  feature_names: list[str] = []):
        self.feature_names = feature_names
        
    def fit(self, X, y=None):
        if self.feature_names:
            if any(col not in X.columns for col in self.feature_names):
                invalid_features = [col for col in self.feature_names if col not in X.columns]
                raise ValueError(f"Invalid feature indexes: {invalid_features}")
        else:
            self.feature_names = X.columns.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        return X.loc[:,self.feature_names]
