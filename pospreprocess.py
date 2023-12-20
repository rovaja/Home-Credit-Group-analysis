"""Custom transformer to preprocess previous point of sales data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class POSDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for previous point of sales data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X_agg = self.aggregations(X)
        return X_agg

    @CustomTransformer._logging
    def aggregations(self, X, weight=0.05):
        """Aggregating the previous point of sales data of the current applicant."""
        exclude = ["SK_ID_CURR", "SK_ID_PREV"]

        self.num_columns = self.numerical_columns(
            X, exclude=exclude)
        num_aggre = {}
        for num in self.num_columns:
            if 'CNT' in num:
                num_aggre[num] = ['mean', 'sum']
            else:
                num_aggre[num] = ['max', 'mean']

        X = self.ord_encoding(X)
        X, new_cat_features = self.oh_encoding(X)
        cat_features = new_cat_features + self.ord_new_columns

        cat_aggre = {}
        for cat in cat_features:
            cat_aggre[cat] = ['mean', 'sum']

        X['WEIGHT'] = np.exp(X['MONTHS_BALANCE']*weight)
        X[self.num_columns + cat_features] = X[self.num_columns +
                                               cat_features].multiply(X['WEIGHT'], axis=0)
        X_agg = X.groupby('SK_ID_CURR').agg({**num_aggre, **cat_aggre})
        X_agg.columns = pd.Index(
            ['POS_' + e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()])

        X_agg['POS_COUNT'] = X.groupby('SK_ID_CURR').size()

        X_agg = self.clean_column_names(X_agg)
        return X_agg
