"""Custom transformer to preprocess credit balance data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class CreditDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for credit balance data."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = self.flags_for_missing_vals(X)
        X_agg = self.aggregations(X)
        return X_agg

    @CustomTransformer._logging
    def flags_for_missing_vals(self, X):
        """Create flag features for missing values."""

        X["FLAG_MISSING_INST"] = 0
        X.loc[
            X.AMT_INST_MIN_REGULARITY.isna(), "FLAG_MISSING_INST"
        ] = 1

        X["FLAG_MISSING_DRAWING"] = 0
        X.loc[
            (X.AMT_DRAWINGS_ATM_CURRENT.isna()) | (
                X.AMT_PAYMENT_CURRENT.isna()), "FLAG_MISSING_DRAWING"
        ] = 1

        return X

    @CustomTransformer._logging
    def aggregations(self, X, weight=0.05):
        """Aggregating the credit balance data of the current applicant."""
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
            ['CC_' + e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()])

        X_agg['CC_COUNT'] = X.groupby('SK_ID_CURR').size()

        X_agg = self.clean_column_names(X_agg)
        return X_agg
