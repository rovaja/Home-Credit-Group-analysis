"""Custom transformer to preprocess installment data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class InstDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for installment data."""

    def fit(self, X, y=None):
        for col in X.columns:
            if any(x in ['XNA'] for x in X[col].unique().tolist()):
                self.xna_columns.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.feature_engineering(X)
        X_agg = self.aggregations(X)
        return X_agg

    @CustomTransformer._logging
    def feature_engineering(self, X):
        """Feature engineering"""
        X['DAYS_PASS_DUE'] = X['DAYS_ENTRY_PAYMENT'] - X['DAYS_INSTALMENT']
        X['DAYS_BEFORE_DUE'] = X['DAYS_INSTALMENT'] - X['DAYS_ENTRY_PAYMENT']
        X['DAYS_PASS_DUE'] = X['DAYS_PASS_DUE'].apply(
            lambda x: x if x > 0 else 0)
        X['DAYS_BEFORE_DUE'] = X['DAYS_BEFORE_DUE'].apply(
            lambda x: x if x > 0 else 0)
        X['PAYMENT_INST_RATIO'] = X['AMT_PAYMENT'] / X['AMT_INSTALMENT']
        X.loc[X['AMT_INSTALMENT'] == 0, 'PAYMENT_INST_RATIO'] = np.nan
        X['INST_PAYMENT_DIFFERENCE'] = X['AMT_INSTALMENT'] - X['AMT_PAYMENT']
        return X

    @CustomTransformer._logging
    def aggregations(self, X, weight=0.05):
        """Aggregating the installment data of the current applicant."""
        exclude = ["SK_ID_CURR", "SK_ID_PREV"]

        self.num_columns = self.numerical_columns(
            X, exclude=exclude)
        num_aggre = {}
        for num in self.num_columns:
            if num == 'NUM_INSTALMENT_VERSION':
                num_aggre[num] = ['nunique']
            elif num == 'DAYS_PASS_DUE' or num == 'DAYS_BEFORE_DUE':
                num_aggre[num] = ['max', 'mean', 'sum']
            else:
                num_aggre[num] = ['max', 'mean']

        X['WEIGHT'] = np.exp(X['DAYS_INSTALMENT']*weight)
        X[self.num_columns] = X[self.num_columns].multiply(X['WEIGHT'], axis=0)
        X_agg = X.groupby('SK_ID_CURR').agg({**num_aggre})
        X_agg.columns = pd.Index(
            ['INSTAL_' + e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()])

        X_agg['INSTAL_COUNT'] = X.groupby('SK_ID_CURR').size()

        X_agg = self.clean_column_names(X_agg)
        return X_agg
