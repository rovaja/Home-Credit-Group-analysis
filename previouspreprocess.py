"""Custom transformer to preprocess previous application data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class PreAppDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for previous application data."""

    def fit(self, X, y=None):
        for col in X.columns:
            if any(x in ['XNA'] for x in X[col].unique().tolist()):
                self.xna_columns.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.flags_for_missing_vals(X)
        X = self.replace_value(X, [
            "DAYS_FIRST_DRAWING",
            "DAYS_FIRST_DUE",
            "DAYS_LAST_DUE_1ST_VERSION",
            "DAYS_LAST_DUE",
            "DAYS_TERMINATION",
        ], [365243], np.nan)
        X = self.replace_value(X, self.xna_columns, ['XNA'], np.nan)
        X = self.replace_value(X, ['SELLERPLACE_AREA'], [-1], np.nan)
        X = self.combining_rare_categories(X, 'NAME_CASH_LOAN_PURPOSE')
        X = self.combining_rare_categories(X, 'NAME_GOODS_CATEGORY')
        X = self.feature_engineering(X)
        X_agg = self.aggregations(X)
        return X_agg

    @CustomTransformer._logging
    def flags_for_missing_vals(self, X):
        """Create flag features for missing values. 
        First for missing drawing and ue date information. 
        Then a flag for missing annuity or payment count information"""

        X["FLAG_MISSING_FIRST_DRAW"] = 0
        X.loc[
            X.NFLAG_INSURED_ON_APPROVAL.isna(), "FLAG_MISSING_FIRST_DRAW"
        ] = 1

        X["FLAG_MISSING_ANNUITY"] = 0
        X.loc[
            (X.CNT_PAYMENT.isna()) | (X.AMT_ANNUITY.isna()), "FLAG_MISSING_ANNUITY"
        ] = 1

        return X

    @CustomTransformer._logging
    def feature_engineering(self, X):
        """Feature engineering"""

        X["INTEREST_AMT"] = (
            X["CNT_PAYMENT"] *
            X["AMT_ANNUITY"] - X["AMT_CREDIT"]
        )
        X["INTEREST_RATIO"] = X["INTEREST_AMT"] / \
            X["AMT_CREDIT"]
        X["INTEREST_RATE"] = (
            2 * 12
            * X["INTEREST_AMT"]
            / (X["AMT_CREDIT"] * (X["CNT_PAYMENT"] + 1))
        )
        X.loc[X['CNT_PAYMENT'] == 0, 'INTEREST_RATE'] = np.nan
        X["AMT_REFUSED"] = X["AMT_APPLICATION"] - \
            X["AMT_CREDIT"]
        X["AMT_REFUSED_RATIO"] = (
            X["AMT_APPLICATION"] / X["AMT_CREDIT"]
        )
        X["AMT_CREDIT_GOODS_RATIO"] = (
            X["AMT_CREDIT"] / X["AMT_GOODS_PRICE"]
        )
        X.loc[X['AMT_GOODS_PRICE'] == 0, 'AMT_CREDIT_GOODS_RATIO'] = np.nan
        X["MISSING_VALUES_RATIO"] = X.isna().mean(axis=1)
        X["CREDIT_DOWNPAYMENT_RATIO"] = (
            X["AMT_DOWN_PAYMENT"] / X["AMT_CREDIT"]
        )

        X["REFUSED_APPLICATION"] = 0
        X.loc[X['NAME_CONTRACT_STATUS'] ==
              'Refused', "REFUSED_APPLICATION"] = 1
        self.ord_columns['REFUSED_APPLICATION'] = []
        return X

    @CustomTransformer._logging
    def aggregations(self, X, weight=0.05):
        """Aggregating the previous application information of the current applicant."""
        self.num_columns = self.numerical_columns(
            X, exclude=["SK_ID_PREV", "SK_ID_CURR"])

        num_aggre = {}
        for num in self.num_columns:
            num_aggre[num] = ['mean', 'max']

        X = self.ord_encoding(X)
        X, new_cat_features = self.oh_encoding(X)
        cat_features = new_cat_features + self.ord_new_columns

        cat_aggre = {}
        for cat in cat_features:
            cat_aggre[cat] = ['mean', 'sum']

        X = X.sort_values(
            by=["SK_ID_CURR", "DAYS_DECISION"], ascending=[True, False])

        X_app = X[X['NAME_CONTRACT_STATUS'] == 3]

        X['WEIGHT'] = np.exp(X['DAYS_DECISION']*weight)
        X[self.num_columns + cat_features] = X[self.num_columns +
                                               cat_features].multiply(X['WEIGHT'], axis=0)
        X_agg = X.groupby('SK_ID_CURR').agg({**num_aggre, **cat_aggre})
        X_agg.columns = pd.Index(
            ['PREV_' + e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()])

        X_app_agg = X_app.groupby('SK_ID_CURR').agg({**num_aggre})
        X_app_agg.columns = pd.Index(
            ['APPROVED_' + e[0] + "_" + e[1].upper() for e in X_app_agg.columns.tolist()])
        X_agg = X_agg.join(X_app_agg, how='left', on='SK_ID_CURR')

        X_agg = self.clean_column_names(X_agg)
        return X_agg
