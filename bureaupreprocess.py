"""Custom transformer to preprocess bureau data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OrdinalEncoder
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class BureauDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for bureau data."""

    def fit(self, X, y=None):
        for col in X.columns:
            if any(x in ['XNA'] for x in X[col].unique().tolist()):
                self.xna_columns.append(col)
        return self

    def transform(self, X, X_secondary=pd.DataFrame()):
        X = X.copy()
        X = self.flags_for_missing_vals(X)
        X = self.replace_value(X, self.xna_columns, ['XNA'], np.nan)
        X = self.combining_rare_categories(X, 'CREDIT_TYPE')
        X = self.feature_engineering(X)
        X_agg = self.aggregations(X, X_balance=X_secondary)
        return X_agg

    @CustomTransformer._logging
    def flags_for_missing_vals(self, X):
        """Create flag features for missing values. 
        First for missing annuity and max redit overdue data.
        Then a ratio of missing values for missing credit data."""

        X["FLAG_MISSING_AMT_ANNUITY"] = 0
        X.loc[
            X.AMT_ANNUITY.isna(), "FLAG_MISSING_AMT_ANNUITY"
        ] = 1
        X["FLAG_MISSING_CREDIT_MAX_OVERDUE"] = 0
        X.loc[
            X.AMT_CREDIT_MAX_OVERDUE.isna(), "FLAG_MISSING_CREDIT_MAX_OVERDUE"
        ] = 1
        credit_columns = [
            'DAYS_CREDIT_ENDDATE',
            'DAYS_ENDDATE_FACT',
            'AMT_CREDIT_SUM_DEBT',
            'AMT_CREDIT_SUM_LIMIT'
        ]
        X["CREDIT_INFO_MISSING_RATIO"] = X[credit_columns].isnull().mean(axis=1)

        return X

    @CustomTransformer._logging
    def feature_engineering(self, X):
        """Feature engineering"""
        X["AMT_CREDIT_DEBT_RATE"] = X['AMT_CREDIT_SUM_DEBT'] / \
            (1 + X['AMT_CREDIT_SUM'])
        X['FLAG_ACTIVE_CREDIT'] = 0
        X.loc[X['CREDIT_ACTIVE'] == 'Active', 'FLAG_ACTIVE_CREDIT'] = 1
        return X

    def aggregations(self, X, weight=0.05, X_balance=pd.DataFrame()):
        """Aggregating the bureau data of the current applicant.
        If provided conbines the bureau balance data to the bureau data."""
        cat_bal = []
        num_bal = []
        if not X_balance.empty:
            X_balance_agg = self.aggregate_balance(X_balance)
            balance_columns = [
                c for c in X_balance_agg.columns if c not in X.columns]
            cat_bal = [c for c in balance_columns if "STATUS" in c]
            num_bal = [c for c in balance_columns if c not in cat_bal]
            X = X.merge(X_balance_agg, how="left",
                        left_on="SK_ID_BUREAU", right_index=True)
            X.drop(["SK_ID_BUREAU"], axis=1, inplace=True)
        exclude = ["SK_ID_CURR"] + cat_bal

        self.num_columns = self.numerical_columns(
            X, exclude=exclude)
        num_aggre = {}
        for num in self.num_columns:
            if num == 'MONTHS_BALANCE_MIN':
                num_aggre[num] = ['min']
            elif num == 'MONTHS_BALANCE_MAX':
                num_aggre[num] = ['max']
            elif num == "CNT_CREDIT_PROLONG":
                num_aggre[num] = ["sum"]
            else:
                num_aggre[num] = ['mean', 'max']

        X, new_cat_features = self.oh_encoding(X)
        cat_features = new_cat_features + cat_bal
        cat_aggre = {}
        for cat in cat_features:
            cat_aggre[cat] = ['mean', 'sum']

        X_app = X[X['CREDIT_ACTIVE_Active'] == 1]
        X_ref = X[X['CREDIT_ACTIVE_Closed'] == 1]

        X['WEIGHT'] = np.exp(X['DAYS_CREDIT']*weight)
        X[self.num_columns + cat_features] = X[self.num_columns +
                                               cat_features].multiply(X['WEIGHT'], axis=0)
        X_agg = X.groupby('SK_ID_CURR').agg({**num_aggre, **cat_aggre})
        X_agg.columns = pd.Index(
            ['BURO_' + e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()])

        X_app_agg = X_app.groupby('SK_ID_CURR').agg({**num_aggre})
        X_app_agg.columns = pd.Index(
            ['ACTIVE_' + e[0] + "_" + e[1].upper() for e in X_app_agg.columns.tolist()])
        X_agg = X_agg.join(X_app_agg, how='left', on='SK_ID_CURR')

        X_ref_agg = X_ref.groupby('SK_ID_CURR').agg({**num_aggre})
        X_ref_agg.columns = pd.Index(
            ['CLOSED_' + e[0] + "_" + e[1].upper() for e in X_ref_agg.columns.tolist()])
        X_agg = X_agg.join(X_ref_agg, how='left', on='SK_ID_CURR')

        X_agg = self.clean_column_names(X_agg)
        return X_agg

    @CustomTransformer._logging
    def aggregate_balance(self, X):
        """Aggregate bureau balance data"""
        X, new_cats = self.oh_encoding(X, columns=['STATUS'])
        aggre = {'MONTHS_BALANCE': ['min', 'max', 'size']}
        for col in new_cats:
            aggre[col] = ['mean', 'sum']

        X_agg = X.groupby("SK_ID_BUREAU").agg(aggre)
        X_agg.columns = pd.Index(
            [e[0] + "_" + e[1].upper() for e in X_agg.columns.tolist()]
        )

        return X_agg
