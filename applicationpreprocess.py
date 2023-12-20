"""Custom transformer to preprocess application data in 
Home Credit data for modelling purposes."""
from sklearn.base import BaseEstimator, TransformerMixin
from customtransformer import CustomTransformer
import numpy as np
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class AppDataEncoder(BaseEstimator, TransformerMixin, CustomTransformer):
    """Encodes for application data."""

    def fit(self, X, y=None):
        for col in X.columns:
            if any(x in ['XNA'] for x in X[col].unique().tolist()):
                self.xna_columns.append(col)
        return self

    def transform(self, X):
        X = X.copy()
        X = self.flags_for_missing_vals(X)
        X = self.replace_value(X, ["DAYS_EMPLOYED"], [365243], np.nan)
        X = self.replace_value(X, self.xna_columns, ['XNA'], np.nan)
        X = self.replace_value(X, ["NAME_EDUCATION_TYPE"], [
                               'Academic degree'], "Higher education")
        X = self.combining_org_type(X)
        X = self.combining_occ_type(X)
        X = self.combining_inc_type(X)
        X_agg = self.aggregations(X)
        return X_agg

    @CustomTransformer._logging
    def flags_for_missing_vals(self, X):
        """Create flag features for missing values. 
        First for missing housing information features,
        and then for missing credit information."""
        housing_columns = X.columns[
            X.columns.str.contains("MEDI|MODE|AVG")
        ].tolist()
        X["HOUSING_INFO_MISSING_RATIO"] = X[housing_columns].isnull().mean(axis=1)

        credit_columns = [
            "AMT_REQ_CREDIT_BUREAU_HOUR",
            "AMT_REQ_CREDIT_BUREAU_DAY",
            "AMT_REQ_CREDIT_BUREAU_WEEK",
            "AMT_REQ_CREDIT_BUREAU_MON",
            "AMT_REQ_CREDIT_BUREAU_QRT",
            "AMT_REQ_CREDIT_BUREAU_YEAR",
        ]
        X["FLAG_CREDIT_BUREAU_MISSING"] = 0
        X.loc[X[credit_columns].isnull().any(axis=1),
              "FLAG_CREDIT_BUREAU_MISSING"] = 1

        return X

    @CustomTransformer._logging
    def combining_org_type(self, X):
        """Combine organisation type classes"""
        new_label_list = [
            "Business Entity",
            "Industry",
            "Trade",
            "Transport",
            "Public",
            "Finance",
            "House",
            "Services",
        ]
        old_label_list = [
            [
                "Advertising",
                "Business Entity Type 1",
                "Business Entity Type 2",
                "Business Entity Type 3",
            ],
            X.loc[
                X.ORGANIZATION_TYPE.str.contains("Industry", na=False), "ORGANIZATION_TYPE",].unique().tolist(),
            X.loc[X.ORGANIZATION_TYPE.str.contains(
                "Trade", na=False), "ORGANIZATION_TYPE",].unique().tolist(),
            X.loc[X.ORGANIZATION_TYPE.str.contains(
                "Transport", na=False), "ORGANIZATION_TYPE",].unique().tolist(),
            [
                "School",
                "University",
                "Kindergarten",
                "Emergency",
                "Government",
                "Medicine",
                "Military",
                "Police",
                "Postal",
                "Religion",
                "Security Ministries",
                "Electricity",
                "Telecom",
            ],
            [
                "Bank",
                "Insurance",
                "Legal Services",
            ],
            [
                "Realtor",
                "Housing",
            ],
            [
                "Hotel",
                "Restaurant",
                "Cleaning",
                "Culture",
                "Services",
                "Security",
                "Mobile",
                "Agriculture",
            ],
        ]
        for new_label, label_list in zip(new_label_list, old_label_list):
            X = self.replace_value(
                X, ['ORGANIZATION_TYPE'], label_list, new_label)
        return X

    @CustomTransformer._logging
    def combining_occ_type(self, X):
        """Combine organisation type classes"""
        new_label_list = ["administration staff",
                          "low-skill Laborers", "sales staff"]
        old_label_list = [
            ["IT staff", "HR staff", "Secretaries"],
            ["Low-skill Laborers", "Waiters/barmen staff", "Cleaning staff"],
            ["Realty agents", "Sales staff"],
        ]
        for new_label, label_list in zip(new_label_list, old_label_list):
            X = self.replace_value(
                X, ['OCCUPATION_TYPE'], label_list, new_label)
        return X

    @CustomTransformer._logging
    def combining_inc_type(self, X):
        """Combine organisation type classes"""
        new_label_list = ["beneficiary", "commercial associate"]
        old_label_list = [
            ["Maternity leave", "Student", "Unemployed", "Pensioner"],
            ["Businessman", "Commercial associate"],
        ]
        for new_label, label_list in zip(new_label_list, old_label_list):
            X = self.replace_value(
                X, ['NAME_INCOME_TYPE'], label_list, new_label)
        return X

    @CustomTransformer._logging
    def encode(self, X):
        """Run encoders for given categorical columns."""
        X = self.ord_encoding(X)
        X, new_cat_features = self.oh_encoding(X)
        X = self.clean_column_names(X)
        return X
