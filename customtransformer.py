"""General transformer and encoder class"""
from sklearn.base import BaseEstimator, TransformerMixin
from collections import defaultdict
import numpy as np
import re
import pandas as pd
import sklearn
import datetime as dt
sklearn.set_config(transform_output="pandas")


class CustomTransformer():
    """Custom encoders and transformers"""

    def _logging(f):
        """Create logging for each method."""

        def wrapper(self, data, *args, **kwargs):
            start = dt.datetime.now()
            funtion_result = f(self, data, *args, **kwargs)
            stop = dt.datetime.now()
            print(
                f"{f.__name__} time spent {stop - start} seconds, shape = {funtion_result.shape}")
            return funtion_result
        return wrapper

    def __init__(self, oh_columns: list[str] = [], ord_columns: dict[str, dict[str, int]] = {}, num_columns: list[str] = []):
        self.oh_columns = oh_columns
        self.ord_columns = ord_columns
        self.num_columns = num_columns
        self.ord_maps = {}
        self.oh_new_columns = []
        self.ord_new_columns = []
        self.xna_columns = []

    @_logging
    def replace_value(
            self,
            X,
            columns: list[str],
            vals_to_change: list[str | float | int],
            new_value: str | float | int):
        """Replacing given values with the new value from predefined columns."""
        for col in columns:
            for vals in vals_to_change:
                X[col].replace(vals, new_value, inplace=True)
        return X

    @_logging
    def combining_rare_categories(self, X, column: str, threshold: float = 0.001):
        """Combine loan purposes type classes.
        All classes smaller than the threshold value are combined into Others."""
        categories = X[column].value_counts(
            normalize=True).reset_index()
        rare_categories = categories[categories['proportion']
                                     < threshold][column].tolist()
        X[column] = X[column].apply(
            lambda x: 'Other' if x in rare_categories else x)
        return X

    def numerical_columns(self, X, exclude=[]):
        """Define numerical columns"""
        return X.columns[~X.columns.isin(self.oh_columns + list(self.ord_columns.keys()) + exclude)].tolist()

    def oh_encoding(self, X, columns=[], nan_as_category=False):
        """OneHotEncoding for categorical features."""
        if not columns:
            columns = self.oh_columns
        X_wdummies = pd.get_dummies(X, columns=columns,
                                    dummy_na=nan_as_category, dtype='int32')
        new_columns = [c for c in X_wdummies.columns if c not in X.columns]

        self.oh_new_columns += new_columns
        X = X_wdummies
        return X, new_columns

    def ord_encoding(self, X):
        """Ordinal encoding of categorical features.
        If order is provided as list then it is used."""

        for col in self.ord_columns.keys():
            X[col] = X[col].apply(lambda x: x.lower()
                                  if isinstance(x, str) else x)
            map_li: list = self.ord_columns[col]
            if not map_li:
                map_li = X[col].unique().tolist()
                map_li = [x for x in map_li if not (
                    isinstance(x, float) and np.isnan(x))]
                try:
                    map_li.sort()
                except TypeError:
                    print("WARNING: TypeError. Can not sort list.")
            map_di: dict = {x: i for i, x in enumerate(map_li)}
            if len(map_li) == 2:
                col_name = col+'_' + str(map_li[-1])
            else:
                col_name = col
            self.ord_maps[col_name] = map_di
            X[col] = X[col].map(map_di)
            X.rename(columns={col: col_name}, inplace=True)
        self.ord_new_columns += list(self.ord_maps.keys())
        return X

    def remove_special_characters(self, input_string):
        """Remove special characters from the string and replace them with '_'.
        Returns the modified string."""
        pattern = r'[^a-zA-Z0-9_]'
        result_string = re.sub(pattern, '_', input_string)
        return result_string

    def clean_column_names(self, X):
        """Removes special characters from column names."""
        rename_mapping = {}
        for c in X.columns:
            r = self.remove_special_characters(c)
            rename_mapping[c] = r
        return X.rename(columns=rename_mapping)

    @_logging
    def flag_missing_values_aggregated(self, X_agg):
        """Create missing value ratios for aggregated features per original table."""
        list_groups: dict[str, list[str]] = defaultdict(list)

        for c in X_agg.columns:
            if 'PREV_' in c:
                list_groups['PREV_'].append(c)
            elif 'CC_' in c:
                list_groups['CC_'].append(c)
            elif 'APPROVED_' in c:
                list_groups['APPROVED_'].append(c)
            elif 'INSTAL_' in c:
                list_groups['INSTAL_'].append(c)
            elif 'POS_' in c:
                list_groups['POS_'].append(c)
            elif 'BURO_' in c:
                list_groups['BURO_'].append(c)
            elif 'ACTIVE_' in c:
                list_groups['ACTIVE_'].append(c)
            elif 'CLOSED_' in c:
                list_groups['CLOSED_'].append(c)

        for group, columns in list_groups.items():
            X_agg[group+'_TOTAL_MISSING_RATIO'] = X_agg[columns].isna().mean(axis=1)

        return X_agg
