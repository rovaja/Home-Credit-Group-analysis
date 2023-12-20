"""Helper module for EDA notebook to perform data cleaning and preprocessing"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def check_null_values(df: pd.DataFrame, plot=True, return_nulls=False) -> pd.DataFrame:
    """Checks for null values of the given dataset"""
    if df.empty:
        print("Dataframe is empty")
    else:
        amount_of_nulls: pd.Series = pd.isnull(df).sum()
        total_amount: int = amount_of_nulls.sum()
        print(f"Total number of null values in data: {total_amount}")
        if total_amount > 0:
            columns_with_nan = amount_of_nulls[amount_of_nulls > 0].index.tolist(
            )
            print(
                f"Number of null values per column:\n{amount_of_nulls[amount_of_nulls>0]}"
            )
            if plot:
                df_proportions = (
                    (amount_of_nulls[amount_of_nulls > 0] /
                     df.shape[0]).rename("proportion").reset_index()
                )
                height = df.shape[1]/18*3
                plt.figure(figsize=(10, height))
                cols = ['red' if x >
                        0.50 else 'steelblue' for x in df_proportions.proportion]
                ax = df_proportions.pipe(
                    (sns.barplot, "data"), x="proportion", y='index', palette=cols)
                ax.set_xlim([0, 1])
                ax.set_title('Columns with null values')
                ax.set_ylabel('Columns')
                for container in ax.containers:
                    ax.bar_label(container, padding=0,
                                 color="black", fmt="{:.3%}")
            if return_nulls:
                return df[df.isnull().any(axis=1)]


def check_duplicated_rows(df: pd.DataFrame) -> list:
    """Checks for duplicated rows of the given dataset"""
    if df.empty:
        print("Dataframe is empty")
    else:
        duplicates_mask = df.duplicated(keep='first')
        total_amount: int = duplicates_mask.sum()
        print(f"Total number of duplicated rows in data: {total_amount}")

        if total_amount > 0:
            duplicates = df[duplicates_mask == True]
            print(duplicates)
            return duplicates.index.tolist()


def find_outliers_IRQ(df: pd.DataFrame, coefficient: float = 1.5) -> list[tuple[str, float]]:
    """Finds outliers from the given numerical features using 
    the Interquartile range and given coefficient.
    Return number of outliers as percentage for each column."""
    outlier_percentages = []
    total_samples = df.shape[0]

    for column in df.select_dtypes(include=[np.number]):
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - coefficient * iqr
        upper_bound = q3 + coefficient * iqr

        column_outliers = df[(df[column] < lower_bound) |
                             (df[column] > upper_bound)][column]

        percentage = column_outliers.count() / total_samples
        outlier_percentages.append((column, percentage))

    return outlier_percentages


def drop_rows_without_values(X: pd.DataFrame, column: str, value_list: list[str], y: pd.Series = pd.Series()) -> tuple[pd.DataFrame, pd.Series]:
    """Remove data rows that do not have values listed in a given column.
    Removes the rows from both faeture matrix and target feature vector"""
    indx = X[~X[column].isin(value_list)].index
    if y.empty:
        return X.drop(indx)
    else:
        return X.drop(indx), y.drop(indx)


def drop_rows_outside_range(X: pd.DataFrame, column: str, upper_boundary: float, lower_boundary: float, y: pd.Series = pd.Series()) -> tuple[pd.DataFrame, pd.Series]:
    """Remove data rows that do not have values listed in a given column.
    Removes the rows from both faeture matrix and target feature vector"""
    indx = X.loc[(X[column] < lower_boundary) |
                 (X[column] > upper_boundary)].index
    print(
        f"Number of data instances removed: {X.iloc[indx,:].shape[0]}"
    )
    if y.empty:
        return X.drop(indx)
    else:
        return X.drop(indx), y.drop(indx)
