"""Helper module for EDA notebook to perform data visualizing and summary statistics"""
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, kruskal, mannwhitneyu
import scikit_posthocs as sp
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


def dataframe_info(df, topN=10) -> pd.DataFrame:
    """Return summary information of given dataframe. This includes 
    shape of dataframe, 
    for each column:
        list of N most frequent values,
        summary statistics,
        and number and ratio of null values."""
    number_of_rows = df.shape[0]
    print(f'Shape: {df.shape}')

    df_summary = df.dtypes.to_frame()
    df_summary.columns = ['DataType']
    df_summary['#Nulls'] = df.isnull().sum()
    df_summary['%Nulls'] = df.isnull().sum()/number_of_rows
    df_summary['#Uniques'] = df.nunique()

    df_summary['Min'] = df.min(numeric_only=True)
    df_summary['Mean'] = df.mean(numeric_only=True)
    df_summary['Median'] = df.median(numeric_only=True)
    df_summary['Max'] = df.max(numeric_only=True)
    df_summary['Std'] = df.std(numeric_only=True)

    df_summary[f'top{topN} value'] = 0
    df_summary[f'top{topN} count'] = 0
    df_summary[f'top{topN} ratio'] = 0
    for c in df_summary.index:
        vc = df[c].value_counts().head(topN)
        val = list(vc.index)
        cnt = list(vc.values)
        ratio = list((vc.values / number_of_rows).round(2))
        df_summary.loc[c, f'top{topN} values'] = str(val)
        df_summary.loc[c, f'count of top{topN} values'] = str(cnt)
        df_summary.loc[c, f'proportion of top{topN} values'] = str(ratio)

    return df_summary


def absolute_high_pass_filter(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Filters datapoints according the absolute threshold."""
    passed = set()
    for (r, c) in combinations(df.columns, 2):
        if (abs(df.loc[r, c]) >= threshold):
            passed.add(r)
            passed.add(c)
    passed = sorted(passed)
    return df.loc[passed, passed]


def categorical_correlations(df: pd.DataFrame, columns: list[str]
                             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conduct Chi-square test of independence and Cramér's V test for
    each pair of categorical features of the given data"""

    correlation_matrix = pd.DataFrame(index=columns, columns=columns)
    p_matrix = pd.DataFrame(index=columns, columns=columns)

    for col1 in columns:
        for col2 in columns:
            if col1 == col2:
                # Set diagonal values
                correlation_matrix.loc[col1, col2] = 1.0
                p_matrix.loc[col1, col2] = 0.0
            else:
                # Conduct the chi-square test
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2, p, dof, expected = chi2_contingency(contingency_table)
                if np.min(expected) <= 5:
                    print(contingency_table)
                    print(expected)
                    print(f"""One of the {col1} and {col2} combinations has
                          expected value less than 5.""")

                # Calculate Cramér's V
                n = contingency_table.sum().sum()
                phi2 = chi2 / n
                r, k = contingency_table.shape
                phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
                rcorr = r - ((r - 1) ** 2) / (n - 1)
                kcorr = k - ((k - 1) ** 2) / (n - 1)
                correlation_matrix.loc[col1, col2] = np.sqrt(
                    phi2corr / min((kcorr - 1), (rcorr - 1))
                )
                p_matrix.loc[col1, col2] = p

    correlation_matrix = correlation_matrix.astype("float")
    p_matrix = p_matrix.astype("float")
    return correlation_matrix, p_matrix


def categorical_numerical_correlation(df: pd.DataFrame,
                                      categorical_columns: list[str],
                                      numerical_columns: list[str]
                                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Conduct Kruskal-Wallis test for each pair of categorical features of the given data"""

    effect_size_matrix = pd.DataFrame(
        index=categorical_columns, columns=numerical_columns)
    p_matrix = pd.DataFrame(index=categorical_columns,
                            columns=numerical_columns)

    if df.empty:
        raise ValueError("Input DataFrame 'df' is empty.")
    for col in categorical_columns + numerical_columns:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' does not exist in the DataFrame.")

    for cat_col in categorical_columns:
        for num_col in numerical_columns:
            grouped_data = []
            for category in df[cat_col].unique():
                values = df[df[cat_col] == category][num_col]
                grouped_data.append(values)

            max_length = max(len(sublist) for sublist in grouped_data)
            min_length = min(len(sublist) for sublist in grouped_data)
            # Checking if the sizes of subgroupds (num_cols) differ too much from each other.
            if max_length / min_length > 100 or min_length < 50:
                print("The size of the largest and smaller group differs too much.")
                print(f"{cat_col} vs {num_col}")
                print(df[cat_col].unique())
                print([len(sublist) for sublist in grouped_data])

            # Perform the Kruskal-Wallis test
            try:
                statistic, p = kruskal(*grouped_data)
                p_matrix.loc[cat_col, num_col] = p
            except ValueError as e:
                print(f"An error occurred: {e}")
                print(f"Categorical: {cat_col}")
                print(f"Numerical: {num_col}")
                p = np.nan
                p_matrix.loc[cat_col, num_col] = p

            if p < 0.05:
                # Calculate Epsilon-squared (Effect size)
                # Effect size range between 0 to 1.
                n_total = sum(len(sublist) for sublist in grouped_data)
                k = len(grouped_data)
                epsilon_squared = (statistic - (k - 1)) / (n_total - k)
                effect_size_matrix.loc[cat_col, num_col] = epsilon_squared
            else:
                effect_size_matrix.loc[cat_col, num_col] = np.nan

    effect_size_matrix = effect_size_matrix.astype("float")
    p_matrix = p_matrix.astype("float")
    return effect_size_matrix, p_matrix


def compare_means_mannwhitneyu(ds1: pd.Series, ds2: pd.Series, significance_level: float = 0.05, alternative: str = 'two-sided') -> tuple[float, float]:
    """Conduct non-parameteric Mann-Whitney U test for given datasets.
    Define alternative hypothesis with a parameter.

    Then the following alternative hypotheses are available:
    - two-sided: the distributions are not equal
    - less, the distribution underlying ds1 is stochastically less than the distribution underlying ds2
    - greater: the distribution underlying ds2 is stochastically greater than the distribution underlying ds2

    Assumptions:
    1. Independent samples
    2. Similar shape of distributions, roughly similarly distributed.
    3. Random sampling"""

    print(f"Mean value of the group 1: {ds1.mean():.3f}")
    print(f"Mean value of the group 2: {ds2.mean():.3f}")
    print(f"Difference of mean values: {ds2.mean() - ds1.mean():.3f}")

    statistic, pvalue = mannwhitneyu(ds1, ds2, alternative=alternative)

    print(f"The test results: statistic {statistic:.5f}, p-value {pvalue:.5e}")
    if pvalue < significance_level:
        print(f"""The p-value is smaller than the significant level {significance_level:.4f},
              and hence, the null-hypothesis can be rejected.""")
    else:
        print(f"""The p-value is not smaller than the significant level {significance_level:.2f},
              and hence, the null-hypothesis can not be rejected.""")
    return statistic, pvalue
