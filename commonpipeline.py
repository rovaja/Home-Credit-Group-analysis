"""Helper module for EDA notebook to create preprocessing pipelines for data cleaning"""
import pandas as pd
import datetime as dt


def logging(f):
    """Create logging for each function.
    Based on Vincent D Warmerdam Untitled12 lecture 2019."""
    def wrapper(data, *args, **kwargs):
        start = dt.datetime.now()
        funtion_result = f(data, *args, **kwargs)
        stop = dt.datetime.now()
        print(
            f"{f.__name__} spent {stop - start} seconds, shape = {funtion_result.shape}")
        return funtion_result
    return wrapper


@logging
def start_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Create a copy and start cleaining pipeline."""
    return df.copy()


@logging
def select_columns(df: pd.DataFrame, columns_to_select: list[str] = []) -> pd.DataFrame:
    """Filter dataset with the given column list."""
    return df[columns_to_select]


@logging
def remove_columns(df: pd.DataFrame, columns_to_drop: list[str] = []) -> pd.DataFrame:
    """Remove given columns in the pipeline."""
    return df.drop(columns=columns_to_drop)


@logging
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated rows"""
    return df.drop_duplicates(keep='first')


@logging
def clean_column_names(df: pd.DataFrame, new_names: dict[str, str] = {}) -> pd.DataFrame:
    """Remove white spaces from columns or sets new names with given dictionary"""
    if new_names:
        return df.rename(columns=new_names)
    else:
        df.columns = [col.strip() for col in df.columns]
        df.columns = [col.replace(' ', '_') for col in df.columns]
        return df
