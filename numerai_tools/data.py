from typing import List, Union, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder  # type: ignore

from numerai_tools.scoring import tie_kept_rank

DEFAULT_BINS = (0.0, 0.25, 0.5, 0.75, 1.0)
DEFAULT_QUANTILES = (0.05, 0.25, 0.75, 0.95)


def one_hot_encode(
    df: pd.DataFrame, columns: List[str], dtype: type = np.float64
) -> pd.DataFrame:
    """One-hot encodes specified columns in a pandas dataframe.
    Each column i should have x_i discrete values (eg. categories, bucket values, etc.)
    and will be converted to x_i columns that each have 0s for rows that don't have
    the associated value and 1s for rows that do have that value.

    Arguments:
        df: pd.DataFrame - the data with columns to one-hot encode
        columns: List[str] - list of columns names to replace w/ one-hot encoding
        dtype: type = np.float64 - the target datatype for the resulting columns

    Returns:
        pd.DataFrame - original data, but specified cols replaced w/ one-hot encoding
    """
    for col in columns:
        encoder = OneHotEncoder(dtype=dtype)
        one_hot = encoder.fit_transform(df[[col]])
        one_hot = pd.DataFrame(
            one_hot.toarray(),
            columns=encoder.get_feature_names_out(),
            index=df.index,
        )
        df = df.join(one_hot).drop(columns=col)
    return df


def balanced_rank_transform(
    df: pd.DataFrame,
    cols: List[str],
    rank_group: Optional[str] = None,
    rank_filter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Perform a balanced rank transformation on specified columns of a DataFrame,
    optionally within groups and with a filter.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing the data to be ranked.
    cols : list of str
        List of column names to apply the rank transformation to.
    rank_group : str
        Column name to group by before ranking.
    rank_filter : str, optional
        Column name to filter rows before ranking. Only rows where this column is True
        will be ranked. If None, no filtering is applied.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as the input, containing the ranked columns.
    """
    if rank_filter is not None:
        df = df.loc[df[rank_filter]]
    else:
        df = df
    if rank_group is not None:
        df = df.groupby(rank_group, group_keys=False).apply(
            lambda d: tie_kept_rank(d[cols])
        )
    else:
        df = tie_kept_rank(df[cols])
    return df[cols]


def quantile_bin(
    data: Union[pd.Series, pd.DataFrame],
    bins: tuple[float, ...] = DEFAULT_BINS,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
) -> pd.DataFrame:
    """
    Bin a Series or DataFrame into discrete quantile-based bins.
    Handles identical-value columns by assigning all values to the lowest bin.

    Parameters
    ----------
    data : pd.Series or pd.DataFrame
        Data to bin.
    bins : list of float
        Values to assign to each bin.
    quantiles : list of float
        Quantile thresholds to use for binning (len = number of bins - 1)

    Returns
    -------
    pd.DataFrame
        Binned values, same shape as input.
    """
    assert len(bins), "Invalid bins! Must not be empty."
    assert len(quantiles), "Invalid quantiles! Must not be empty."
    assert len(quantiles) == (
        len(bins) - 1
    ), "Invalid quantiles! Length must be 1 less than bins."

    if isinstance(data, pd.Series):
        data = data.to_frame(name="value")

    binned = data.copy()
    for col in binned.columns:
        s = binned[col].astype(float)

        # handle all-identical values
        if s.nunique() <= 1:
            binned[col] = 0.0
            continue

        # calculate quantile thresholds
        q = s.quantile(quantiles)

        # assign bins according to quantiles using pd.cut for mutually exclusive bins
        bin_edges = [-np.inf] + [q[q_idx] for q_idx in quantiles] + [np.inf]
        s = pd.cut(s, bins=bin_edges, labels=bins, include_lowest=True).astype(float)

        binned[col] = s.astype(float)

    return binned
