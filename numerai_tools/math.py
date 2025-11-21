from typing import Optional, cast, Literal

import numpy as np
import pandas as pd
from scipy import stats

# leaving this here for backwards compatibility
from numerai_tools.typing import S1

from numerai_tools.indexing import (
    filter_sort_index,
    filter_sort_top_bottom_concat,
)


RANK_METHOD_TYPE = Literal["average", "min", "max", "first", "dense"]


def rank_series(s: pd.Series, method: RANK_METHOD_TYPE = "average") -> pd.Series:
    """Percentile rank a pandas Series, centering values around 0.5.

    Arguments:
        s: pd.Series - the data to rank
        method: str - the pandas ranking method to use, options:
            'average' (default) - keeps ties
            'first' - breaks ties by index

    Returns:
        pd.Series - the ranked Series
    """
    assert np.array_equal(s.index.sort_values(), s.index), "unsorted index found"
    # Ensure denominator is at least 1 to avoid division by zero
    denom = max(int(s.count()), 1)
    return (s.rank(method=method) - 0.5) / denom


def rank(s: S1, method: RANK_METHOD_TYPE = "average") -> S1:
    """Percentile rank each columns or series, centering values around 0.5

    Arguments:
        s: pd.DataFrame | pd.Series - the data to rank
        method: str - the pandas ranking method to use, options:
            'average' (default) - keeps ties
            'first' - breaks ties by index

    Returns:
        pd.DataFrame | pd.Series - the ranked input data
    """
    if isinstance(s, pd.Series):
        return cast(S1, rank_series(s, method))
    else:
        return s.apply(lambda series: rank(series, method=method))


def tie_broken_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank columns, breaking ties by index."""
    return rank(df, "first")


def tie_kept_rank(s: S1) -> S1:
    """Rank columns, but keep ties."""
    return cast(S1, rank(s, "average"))


def min_max_normalize(s: pd.Series) -> pd.Series:
    """Scale a series to be between 0 and 1."""
    return (s - s.min()) / (s.max() - s.min())


def variance_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale a df such that all columns have std == 1."""
    return df / np.std(df, axis=0)


def weight_normalize(s: S1) -> S1:
    """Scale a input such that all columns have absolute value sum == 1."""
    return cast(S1, s / s.abs().sum(axis=0))


def center(s: S1) -> S1:
    """Shift the input such that all columns have mean == 0."""
    return cast(S1, s - s.mean())


def standardize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale a df such that all columns have mean == 0 and std == 1."""
    return variance_normalize(center(df))


def validate_indices(live_targets: pd.Series, predictions: pd.Series) -> None:
    # ensure the ids are equivalent and sorted
    assert np.array_equal(predictions.index, live_targets.index.sort_values())
    assert np.array_equal(live_targets.index, live_targets.index.sort_values())
    assert np.array_equal(predictions.index, predictions.index.sort_values())
    # ensure no nans
    assert not predictions.isna().any()
    assert not live_targets.isna().any()


def correlation(live_targets: pd.Series, predictions: pd.Series) -> float:
    validate_indices(live_targets, predictions)
    # calculate correlation coefficient
    return np.corrcoef(live_targets, predictions)[0, 1]


def tie_broken_rank_correlation(target: pd.Series, predictions: pd.Series) -> float:
    # percentile rank the predictions and get the correlation with the target
    ranked_predictions = tie_broken_rank(predictions.to_frame())[predictions.name]
    return correlation(target, ranked_predictions)


def spearman_correlation(target: pd.Series, predictions: pd.Series) -> float:
    validate_indices(target, predictions)
    return target.corr(predictions, method="spearman")


def pearson_correlation(
    target: pd.Series, predictions: pd.Series, top_bottom: Optional[int] = None
) -> float:
    if top_bottom is not None and top_bottom > 0:
        predictions = filter_sort_top_bottom_concat(predictions, top_bottom)
        target, predictions = filter_sort_index(
            target, predictions, (1 - top_bottom / len(target))
        )
    validate_indices(target, predictions)
    return target.corr(predictions, method="pearson")


def sharpe_ratio(s: pd.Series) -> float:
    # calculate the sharpe ratio of a series
    return np.mean(s) / np.std(s)


def gaussian(df: pd.DataFrame) -> pd.DataFrame:
    """Gaussianize each column of a pandas DataFrame using a normal percent point func.
    Effectively scales each column such that mean == 0 and std == 1.

    Arguments:
        df: pd.DataFrame - the data to gaussianize

    Returns:
        pd.DataFrame - the gaussianized data
    """
    assert np.array_equal(df.index.sort_values(), df.index)
    return df.apply(lambda series: cast(np.ndarray, stats.norm.ppf(series)))


def power(df: pd.DataFrame, p: float) -> pd.DataFrame:
    """Raise given predictions series to the given power.

    Arguments:
        df: pd.DataFrame - the data to raise to the given power
        p: float - the power to which we exponentiate the data

    Returns:
        pd.DataFrame - the predictions raised to the given power,
            each column should be at least 90% correlated with the original data
    """
    assert not df.isna().any().any(), "Data contains NaNs"
    assert np.array_equal(df.index.sort_values(), df.index), "Index is not sorted"
    result = cast(pd.DataFrame, np.sign(df) * np.abs(df) ** p)
    assert ((result.std() == 0) | (result.corrwith(df) >= 0.9)).all()
    return result


def tie_kept_rank__gaussianize__pow_1_5(df: pd.DataFrame) -> pd.DataFrame:
    """Perform the 3 functions in order on the given pandas DataFrame.
    Will tie-kept rank then gaussianize then exponentiate to the 1.5 power.

    Arguments:
        df: pd.DataFrame - the data to transform

    Returns:
        pd.DataFrame - the resulting data after applying the 3 functions
    """
    return power(gaussian(tie_kept_rank(df)), 1.5)


def tie_kept_rank__gaussianize__neutralize__variance_normalize(
    df: pd.DataFrame, neutralizers: pd.DataFrame
) -> pd.DataFrame:
    """Perform the 4 functions in order on the given pandas DataFrame.
    1. tie-kept rank each column
    2. gaussianize each column
    3. neutralize each column to the neutralizers
    4. variance normalize each column

    Arguments:
        df: pd.DataFrame - the data to transform

    Returns:
        pd.DataFrame - the resulting data after applying the 3 functions
    """
    return variance_normalize(neutralize(gaussian(tie_kept_rank(df)), neutralizers))


def orthogonalize(v: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Orthogonalizes v with respect to u by projecting v onto u,
    then subtracting that projection from v.

    This will reach the same result as the neutralize
    function when v and u are centered single column vectors,
    but this is much faster.

    Arguments:
        v: np.ndarray - the vector to orthogonalize
        u: np.ndarray - the vector orthogonalize v

    Returns:
        np.ndarray - the orthogonalized vector v
    """
    return v - np.outer(u, (v.T @ u) / (u.T @ u))


def stake_weight(
    predictions: pd.DataFrame,
    stakes: pd.Series,
) -> pd.Series:
    """Create a stake-weighted meta model from the given predictions and stakes.

    Arguments:
        predictions: pd.DataFrame - the predictions to weight
        stakes: pd.Series - the stakes to use as weights

    Returns:
        pd.Series - the stake-weighted meta model
    """
    return (predictions[stakes.index] * stakes).sum(axis=1) / stakes.sum()


def neutralize(
    df: pd.DataFrame,
    neutralizers: pd.DataFrame,
    proportion: float = 1.0,
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame. Neutralization uses least-squares regression to
    find the orthogonal projection of each column onto the neutralizers, then
    subtracts the result from the original predictions.

    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data
    """
    assert not df.isna().any().any(), "Data contains NaNs"
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    neutralizer_arr = np.hstack(
        # add a column of 1s to the neutralizer array in case neutralizer_arr is a single column
        (neutralizer_arr, np.array([1] * len(neutralizer_arr)).reshape(-1, 1))
    )
    least_squares = np.linalg.lstsq(neutralizer_arr, df_arr, rcond=1e-6)[0]
    adjustments = proportion * neutralizer_arr.dot(least_squares)
    neutral = df_arr - adjustments
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)
