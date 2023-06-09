from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import OneHotEncoder


# this is primarily used b/c round 326 had too many stocks,
# so we need to filter out the unnecessary ids here just in case
# it's also just convenient way to ensure everything is sorted/matching
def filter_sort_index(
    s1: Union[pd.DataFrame, pd.Series],
    s2: Union[pd.DataFrame, pd.Series],
    max_filtered_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ids = s1.dropna().index.intersection(s2.dropna().index)
    # ensure we didn't filter too many ids
    assert len(ids) / len(s1) >= (1 - max_filtered_ratio)
    assert len(ids) / len(s2) >= (1 - max_filtered_ratio)
    return s1.loc[ids].sort_index(), s2.loc[ids].sort_index()


def rank(df: pd.DataFrame, method: str = 'average') -> pd.DataFrame:
    """Percentile rank each column of a pandas DataFrame, centering values around 0.5

    Arguments:
        df: pd.DataFrame - the data to rank
        method: str - the pandas ranking method to use, options:
            'average' (default) - keeps ties
            'first' - breaks ties by index

    Returns:
        pd.DataFrame - the ranked DataFrame
    """
    assert np.array_equal(df.index.sort_values(), df.index), "unsorted index found"
    return df.apply(
        lambda series: (series.rank(method=method).values - 0.5) / series.count()
    )


def tie_broken_rank(df: pd.DataFrame) -> pd.DataFrame:
    # rank columns, breaking ties by index
    return rank(df, "first")


def tie_kept_rank(df: pd.DataFrame) -> pd.DataFrame:
    # rank columns, but keep ties
    return rank(df, "average")


def min_max_normalize(s: pd.Series) -> pd.Series:
    # scale a series to be between 0 and 1
    return (s - s.min()) / (s.max() - s.min())


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


def tie_broken_rank_correlation(
    live_targets: pd.Series, predictions: pd.Series
) -> float:
    # percentile rank the predictions and get the correlation with live_targets
    ranked_predictions = tie_broken_rank(predictions.to_frame())[predictions.name]
    return correlation(live_targets, ranked_predictions)


def spearman_correlation(live_targets: pd.Series, predictions: pd.Series) -> float:
    validate_indices(live_targets, predictions)
    # calculate corr
    return live_targets.corr(predictions, method="spearman")


def pearson_correlation(live_targets: pd.Series, predictions: pd.Series) -> float:
    validate_indices(live_targets, predictions)
    # calculate corr
    return live_targets.corr(predictions, method="pearson")


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
    result = np.sign(df) * np.abs(df) ** p
    assert ((result.std() == 0) | (result.corrwith(df) >= 0.9)).all()
    return result


def gaussian(df: pd.DataFrame) -> pd.DataFrame:
    """Gaussianize each column of a pandas DataFrame using a normal percent point func

    Arguments:
        df: pd.DataFrame - the data to gaussianize

    Returns:
        pd.DataFrame - the gaussianized data
    """
    assert np.array_equal(df.index.sort_values(), df.index)
    return df.apply(lambda series: stats.norm.ppf(series))


def neutralize(
    df: pd.DataFrame, neutralizers: np.ndarray, proportion: float = 1.0
) -> pd.DataFrame:
    """Neutralize each column of a given DataFrame by each feature in a given
    neutralizers DataFrame.

    Arguments:
        df: pd.DataFrame - the data with columns to neutralize
        neutralizers: pd.DataFrame - the neutralizer data with features as columns
        proportion: float - the degree to which neutralization occurs

    Returns:
        pd.DataFrame - the neutralized data

    """
    assert not neutralizers.isna().any().any(), "Neutralizers contain NaNs"
    assert len(df.index) == len(neutralizers.index), "Indices don't match"
    assert (df.index == neutralizers.index).all(), "Indices don't match"
    df[df.columns[df.std() == 0]] = np.nan
    df_arr = df.values
    neutralizer_arr = neutralizers.values
    inverse_neutralizers = np.linalg.pinv(neutralizer_arr, rcond=1e-6)
    adjustments = proportion * neutralizer_arr.dot(inverse_neutralizers.dot(df_arr))
    neutral = df_arr - adjustments
    neutral /= np.std(neutral, axis=0)
    return pd.DataFrame(neutral, index=df.index, columns=df.columns)


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
            columns=encoder.get_feature_names(),
            index=df.index,
        )
        df = df.join(one_hot).drop(columns=col)
    return df


def tie_kept_rank__gaussianize__pow_1_5(df: pd.DataFrame) -> pd.DataFrame:
    """Perform the 3 functions in order on the given pandas DataFrame.
    Will tie-kept rank then gaussianize then exponentiate to the 1.5 power.

    Arguments:
        df: pd.DataFrame - the data to transform

    Returns:
        pd.DataFrame - the resulting data after applying the 3 functions

    """
    return power(gaussian(tie_kept_rank(df)), 1.5)


def tie_kept_rank__gaussianize__neutralize(
    df: pd.DataFrame, neutralizers: pd.DataFrame
) -> pd.DataFrame:
    """Perform the 3 functions in order on the given pandas DataFrame.
    Will tie-kept rank then gaussianize then neutralize the df to the neutralizers.

    Arguments:
        df: pd.DataFrame - the data to transform

    Returns:
        pd.DataFrame - the resulting data after applying the 3 functions
    """
    return neutralize(gaussian(tie_kept_rank(df)), neutralizers)


def numerai_corr(predictions: pd.DataFrame, targets: pd.Series) -> pd.Series:
    """Recenter the target on 0, filter and sort indices, apply tie_kept_rank__gaussianize__pow_1_5
    to the predictions, raise the targets to the 1.5 power, then calculate the
    pearson correlation between the predictions and targets.

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        targets: pd.Series - the live targets to evaluate against

    Returns:
        pd.Series - the resulting correlation scores for each column in predictions

    """
    targets -= targets.mean()
    targets, predictions = filter_sort_index(targets, predictions)
    predictions = tie_kept_rank__gaussianize__pow_1_5(predictions)
    targets = power(targets.to_frame(), 1.5)[targets.name]
    scores = predictions.apply(lambda sub: pearson_correlation(targets, sub))
    return scores
