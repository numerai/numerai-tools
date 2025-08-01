from typing import List, Tuple, Union, Optional, TypeVar

import numpy as np
import pandas as pd  # type: ignore
from scipy import stats  # type: ignore
from sklearn.preprocessing import OneHotEncoder  # type: ignore


# sometimes when we match up the target/prediction indices,
# changes in stock universe causes some stocks to enter / leave,
# this ensures we don't filter too much
DEFAULT_MAX_FILTERED_INDEX_RATIO = 0.2


S1 = TypeVar("S1", bound=Union[pd.DataFrame, pd.Series])
S2 = TypeVar("S2", bound=Union[pd.DataFrame, pd.Series])


def filter_sort_index(
    s1: S1, s2: S2, max_filtered_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO
) -> Tuple[S1, S2]:
    """Filters the indices of the given series to match each other,
    then sorts the indices, then checks that we didn't filter too many indices
    before returning the filtered and sorted series.

    Arguments:
        s1: Union[pd.DataFrame, pd.Series] - the first dataset to filter and sort
        s2: Union[pd.DataFrame, pd.Series] - the second dataset to filter and sort

    Returns:
        Tuple[
            Union[pd.DataFrame, pd.Series],
            Union[pd.DataFrame, pd.Series],
        ] - the filtered and sorted datasets
    """
    ids = s1.dropna().index.intersection(s2.dropna().index)
    # ensure we didn't filter too many ids
    assert len(ids) / len(s1) >= (1 - max_filtered_ratio), (
        "s1 does not have enough overlapping ids with s2,"
        f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
    )
    assert len(ids) / len(s2) >= (1 - max_filtered_ratio), (
        "s2 does not have enough overlapping ids with s1,"
        f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
    )
    return s1.loc[ids].sort_index(), s2.loc[ids].sort_index()


def filter_sort_index_many(
    inputs: List[S1], max_filtered_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO
) -> List[S1]:
    """Filters the indices of the given list of series to match each other,
    then sorts the indices, then checks that we didn't filter too many indices
    before returning the filtered and sorted series.

    Arguments:
        inputs: List[Union[pd.DataFrame, pd.Series]] - the list of datasets to filter and sort

    Returns:
        List[Union[pd.DataFrame, pd.Series]] - the filtered and sorted datasets
    """
    assert len(inputs) > 0, "List must contain at least one element"
    ids = inputs[0].dropna().index
    for i in range(1, len(inputs)):
        ids = ids.intersection(inputs[i].dropna().index)
    result = [inputs[i].loc[ids].sort_index() for i in range(len(inputs))]
    # ensure we didn't filter too many ids
    for i in range(len(result)):
        assert len(result[i]) / len(inputs[i]) >= (1 - max_filtered_ratio), (
            f"inputs[{i}] does not have enough overlapping ids with the others,"
            f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
        )
    return result


def filter_sort_top_bottom(
    s: pd.Series, top_bottom: int, return_concatenated: bool = True
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """Filters the series according to the top n and bottom n values
    then sorts the index and returns the filtered and sorted series.

    Arguments:
        s: pd.Series - the data to filter and sort
        top_bottom: int - the number of top n and bottom n values to keep

    Returns:
        pd.Series - the filtered and sorted data
    """
    tb_idx = np.argsort(s, kind="stable")
    bot = s.iloc[tb_idx[:top_bottom]]
    top = s.iloc[tb_idx[-top_bottom:]]
    if return_concatenated:
        return pd.concat([top, bot]).sort_index()
    else:
        return top.sort_index(), bot.sort_index()


def rank(df: pd.DataFrame, method: str = "average") -> pd.DataFrame:
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
    """Rank columns, breaking ties by index."""
    return rank(df, "first")


def tie_kept_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank columns, but keep ties."""
    return rank(df, "average")


def min_max_normalize(s: pd.Series) -> pd.Series:
    """Scale a series to be between 0 and 1."""
    return (s - s.min()) / (s.max() - s.min())


def variance_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale a df such that all columns have std == 1."""
    return df / np.std(df, axis=0)


def weight_normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Scale a df such that all columns have absolute value sum == 1."""
    return df / df.abs().sum(axis=0)


def center(df: pd.DataFrame) -> pd.DataFrame:
    """Shift the df such that all columns have mean == 0."""
    return df - df.mean()


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
        predictions = filter_sort_top_bottom(predictions, top_bottom)
        target, predictions = filter_sort_index(
            target, predictions, (1 - top_bottom / len(target))
        )
    validate_indices(target, predictions)
    return target.corr(predictions, method="pearson")


def sharpe_ratio(s: pd.Series) -> float:
    # calculate the sharpe ratio of a series
    return np.mean(s) / np.std(s)


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
    """Gaussianize each column of a pandas DataFrame using a normal percent point func.
    Effectively scales each column such that mean == 0 and std == 1.

    Arguments:
        df: pd.DataFrame - the data to gaussianize

    Returns:
        pd.DataFrame - the gaussianized data
    """
    assert np.array_equal(df.index.sort_values(), df.index)
    return df.apply(lambda series: stats.norm.ppf(series))


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


def correlation_contribution(
    predictions: pd.DataFrame,
    meta_model: pd.Series,
    live_targets: pd.Series,
    top_bottom: Optional[int] = None,
) -> pd.Series:
    """Calculate how much the given predictions contribute to the
    given Meta Model's correlation with the target.

    Then calculate contributive correlation by:
    1. tie-kept ranking each prediction and the meta model
    2. gaussianizing each prediction and the meta model
    3. orthogonalizing each prediction wrt the meta model
    4. dot product the orthogonalized predictions and the targets
       then normalize by the length of the target (equivalent to covariance)

    This is 100% correlated with the following formula:
    pearson_corr(
        live_targets, 0.999 * meta_model + 0.001 * predictions
    ) - pearson_corr(
        live_targets, meta_model
    )

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        meta_model: pd.Series - the meta model to evaluate against
        live_targets: pd.Series - the live targets to evaluate against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        pd.Series - the resulting contributive correlation
                    scores for each column in predictions
    """
    # filter and sort preds, mm, and targets wrt each other
    meta_model, predictions = filter_sort_index(meta_model, predictions)
    live_targets, predictions = filter_sort_index(live_targets, predictions)
    live_targets, meta_model = filter_sort_index(live_targets, meta_model)

    # rank and normalize meta model and predictions so mean=0 and std=1
    p = gaussian(tie_kept_rank(predictions)).values
    m = gaussian(tie_kept_rank(meta_model.to_frame()))[meta_model.name].values

    # orthogonalize predictions wrt meta model
    neutral_preds = orthogonalize(p, m)

    # convert target to buckets [-2, -1, 0, 1, 2]
    if (live_targets >= 0).all() and (live_targets <= 1).all():
        live_targets = live_targets * 4
    live_targets -= live_targets.mean()

    if top_bottom is not None and top_bottom > 0:
        # filter each column to its top and bottom n predictions
        neutral_preds_df = pd.DataFrame(
            neutral_preds, columns=predictions.columns, index=predictions.index
        ).apply(lambda p: filter_sort_top_bottom(p, top_bottom))
        # create a dataframe for targets to match the filtered predictions
        live_targets = (
            neutral_preds_df.apply(
                lambda p: filter_sort_index(
                    p,
                    live_targets,
                    (1 - top_bottom / len(live_targets)),
                )[1]
            )
            .fillna(0)
            .T.values
        )
        # fillna with 0 so we don't get NaNs in the dot product
        neutral_preds = neutral_preds_df.fillna(0).values

    # multiply target and neutralized predictions
    # this is equivalent to covariance b/c mean = 0
    mmc = live_targets @ neutral_preds
    if top_bottom is not None and top_bottom > 0:
        # only the diagonal is the proper score
        mmc = np.diag(mmc) / (top_bottom * 2)
    else:
        mmc /= len(live_targets)

    return pd.Series(mmc, index=predictions.columns)


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


def numerai_corr(
    predictions: pd.DataFrame,
    targets: pd.Series,
    max_filtered_index_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO,
    top_bottom: Optional[int] = None,
) -> pd.Series:
    """Calculates the canonical Numerai correlation.
    1. Re-center the target on 0
    2. filter and sort indices
    3. apply tie_kept_rank__gaussianize__pow_1_5 to the predictions
    4. raise the targets to the 1.5 power
    5. calculate the pearson correlation between the predictions and targets.

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        targets: pd.Series - the live targets to evaluate against
        max_filtered_index_ratio: float - the maximum ratio of indices that can be dropped
                                          when matching up the targets and predictions
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        pd.Series - the resulting correlation scores for each column in predictions
    """
    targets = center(targets)
    targets, predictions = filter_sort_index(
        targets, predictions, max_filtered_index_ratio
    )
    predictions = tie_kept_rank__gaussianize__pow_1_5(predictions)
    targets = power(targets.to_frame(), 1.5)[targets.name]
    scores = predictions.apply(
        lambda sub: pearson_correlation(targets, sub, top_bottom)
    )
    return scores


def feature_neutral_corr(
    predictions: pd.DataFrame,
    features: pd.DataFrame,
    targets: pd.Series,
    top_bottom: Optional[int] = None,
):
    """Calculates the canonical Numerai feature-neutral correlation.
    1. neutralize predictions relative to the features
    2. calculate the numerai_corr between the neutralized predictions and targets

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        features: pd.DataFrame - the features to neutralize the predictions against
        targets: pd.Series - the live targets to evaluate against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        pd.Series - the resulting correlation scores for each column in predictions
    """
    neutral_preds = tie_kept_rank__gaussianize__neutralize__variance_normalize(
        predictions, features
    )
    return numerai_corr(neutral_preds, targets, top_bottom=top_bottom)


def max_feature_correlation(
    s: pd.Series,
    features: pd.DataFrame,
    top_bottom: Optional[int] = None,
) -> Tuple[str, float]:
    """Calculates the maximum correlation between the given series and each feature
    and returns the name of the feature and the correlation with that feature.

    Arguments:
        s: pd.Series - the series to calculate correlations against
        features: pd.DataFrame - the features to calculate correlations against
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        Tuple[str, float] - the name of the feature with the highest correlation
                            and the correlation with that feature
    """
    feature_correlations = features.apply(
        lambda f: pearson_correlation(f, s, top_bottom)
    )
    feature_correlations = np.abs(feature_correlations)
    max_feature = feature_correlations.idxmax()
    max_corr = feature_correlations[max_feature]
    return max_feature, max_corr


def generate_neutralized_weights(
    predictions: pd.Series,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
) -> pd.Series:
    neutral_preds = predictions - (
        neutralizers @ (neutralizers.T @ (sample_weights * predictions))
    )
    neutral_weights = neutral_preds * sample_weights
    return neutral_weights


def alpha(
    predictions: pd.DataFrame,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
    targets: pd.Series,
) -> pd.Series:
    """Calculates the "alpha" score:
        - rank, normalize, and power the signal
        - convert signal into neutralized weights
        - multiplying the weights by the targets

    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        neutralizers: pd.DataFrame - the neutralization columns
        sample_weights: pd.Series - the universe sampling weights
        targets: pd.Series - the live targets to evaluate against
    """
    targets = center(targets)
    assert not predictions.isna().any().any(), "Predictions contain NaNs"
    assert not neutralizers.isna().any().any(), "Normalization factors contain NaNs"
    assert not sample_weights.isna().any(), "Weights contain NaNs"
    predictions, neutralizers, sample_weights, targets = filter_sort_index_many(
        [predictions, neutralizers, sample_weights, targets]
    )
    ranked_preds = tie_kept_rank__gaussianize__pow_1_5(predictions)
    weights = ranked_preds.apply(
        lambda s_prime: generate_neutralized_weights(
            s_prime, neutralizers, sample_weights
        )
    )
    alpha_scores = weights.apply(lambda w: w @ targets) / len(targets)
    return alpha_scores


def meta_portfolio_contribution(
    predictions: pd.DataFrame,
    stakes: pd.Series,
    neutralizers: pd.DataFrame,
    sample_weights: pd.Series,
    targets: pd.Series,
) -> pd.Series:
    """Calculates the "meta portfolio" score:
        - rank, normalize, and power each signal
        - convert each signal into neutralized weights
        - generate the stake-weighted portfolio
        - calculate the gradient of the portfolio w.r.t. the stakes
        - multiplying the weights by the targets
    Arguments:
        predictions: pd.DataFrame - the predictions to evaluate
        stakes: pd.Series - the stakes to use as weights
        neutralizers: pd.DataFrame - the neutralization columns
        sample_weights: pd.Series - the universe sampling weights
        targets: pd.Series - the live targets to evaluate against
    """
    targets = center(targets)
    assert not predictions.isna().any().any(), "Predictions contain NaNs"
    assert not neutralizers.isna().any().any(), "Normalization factors contain NaNs"
    assert not sample_weights.isna().any(), "Weights contain NaNs"
    predictions, neutralizers, sample_weights, targets = filter_sort_index_many(
        [predictions, neutralizers, sample_weights, targets]
    )
    stake_weights = weight_normalize(stakes.fillna(0))
    assert np.isclose(stake_weights.sum(), 1), "Stakes must sum to 1"
    weights = tie_kept_rank__gaussianize__pow_1_5(predictions).apply(
        lambda s_prime: generate_neutralized_weights(
            s_prime, neutralizers, sample_weights
        )
    )
    w = weights[stakes.index].values
    s = stake_weights.values
    t = targets.values
    swp = w @ s
    swp = swp - swp.mean()
    l1_norm = np.sum(np.abs(swp))
    l1_norm_squared = np.power(l1_norm, 2)
    swp_sign = np.sign(swp)
    swp_alpha = np.dot(swp, t)
    directional_gradient = l1_norm * t - swp_sign * swp_alpha
    jacobian_vector_product = directional_gradient.reshape(-1, 1) / l1_norm_squared
    centered_jacobian = jacobian_vector_product - jacobian_vector_product.mean()
    mpc = (w.T @ centered_jacobian).squeeze()
    return pd.Series(mpc, index=stakes.index)
